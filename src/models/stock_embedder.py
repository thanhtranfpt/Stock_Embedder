import lightning as L
# import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from src.models.stock_embedding_core import StockEmbeddingCore
from src.models.utils import generate_random_masks


class StockEmbedderLightning(L.LightningModule):
    def __init__(self, cfg: dict, is_training: bool = False, override_cfg: dict = None, program_logger = None):
        """
        Args:
            cfg (dict):     {
                                'training': {
                                    'mode': int = 3  # 1: train_ae | 2: train_embed | 3: train_recon
                                },
                                'model': {
                                    'z_dim': int = 6,  # Number of Features
                                    'ts_size': int = 24  # Time-Series size,
                                    'mask_size': int = 1,
                                    'num_masks': int = 3,
                                    'hidden_dim': int = 12,
                                    'embed_dim': int = 6,
                                    'num_layers': int = 3,
                                    'num_embed': int = 32,
                                }
                            }

        """

        super().__init__()

        if override_cfg:
            cfg.update(override_cfg)

        self.config = cfg

        self.is_training = is_training
        
        self.program_logger = program_logger

        if self.is_training:
            self.save_hyperparameters({'cfg': cfg})
        
        self.model = StockEmbeddingCore(cfg=self.config['model'])

        self.criterion = nn.MSELoss(reduction='mean')


        print('StockEmbedderLightning initialized')

    
    def check_config(self):
        """
            *   Check config
        """

        if self.config['training']['mode'] not in [1, 2, 3]:
            raise Exception('training_mode must be: 1 or 2 or 3.')


    def get_embedding(self, x: torch.Tensor, embedding_used: str):
        """
            defines the prediction/inference actions
            
            *   INPUT:
                        x (torch.Tensor):       --> shape: (batch_size, ts_size, z_dim)
                                                --> NORMALIZED using scaler
                        embedding_used (str):   --> encoder | decoder
            *   OUTPUT:
                        stock_embedding -->  shape: (batch_size, ts_size, z_dim)

        """

        if embedding_used not in ['encoder', 'decoder']:
            raise Exception('embedding_used must be: encoder or decoder')

        self.model.eval()  # Đảm bảo mô hình ở chế độ đánh giá
        with torch.no_grad():  # Tắt tính toán gradient để tăng tốc độ và tiết kiệm bộ nhớ
            x_enc, x_dec = self.model.forward_ae(x)
        
        if embedding_used == 'encoder':
            return x_enc
        
        return x_dec
    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer
    

    def ae_step(self, x: torch.Tensor):
        """
            ae_step defines the step in train loop. (mode: train_ae)

            Args:

                x (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)
                                    --> NORMALIZED using scaler

        """

        batch_size, ts_size, z_dim = x.shape

        x_enc, x_dec = self.model.forward_ae(x)

        # Calculate loss
        loss = self.criterion(x_dec, x)


        return loss
    

    def embed_step(self, x: torch.Tensor):
        """
            embed_step defines the step in train loop. (mode: train_embed)

            Args:

                x (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)
                                    --> NORMALIZED using scaler

        """

        batch_size, ts_size, z_dim = x.shape

        random_masks = generate_random_masks(num_samples=batch_size, ts_size=ts_size, mask_size=self.config['model']['mask_size'], num_masks=self.config['model']['num_masks'])  # shape: (batch_size, ts_size)

        # Get the target x_ori_enc by Autoencoder
        self.model.eval()
        x_enc_ae, x_dec_ae = self.model.forward_ae(x)
        x_enc_ae = x_enc_ae.clone().detach()  # shape: (batch_size, ts_size, hidden_dim)

        self.model.train()

        x_enc_mae, x_inter_mae, x_dec_mae = self.model.forward_mae(x, masks=random_masks)

        # Only calculate loss for those being masked
        x_inter_mae_masked = x_inter_mae[random_masks].reshape(batch_size, -1, self.config['model']['hidden_dim'])
        x_enc_ae_masked = x_enc_ae[random_masks].reshape(batch_size, -1, self.config['model']['hidden_dim'])

        loss = self.criterion(x_inter_mae_masked, x_enc_ae_masked)

        # # By annotate lines above, we take loss on all patches
        # loss = self.criterion(x_inter_mae, x_enc_ae)  # embed_loss


        return loss
    

    def recon_step(self, x: torch.Tensor):
        """
            recon_step defines the step in train loop. (mode: train_recon)

            Args:

                x (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)
                                    --> NORMALIZED using scaler

        """

        batch_size, ts_size, z_dim = x.shape

        random_masks = generate_random_masks(num_samples=batch_size, ts_size=ts_size, mask_size=self.config['model']['mask_size'], num_masks=self.config['model']['num_masks'])  # shape: (batch_size, ts_size)

        x_enc, x_inter, x_dec = self.model.forward_mae(x, masks=random_masks)

        # Calculate loss
        loss = self.criterion(x_dec, x)


        return loss
    

    def training_step(self, batch, batch_idx):
        """
            training_step defines the train loop. It is independent of forward

            Args:
                batch (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)

        """

        if self.config['training']['mode'] == 1:
            loss = self.ae_step(x=batch)
        
        elif self.config['training']['mode'] == 2:
            loss = self.embed_step(x=batch)
        
        else:
            loss = self.recon_step(x=batch)
        

        self.log(name='Loss/Train', value=loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)


        return loss
    

    def validation_step(self, batch, batch_idx):
        """
            Args:
                batch (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)

        """

        if self.config['training']['mode'] == 1:
            loss = self.ae_step(x=batch)
        
        elif self.config['training']['mode'] == 2:
            loss = self.embed_step(x=batch)
        
        else:
            loss = self.recon_step(x=batch)
        

        self.log(name='Loss/Val', value=loss, prog_bar=True)


        return loss
    

    def test_step(self, batch, batch_idx):
        """
            Args:
                batch (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)

        """

        if self.config['training']['mode'] == 1:
            loss = self.ae_step(x=batch)
        
        elif self.config['training']['mode'] == 2:
            loss = self.embed_step(x=batch)
        
        else:
            loss = self.recon_step(x=batch)
        

        self.log(name='Loss/Test', value=loss, prog_bar=True)


        return loss
    
    
    def on_train_epoch_end(self):
        if self.program_logger:
            self.program_logger.info(f'Epoch: {self.current_epoch}')
        
            epoch_mean = torch.stack(self.training_step_outputs).mean()
            self.program_logger.info(f"training_epoch_mean: {epoch_mean}")
        
            # free up the memory
            self.training_step_outputs.clear()