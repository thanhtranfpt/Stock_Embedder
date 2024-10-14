import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from src.models.utils import mask_it


class Encoder(nn.Module):
    def __init__(self, cfg: dict):
        """
        Args:
            cfg (dict):     {
                                'z_dim': int = 6,  # Number of Features
                                'hidden_dim': int = 12,
                                'num_layers': int = 3
                            }

        """

        super().__init__()

        self.rnn = nn.RNN(input_size=cfg['z_dim'],
                          hidden_size=cfg['hidden_dim'],
                          num_layers=cfg['num_layers'])
        
        self.fc = nn.Linear(in_features=cfg['hidden_dim'],
                            out_features=cfg['hidden_dim'])

    def forward(self, x):

        x_enc, _ = self.rnn(x)

        x_enc = self.fc(x_enc)

        return x_enc
    

class Decoder(nn.Module):
    def __init__(self, cfg: dict):
        """
        Args:
            cfg (dict):     {
                                'z_dim': int = 6,  # Number of Features
                                'hidden_dim': int = 12,
                                'num_layers': int = 3
                            }

        """

        super().__init__()

        self.rnn = nn.RNN(input_size=cfg['hidden_dim'],
                          hidden_size=cfg['hidden_dim'],
                          num_layers=cfg['num_layers'])
        
        self.fc = nn.Linear(in_features=cfg['hidden_dim'],
                            out_features=cfg['z_dim'])

    def forward(self, x_enc):

        x_dec, _ = self.rnn(x_enc)

        x_dec = self.fc(x_dec)

        return x_dec
    

class Interpolator(nn.Module):
    def __init__(self, cfg: dict):
        """
        Args:
            cfg (dict):     {
                                'ts_size': int = 24  # Time-Series size,
                                'total_mask_size': int = 3,
                                'hidden_dim': int = 12,
                            }

        """

        super().__init__()

        self.sequence_inter = nn.Linear(in_features=(cfg['ts_size'] - cfg['total_mask_size']),
                                        out_features=cfg['ts_size'])
        
        self.feature_inter = nn.Linear(in_features=cfg['hidden_dim'],
                                       out_features=cfg['hidden_dim'])

    def forward(self, x):
        """
            x.shape = (batch_size, vis_size, hidden_dim)

        """

        x = rearrange(x, 'b l f -> b f l')  # shape: tch_size, hidden_dim, vis_size)
        x = self.sequence_inter(x)  # shape: (batch_size, hidden_dim, ts_size)

        x = rearrange(x, 'b f l -> b l f')  # shape: (batch_size, ts_size, hidden_dim)
        x = self.feature_inter(x)  # shape: (batch_size, ts_size, hidden_dim)

        return x
    

class StockEmbeddingCore(nn.Module):
    def __init__(self, cfg: dict) -> None:

        """
        Args:
            cfg (dict):     {
                                'z_dim': int = 6,  # Number of Features
                                'ts_size': int = 24  # Time-Series size,
                                'mask_size': int = 1,
                                'num_masks': int = 3,
                                'hidden_dim': int = 12,
                                'embed_dim': int = 6,
                                'num_layers': int = 3,
                                'num_embed': int = 32,
                            }

        """
        
        super().__init__()

        self.config = cfg
        
        self.config['total_mask_size'] = self.config['num_masks'] * self.config['mask_size']
        
        self.encoder = Encoder(cfg=self.config)

        self.interpolator = Interpolator(cfg=self.config)

        self.decoder = Decoder(cfg=self.config)


        print('StockEmbeddingCore initialized')


    def forward_ae(self, x: torch.Tensor):
        """
            mae_pseudo_mask is equivalent to the Autoencoder
            There is no interpolator in this mode

        Args:
            x (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)

        """
        
        x_enc = self.encoder(x)

        x_dec = self.decoder(x_enc)
        
        return x_enc, x_dec

    
    def forward_mae(self, x: torch.Tensor, masks: torch.Tensor):
        """
            No mask tokens, using Interpolation in the latent space

        Args:
            x (torch.Tensor):       --> shape: (batch_size, ts_size, z_dim)
            masks (torch.Tensor)    --> shape: (batch_size, ts_size), với mỗi giá trị là True hoặc False (True nghĩa là bị mask)

        """
        
        x_vis = mask_it(x, masks=masks)  # (batch_size, vis_size, z_dim)

        x_enc = self.encoder(x_vis)  # (batch_size, vis_size, hidden_dim)

        x_inter = self.interpolator(x_enc)  # (batch_size, ts_size, hidden_dim)

        x_dec = self.decoder(x_inter)  # (batch_size, ts_size, z_dim)


        return x_enc, x_inter, x_dec