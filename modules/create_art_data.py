import numpy as np
import pandas as pd
import os, json
import torch
from Stock_Embedder.src.models.stock_embedder import StockEmbedderLightning
from Stock_Embedder.src.data.make_dataset import create_dataset
from Stock_Embedder.modules.utils import load_data
from Stock_Embedder.modules.arguments import load_arguments
from Stock_Embedder.modules.generation import *

def load_config(config_file: str):
    with open(file=config_file, mode='r', encoding='UTF-8') as file:
        cfg = json.load(file)
    return cfg

def main():
    '''
    Args:
        [Model related] model_cfg: predict_config.json
        [Stock related] config_dir: stock_config.json'''
    home = os.getcwd()
    args = load_arguments(home)
    print("INITIALIZING ARGUMENTS SUCCESSFULLY")   
    
    model_cfg = load_config(args.model_cfg)
    
    # Load model
    stock_embedder = StockEmbedderLightning.load_from_checkpoint(
        checkpoint_path=model_cfg['checkpoint_path'],
        is_training = False,
        strict=False
    )

    model = stock_embedder.model    # StockEmbedding

    # Get data
    ori_data = load_data(args)  #--> shape = (bs, ts_size, z_dim)
    try:
        ori_data = torch.tensor(ori_data).float()
    except:
        ori_data = torch.tensor(ori_data.astype(np.float32), dtype=torch.float32)
    print("Original data: ", ori_data.shape)
    
    ra_data = random_average_generation(args, model, ori_data)
    ra_data = ra_data.detach().numpy()
    np.save(os.path.join(args.ra_dir, 'art_data.npy'), ra_data)
    print("Random average data was computed successfully.")
    
    cc_data = cross_concat_generation(args, model, ori_data)
    cc_data = cc_data.detach().numpy()
    np.save(os.path.join(args.cc_dir, 'art_data.npy'), cc_data)
    print("Cross concate data was computed successfully.")    
    
    ca_data = cross_average_generation(args, model, ori_data)
    ca_data = ca_data.detach().numpy()
    print("Cross average data was computed successfully.")  
    np.save(os.path.join(args.ca_dir, 'art_data.npy'), ca_data)

if __name__ == '__main__':
    main()
    
    

    