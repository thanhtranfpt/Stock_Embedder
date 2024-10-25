import json
import argparse
import torch
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from src.utils.logger_config import get_logger
from src.models.stock_embedder import StockEmbedderLightning
import joblib


def load_config(config_file: str):
    with open(file=config_file, mode='r', encoding='UTF-8') as file:
        cfg = json.load(file)
    
    return cfg


def main(cfg: dict, stock_df: pd.DataFrame, verbose: bool = True, logger = None):
    """
        Args:
            stock_df (pd.DataFrame):    --> columns including scaler.feature_names_in_
                                        --> original - NOT normalized
                                        --> n_rows >= model.config['ts_size']
                                        --> of 1 Symbol

    """

    if verbose:
        if not logger:
            raise Exception('logger must be provided if verbose is True.')
    
    # Load scaler đã lưu
    scaler = joblib.load(filename=cfg['scaler_file'])

    # Chuẩn hóa dữ liệu bằng scaler
    stock_df[scaler.feature_names_in_] = scaler.transform(stock_df[scaler.feature_names_in_])

    # Get stock data
    stock_data = stock_df.sort_values(by='Date').reset_index(drop=True)[scaler.feature_names_in_].values  # shape: (n_rows, n_features)

    # Load Model
    stock_embedder = StockEmbedderLightning.load_from_checkpoint(
        checkpoint_path=cfg['checkpoint_path'],
        is_training = False,
        strict=False
    )

    # Bạn có thể tùy chọn chuyển mô hình sang chế độ eval
    stock_embedder.eval()

    if verbose:
        logger.info(f"stock_embedder.config = {stock_embedder.config}")
        logger.info(f'stock_embedder.is_training = {stock_embedder.is_training}')
    
    # Dự đoán trên dữ liệu mới

    stock_data = stock_data[ - stock_embedder.config['model']['ts_size'] : ]
    stock_data = torch.tensor(stock_data, dtype=torch.float32)
    stock_data = stock_data.unsqueeze(dim=0)  # Add batch dimension: shape = (1, ts_size, n_features)

    with torch.no_grad():
        stock_data = stock_data.to(stock_embedder.device)
        stock_embedding = stock_embedder.get_embedding(stock_data, embedding_used=cfg['embedding_used'])
    
    stock_embedding = stock_embedding.cpu().numpy()

    if verbose:
        logger.info(f'stock_embedding.shape = {stock_embedding.shape}')
    

    return stock_embedding


if __name__ == '__main__':
    # Stock DataFrame
    stock_df = pd.DataFrame({
        'Date': pd.date_range(start='2024-04-30', periods=100),
        'Adj_Close': np.random.uniform(100, 300, size=100),
        'Close': np.random.uniform(100, 300, size=100),
        'High': np.random.uniform(100, 300, size=100),
        'Low': np.random.uniform(100, 300, size=100),
        'Open': np.random.uniform(100, 300, size=100),
        'Volume': np.random.uniform(100, 300000, size=100),
        **{f: np.random.uniform(100, 300, size=100) for f in ['stoch', 'adx', 'bollinger_hband', 'mfi', 'rsi', 'ma', 'std', 'adl', 'williams', 'macd', 'obv', 'sar', 'ichimoku_a', 'ichimoku_b']}
    })

    # Tạo parser và thêm tham số cho file cấu hình
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=False, help='Path to config file (JSON format).',
                        default='config/predict_config.json')

    args = parser.parse_args()

    # Tải cấu hình từ file được chỉ định
    cfg = load_config(config_file=args.config_file)

    # Run
    logger = get_logger(name=__name__, log_file=os.path.join(cfg['output_dir'], 'logs.log'), mode='w')


    stock_embedding = main(cfg=cfg, stock_df=stock_df, verbose=cfg['verbose'], logger=logger)


    torch.save(stock_embedding, f=os.path.join(cfg['output_dir'], 'stock_embedding.pt'))

    logger.info('Finished.')