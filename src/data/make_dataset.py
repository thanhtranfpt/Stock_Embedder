from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from tqdm import tqdm
import torch


class StockDataset(Dataset):
    def __init__(self, data: list) -> None:
        super().__init__()

        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)


def create_time_series_data(stock_df: pd.DataFrame, features: list, window_size: int, logger = None):
    # Check
    if not all(col in stock_df.columns for col in ['Date', 'Symbol']):
        if logger:
            logger.error(f"One or both of 'Date' and 'Symbol' are missing from stock_df.")
        
        raise Exception(f"One or both of 'Date' and 'Symbol' are missing from stock_df.")
    
    time_series_data = []
    symbols = stock_df['Symbol'].unique()

    for symbol in tqdm(symbols):
        # Lọc dữ liệu cho mỗi symbol
        symbol_data = stock_df[stock_df['Symbol'] == symbol].sort_values(by='Date').reset_index(drop=True)[features].values
        
        # Tạo cửa sổ thời gian cho từng symbol
        for i in range(len(symbol_data) - window_size):
            window_data = symbol_data[i : i + window_size]
            
            time_series_data.append(window_data)
            
    return time_series_data


def create_dataset(cfg: dict, logger = None):
    """
    Args:
        cfg (dict):     {
                            'stock_file': str,
                            'create_new_scaler': bool = True,
                            'scaler_load_path': None  # if create_new_scaler. Else, str = 'scaler.pkl',
                            'ts_size': int = 24,
                            'scaler_save_path': str = 'scaler.pkl'
                        }
    """

    stock_df = pd.read_csv(filepath_or_buffer=cfg['stock_file'], encoding='UTF-8')

    # Check
    if not all(col in stock_df.columns for col in ['Date', 'Symbol']):
        if logger:
            logger.error(f"One or both of 'Date' and 'Symbol' are missing from stock_df.")

        raise Exception(f"One or both of 'Date' and 'Symbol' are missing from stock_df.")

    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    if cfg['create_new_scaler']:
        features = stock_df.drop(columns = ['Date', 'Symbol']).columns.values

        # Tạo scaler
        scaler = MinMaxScaler()

        # Huấn luyện scaler trên toàn bộ dữ liệu stock
        scaler.fit(stock_df[features])
    
    else:
        # Load scaler đã lưu
        scaler = joblib.load(filename=cfg['scaler_load_path'])

    # Lưu scaler để sử dụng sau này
    joblib.dump(scaler, filename=cfg['scaler_save_path'])
    
    # Chuẩn hóa dữ liệu train bằng scaler
    stock_df[scaler.feature_names_in_] = scaler.transform(stock_df[scaler.feature_names_in_])

    time_series_data = create_time_series_data(stock_df=stock_df, features=scaler.feature_names_in_, window_size=cfg['ts_size'], logger=logger)

    # Create dataset
    dataset = StockDataset(data=time_series_data)
    

    return dataset


def create_dataloaders(dataset: Dataset, cfg: dict):
    """
    Args:
        cfg (dict):     {
                            'split_ratio': {
                                'train': 0.8,  # 80% for training
                                'val': 0.1,  # 10% for validation
                                'test': 0.1  # 10% for testing
                            },
                            'batch_size': 32
                        }
    """

    # Random split
    train_size = int(cfg['split_ratio']['train'] * len(dataset))
    val_size = int(cfg['split_ratio']['val'] * len(dataset))
    test_size = len(dataset) - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, val_size, test_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=cfg['batch_size'], shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=cfg['batch_size'], shuffle=False)

    
    return train_dataloader, val_dataloader, test_dataloader