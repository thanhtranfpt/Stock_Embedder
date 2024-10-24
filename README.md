# Stock_Embedder

Stock Embedder of The Finance Forecasting Project


# Guide to Training

**Step 1**. Install: `pip install -r requirements.txt`

**Step 2**. Config in the file: `config/train_config.json`

    ```json
    {
        "model_name": str = "Stock_Embedder_Lightning",
        "version": str = "",
        "training": {
            "resume": bool = true,
            "checkpoint_path": str = ""  # if resume training. Else, null
        },
        "trainer": {
            "min_epochs": int = 1,
            "max_epochs": int = 200,
            "max_steps": int = -1
        },
        "dataset": {
            "stock_file": str = "",
            "create_new_scaler": bool = true,
            "scaler_load_path": null  # if create_new_scaler. Else, str
        },
        "dataloaders": {
            "split_ratio": {
                "train": float = 0.8,
                "val": float = 0.1,
                "test": float = 0.1
            },
            "batch_size": int = 32
        },
        "stock_embedder_lightning": {
            "training": {
                "mode": int = 3  # if not resume training. Else, [Optional] = [Removed]
            },
            "model": {
                "z_dim": int = 6,
                "ts_size": int = 24,
                "mask_size": int = 1,
                "num_masks": int = 3,
                "hidden_dim": int = 12,
                "embed_dim": int = 6,
                "num_layers": int = 3,
                "num_embed": int = 32
            }  # if not resume training. Else, [Removed]
        },
        "output_dir": str = "",
        "verbose": bool = true
    }

**Step 3**. Run in terminal: `python src/train_model.py`


# Guide to Inference

**Step 1**. Install: `pip install -r requirements.txt`

**Step 2**. Config in the file: `config/predict_config.json`

    ```json
    {
        "checkpoint_path": str = "",
        "scaler_file": str = "",
        "embedding_used": str = "encoder",
        "output_dir": str = "",
        "verbose": bool = true
    }

**Step 3**. Run in terminal: `python src/predict_model.py`


# Models Trained

- **Author's pretrained**: https://drive.google.com/drive/folders/1wL1GAkzCax71vM_7tXYlpiSFcGNVb20F?usp=drive_link

- **Models_Notes**: https://docs.google.com/spreadsheets/d/1thZECwq45yi7cjM8rINQkrbYbnbAtw6wWsCQIW1g118/edit?usp=drive_link


# Processed Data

- **ver_2**
    - **Link**: https://drive.google.com/drive/folders/1_HPqG9OrtcBk6o4KLhqHuV8hnCztxUtT?usp=drive_link
    - **Source**: Derived from https://drive.google.com/drive/folders/1MNPU6IWEJxJgCCwGRgy-DsIvFI5SP7K8?usp=drive_link
    - **Symbols Included**: All S&P 500 companies (503 symbols)
    - **Data Time Range**: From January 4, 2010 to October 2, 2020
    - **Columns**: `['Date', 'Symbol', 'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume']`
    - **Comments**:

- **ver_3**
    - **Link**: https://drive.google.com/drive/folders/1LOC_t0--IWT3bLTAIjoaSVVRH__cgcKq?usp=drive_link
    - **Source**: Derived from https://drive.google.com/drive/folders/1MNPU6IWEJxJgCCwGRgy-DsIvFI5SP7K8?usp=drive_link
    - **Symbols Included**: All S&P 500 companies (503 symbols)
    - **Data Time Range**: From January 4, 2010 to October 2, 2020
    - **Technical Indicators**: Yes
    - **Columns**: `['Date', 'Symbol', 'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume']` + *Technical Indicators*
    - **Comments**:


# External Data

- **ver_2**
    - **Link**: https://drive.google.com/drive/folders/1MNPU6IWEJxJgCCwGRgy-DsIvFI5SP7K8?usp=drive_link
    - **Source**: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks
    - **Date Collected (Download Date)**: 2nd October 2024
    - **Data Time Range**: From January 4, 2010 to October 2, 2020
    - **Symbols Included**: All S&P 500 companies (503 symbols)
    - **Columns**:
        - **sp500_stocks.csv**: `['Date', 'Symbol', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']`
    - **Comments**: