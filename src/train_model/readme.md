# Config

```json
{
    "model_name": "Stock_Embedder_Lightning",
    "training": {
        "resume": true,
        "checkpoint_path": ""  # if resume training. Else, null
    },
    "trainer": {
        "min_epochs": 1,
        "max_epochs": 200,
        "max_steps": -1
    },
    "dataset": {
        "stock_file": "",
        "create_new_scaler": true,
        "scaler_load_path": null  # if create_new_scaler. Else, str
    },
    "dataloaders": {
        "split_ratio": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1
        },
        "batch_size": 32
    },
    "stock_embedder_lightning": {
        "training": {
            "mode": 3  # if not resume training. Else, [Optional] = [Removed]
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
    "output_dir": "",
    "verbose": true
}