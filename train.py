import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import OilPriceDataModule
from model import OilPriceLSTM


def main():
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Initialize data module
    oil_data = OilPriceDataModule(
        csv_path='usoil.csv',
        seq_length=60,  # Use 60 time steps of history
        batch_size=32,
        target_col='close'  # Predict closing price
    )
    
    # Prepare data (needs to be called before model creation to know input dimension)
    oil_data.prepare_data()
    oil_data.setup()
    
    # Get input dimension from data module
    input_dim = oil_data.get_feature_dim()
    
    # Initialize model
    oil_model = OilPriceLSTM(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.0001
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="./oil_models",
        filename="oil-lstm-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min"
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator='cpu',  # Updated from 'gpus' which is deprecated
        # devices=1 if torch.cuda.is_available() else 0,
        max_epochs=200,
        fast_dev_run=False,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10
    )
    
    # Train the model
    trainer.fit(oil_model, oil_data)
    
    # Print path to best model
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()