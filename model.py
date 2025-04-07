import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


class OilPriceLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=128, num_layers=4, dropout=0.2, learning_rate=0.001):
        super(OilPriceLSTM, self).__init__()
        self.save_hyperparameters()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, 1)
        self.validation_step_outputs = []
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use only the last time step's output
        out = self.fc(lstm_out[:, -1, :])
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        # Calculate additional metrics
        rmse = torch.sqrt(val_loss)
        mae = F.l1_loss(y_hat, y)
        
        # Log metrics
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)

        self.validation_step_outputs.append({"val_loss": val_loss, "y_pred": y_hat, "y_true": y})

    
    def on_validation_epoch_end(self):
        
        # Aggregate all predictions and actual values
        y_pred = torch.cat([x["y_pred"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        y_true = torch.cat([x["y_true"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        
        # Calculate overall metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Log epoch-level metrics
        self.log("val_epoch_mse", mse)
        self.log("val_epoch_rmse", rmse)
        self.log("val_epoch_mae", mae)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }