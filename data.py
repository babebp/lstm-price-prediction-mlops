import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

class OilPriceDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, seq_length=60, batch_size=32, target_col='close'):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.target_col = target_col
        self.scaler = MinMaxScaler()
        self.feature_cols = None
        self.target_idx = None
        

    def create_sequences(self, data):
        """Convert data into sequences for time series prediction"""
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i+self.seq_length])
            # Target is the next value of the target column
            y.append(data[i+self.seq_length, self.target_idx])
            
        return np.array(X), np.array(y).reshape(-1, 1)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            df = pd.read_csv(self.csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp')  # Ensure data is sorted by time
            
            # Store column names and target column index
            self.feature_cols = [col for col in df.columns if col != 'timestamp']
            self.target_idx = self.feature_cols.index(self.target_col)
            
            # Scale the data
            data = df[self.feature_cols].values
            self.scaler = self.scaler.fit(data)
            self.scaled_data = self.scaler.transform(data)
            
            # Save split point for later
            split_idx = int(len(self.scaled_data) * 0.8)

            # Create sequences for training and validation
            X_train, y_train = self.create_sequences(self.scaled_data[:split_idx])
            X_val, y_val = self.create_sequences(self.scaled_data[split_idx - self.seq_length:])
            
            # Convert to PyTorch tensors
            self.X_train = torch.tensor(X_train, dtype=torch.float32)
            self.y_train = torch.tensor(y_train, dtype=torch.float32)
            self.X_val = torch.tensor(X_val, dtype=torch.float32)
            self.y_val = torch.tensor(y_val, dtype=torch.float32)
            
            # Create tensor datasets
            self.train_data = TensorDataset(self.X_train, self.y_train)
            self.val_data = TensorDataset(self.X_val, self.y_val)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=False  # No shuffling for time series
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def get_feature_dim(self):
        """Return the number of features"""
        return len(self.feature_cols)
    
    def get_scaler(self):
        """Return the scaler for inverse transformations"""
        return self.scaler
    
    def get_target_idx(self):
        """Return the index of the target column"""
        return self.target_idx
        
    def get_feature_names(self):
        """Return the names of the features"""
        return self.feature_cols


if __name__ == "__main__":
    data_module = OilPriceDataModule('usoil.csv')
    data_module.prepare_data()
    data_module.setup()
    
    # Print some information about the data
    print(f"Feature dimension: {data_module.get_feature_dim()}")
    print(f"Feature names: {data_module.get_feature_names()}")
    print(f"Target index: {data_module.get_target_idx()}")
    print(f"Target column: {data_module.feature_cols[data_module.get_target_idx()]}")
    
    # Check the shape of the training data
    batch = next(iter(data_module.train_dataloader()))
    x, y = batch
    print(f"Input shape: {x.shape}")  # Should be [batch_size, seq_length, n_features]
    print(f"Target shape: {y.shape}")  # Should be [batch_size, 1]