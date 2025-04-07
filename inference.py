import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import OilPriceLSTM
from data import OilPriceDataModule


class OilPricePredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        # Load the trained model
        self.model = OilPriceLSTM.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        
        # Initialize data module for preprocessing
        self.data_module = OilPriceDataModule('usoil.csv')
        self.data_module.prepare_data()
        self.data_module.setup()
        
        # Get scaler and target index for inverse transformation
        self.scaler = self.data_module.get_scaler()
        self.target_idx = self.data_module.get_target_idx()
        self.seq_length = self.data_module.seq_length
        
    def preprocess_data(self, data_sequence):
        """Preprocess a sequence of data points for prediction"""
        # If data is a DataFrame, convert to numpy array
        if isinstance(data_sequence, pd.DataFrame):
            data_sequence = data_sequence.values
            
        # Scale the data
        scaled_data = self.scaler.transform(data_sequence)
        
        # Convert to tensor
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
        return tensor_data
    
    def postprocess_prediction(self, prediction):
        """Convert the scaled prediction back to original scale"""
        # Create a zero array with the same number of features as the original data
        inverse_pred = np.zeros((1, self.scaler.scale_.shape[0]))
        
        # Put the prediction value in the target column position
        inverse_pred[0, self.target_idx] = prediction.item()
        
        # Inverse transform
        original_scale_pred = self.scaler.inverse_transform(inverse_pred)[0, self.target_idx]
        return original_scale_pred
    
    def predict(self, data_sequence):
        """Make a prediction for the next time step"""
        # Ensure we have the right sequence length
        if len(data_sequence) < self.seq_length:
            raise ValueError(f"Input sequence must have at least {self.seq_length} time steps")
            
        # Use the last seq_length points
        recent_data = data_sequence[-self.seq_length:]
        
        # Preprocess
        tensor_data = self.preprocess_data(recent_data)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(tensor_data)
            
        # Convert back to original scale
        result = self.postprocess_prediction(prediction)
        
        return result
    
    def predict_next_n_days(self, data_sequence, n_steps=5):
        """Make predictions for the next n time steps"""
        predictions = []
        
        # Use a copy of the data to avoid modifying the original
        working_data = data_sequence.copy()
        
        for _ in range(n_steps):
            # Make a prediction for the next step
            next_pred = self.predict(working_data)
            predictions.append(next_pred)
            
            # Create a new data point with the predicted value
            last_known_point = working_data[-1].copy()
            new_point = last_known_point.copy()
            new_point[self.target_idx] = next_pred
            
            # Add this point to our working data for the next prediction
            working_data = np.vstack([working_data, new_point.reshape(1, -1)])
            
        return predictions
    
    def plot_prediction(self, data_sequence, n_steps=5):
        """Plot the original data and the predictions"""
        # Get predictions
        predictions = self.predict_next_n_days(data_sequence, n_steps)
        
        # Create time indices for the plot
        time_orig = np.arange(len(data_sequence))
        time_pred = np.arange(len(data_sequence), len(data_sequence) + n_steps)
        
        # Extract the target values from the original data
        orig_values = data_sequence[:, self.target_idx]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(time_orig, orig_values, 'b-', label='Historical Data')
        plt.plot(time_pred, predictions, 'r--', label='Predictions')
        plt.axvline(x=len(data_sequence)-1, color='g', linestyle='--', alpha=0.5, label='Prediction Start')
        plt.title('Oil Price Prediction')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return predictions


if __name__ == "__main__":
    # Initialize the predictor with the path to the best model
    predictor = OilPricePredictor("./oil_models/oil-lstm-epoch=140-val_loss=0.0000.ckpt")
    
    # Load the data
    data_module = OilPriceDataModule('usoil.csv')
    data_module.prepare_data()
    
    # Get the raw data
    df = pd.read_csv('usoil.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Get the numerical data for prediction
    feature_data = df[[col for col in df.columns if col != 'timestamp']].values
    
    # Get the last 120 data points
    recent_data = feature_data[-60:]
    
    # Make prediction for next 10 time steps
    predictions = predictor.predict_next_n_days(recent_data, n_steps=5)
    print(f"Predictions for next 10 time steps: {predictions}")
    
    # Plot the results
    predictor.plot_prediction(recent_data, n_steps=5)