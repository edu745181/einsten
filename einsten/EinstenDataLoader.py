import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

class EinstenDataLoader(Dataset):
    def __init__(self, dataset_name, batch_size, tokenizer, max_len):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load the dataset
        self.df = pd.read_csv(f"{dataset_name}.csv")
        
        # Preprocess the data
        self.scaler = StandardScaler()
        self.df_scaled = self.scaler.fit_transform(self.df)
        
        # Split the data into training and validation sets
        self.train_df, self.val_df = self.split_data(self.df_scaled)
        
        # Create the dataloader
        self.dataloader = DataLoader(self.train_df, batch_size=batch_size, shuffle=True)
        
    def split_data(self, df):
        # Split the data into training and validation sets
        train_size = int(len(df) * 0.8)
        val_size = len(df) - train_size
        train_df, val_df = df[:train_size], df[train_size:]
        return train_df, val_df
    
    def __len__(self):
        return len(self.dataloader)
    
    def __getitem__(self, idx):
        # Get the batch from the dataloader
        batch = self.dataloader[idx]
        
        # Convert the batch to a numpy array
        batch = np.array(batch)
        
        # Pad the sequences to the maximum length
        batch = batch + (self.max_len - batch.shape[1]) * [0]
        
        # Return the padded batch
        return batch