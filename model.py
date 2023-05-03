import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.to(self.device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class Model():
    def __init__(self, input_size, hidden_size, num_layers, output_size, device, lr=0.05, num_epochs=240):
        self.device = device
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size, device).to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        x_train_np = x_train.drop('patientunitstayid', axis=1).values.astype(np.float64)
        y_train_np = y_train.values.ravel().astype(np.float64)
        x_val_np = x_val.drop('patientunitstayid', axis=1).values.astype(np.float64)
        y_val_np = y_val.values.ravel().astype(np.float64)

        x_train, y_train, x_val, y_val = torch.tensor(x_train_np).float().to(self.device), torch.tensor(
            y_train_np).float().to(self.device), torch.tensor(x_val_np).float().to(self.device), torch.tensor(
            y_val_np).float().to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(x_train.unsqueeze(1))
            loss = self.criterion(outputs.squeeze(), y_train)
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(x_val.unsqueeze(1))
                val_loss = self.criterion(val_outputs.squeeze(), y_val)

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        return val_loss.item()

    def predict_proba(self, x):
        # Check for non-numeric columns
        non_numeric_columns = x.select_dtypes(include=['object']).columns
        if not non_numeric_columns.empty:
            print(f"Warning: Non-numeric columns found: {non_numeric_columns}. Please preprocess the data accordingly.")
            return None

        # Check for missing values
        if x.isna().any():
            print("Warning: Missing values found. Please preprocess the data accordingly.")
            return None

        x = torch.tensor(x.values, dtype=torch.float32).to(self.device)
        probas = self.model(x).cpu().numpy()
        return probas
