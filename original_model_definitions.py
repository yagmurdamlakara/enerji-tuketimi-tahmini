"""Definitions copied verbatim from makale.ipynb at 6d74b9b.
Final multi-seed factory uses corrected BiLSTM. No architecture changes.
The runner sets device explicitly before calling these functions.
"""
import random
import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPModel(nn.Module):

    def __init__(self, seq_len, input_size):

        super().__init__()

        flattened_size = seq_len * input_size

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        # Son zaman adımının çıktısı
        last_output = lstm_out[:, -1, :]

        prediction = self.fc(
            last_output
        )

        return prediction

class GRUModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        gru_out, _ = self.gru(x)

        # Son zaman adımındaki çıktı
        last_output = gru_out[:, -1, :]

        prediction = self.fc(last_output)

        return prediction

class BiLSTMModelCorrected(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64
    ):

        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hidden_size * 2,
            1
        )

    def forward(self, x):

        _, (h_n, _) = self.bilstm(x)

        # h_n şekli:
        # [2, batch, hidden_size]
        #
        # h_n[0] = ileri yönün son hidden state'i
        # h_n[1] = geri yönün son hidden state'i

        forward_hidden = h_n[0]
        backward_hidden = h_n[1]

        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        prediction = self.fc(combined)

        return prediction

def train_model(
    model,
    train_loader,
    val_loader,
    learning_rate=0.001,
    max_epochs=100,
    patience=10
):

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state = None

    patience_counter = 0

    for epoch in range(max_epochs):

        # ------------------
        # TRAIN
        # ------------------

        model.train()

        train_loss_sum = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            optimizer.step()

            train_loss_sum += (
                loss.item() * X_batch.size(0)
            )

        train_loss = (
            train_loss_sum /
            len(train_loader.dataset)
        )

        # ------------------
        # VALIDATION
        # ------------------

        model.eval()

        val_loss_sum = 0.0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)

                loss = criterion(
                    predictions,
                    y_batch
                )

                val_loss_sum += (
                    loss.item() * X_batch.size(0)
                )

        val_loss = (
            val_loss_sum /
            len(val_loader.dataset)
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # ------------------
        # EARLY STOPPING
        # ------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print(
                f"\nEarly stopping: "
                f"{epoch+1}. epoch"
            )

            break

    # En iyi validation ağırlıklarını geri yükle
    model.load_state_dict(best_state)

    model.to(device)

    return (
        model,
        train_losses,
        val_losses
    )

def predict_model(model, data_loader):

    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            X_batch = X_batch.to(device)

            output = model(X_batch)

            predictions.append(
                output.cpu().numpy()
            )

            actuals.append(
                y_batch.numpy()
            )

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    return predictions, actuals

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_model(model_name):

    if model_name == "MLP":

        model = MLPModel(
            seq_len=24,
            input_size=12
        )

    elif model_name == "LSTM":

        model = LSTMModel(
            input_size=12,
            hidden_size=64
        )

    elif model_name == "GRU":

        model = GRUModel(
            input_size=12,
            hidden_size=64
        )

    elif model_name == "BiLSTM":

        # Düzeltilmiş BiLSTM kullanılıyor
        model = BiLSTMModelCorrected(
            input_size=12,
            hidden_size=64
        )

    else:
        raise ValueError(
            f"Bilinmeyen model: {model_name}"
        )

    return model.to(device)
