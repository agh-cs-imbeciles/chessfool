import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SimpleNNUE(nn.Module):
    def __init__(
        self,
        input_size=768,
        hidden1_size=512,
        hidden2_size=128,
    ) -> None:
        super().__init__()

        self.input_layer = nn.Linear(input_size, hidden1_size, bias=False)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(hidden1_size, hidden2_size)
        self.output_layer = nn.Linear(hidden2_size, 1)

    def forward(self, x: torch.Tensor) -> float:
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)

        return self.output_layer(x)


class ChessDataset(Dataset):
    def __init__(self, board_states, cp_losses):
        self.X = board_states
        self.y = cp_losses

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(
    model: SimpleNNUE,
    dataset: ChessDataset,
    validation_dataset: ChessDataset,
    epochs=10,
    batch_size=32,
    lr=0.001,
    device="cpu",
) -> None:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.to(device)
    model.train()

    for epoch in range(epochs):  # Adjust epochs
        model.train()
        running_loss = 0.0
        validation_loss = 0.0

        for inputs, targets in validation_dataloader:
            inputs = inputs.float().to(device)

            # Reshape to (batch_size, 1)
            targets = targets.float().view(-1, 1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            validation_loss += loss.item()
        for inputs, targets in dataloader:
            inputs = inputs.float().to(device)

            # Reshape to (batch_size, 1)
            targets = targets.float().view(-1, 1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Track the best model based on validation loss
            if (
                epoch == 0
                or validation_loss / len(validation_dataloader) < best_val_loss
            ):
                best_val_loss = validation_loss / len(validation_dataloader)
                best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch + 1},", f"Loss: {running_loss / len(dataloader)}")
        print(
            f"Epoch {epoch + 1},",
            f"Validation loss: {validation_loss / len(validation_dataloader)}",
        )

    print(f"Loaded model with validation loss: {best_val_loss}")
    model.load_state_dict(best_model_state)
