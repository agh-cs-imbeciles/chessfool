import torch
import torch.nn as nn
import chess
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SimpleNNUE(nn.Module):
    def __init__(self, input_size=769, hidden1_size=512, hidden2_size=128):
        super(SimpleNNUE, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden1_size, bias=False)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(hidden1_size, hidden2_size)
        self.output_layer = nn.Linear(hidden2_size, 1)

    def forward(self, x):
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


def encode_board(board: chess.Board | str) -> torch.Tensor:
    if isinstance(board, str):
        board = chess.Board(board)

    board_vector = torch.zeros(769)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_type = piece.piece_type
            piece_color = 0 if piece.color else 1
            index = piece_color * 64 * 6 + (piece_type-1) * 64 + square
            board_vector[index] = 1.0
    board_vector[768] = 0.0 if board.turn else 1.0

    return board_vector

def train_model(model: SimpleNNUE, dataset: ChessDataset, validation_dataset: ChessDataset, epochs=10, batch_size=32, lr=0.001, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    validation = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):  # Adjust epochs
        model.train()
        running_loss = 0.0
        validation_loss = 0.0

        for inputs, targets in validation:
            inputs = inputs.float().to(device)
            targets = targets.float().view(-1, 1).to(device)  # Reshape to (batch_size, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            validation_loss += loss.item()
        for inputs, targets in dataloader:
            inputs = inputs.float().to(device)
            targets = targets.float().view(-1, 1).to(device) # Reshape to (batch_size, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")
        print(f"Epoch {epoch+1}, Validation loss: {validation_loss / len(validation)}")
