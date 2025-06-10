import torch
import torch.nn as nn
import chess
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SimpleNNUE(nn.Module):
    def __init__(self, input_size=768, hidden1_size=512, hidden2_size=32):
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
        self.X = [torch.tensor(fen, dtype=torch.float32) for fen in board_states]
        self.y = [torch.tensor(cp, dtype=torch.float32) for cp in cp_losses]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def encode_board(board: chess.Board | str) -> torch.Tensor:
    if isinstance(board, str):
        board = chess.Board(board)

    board_vector = torch.zeros(768)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_type = piece.piece_type
            piece_color = 0 if piece.color else 1
            index = piece_color * 64 * 6 + (piece_type-1) * 64 + square
            board_vector[index] = 1.0

    return board_vector

def train(model: SimpleNNUE, dataset_train: DataLoader, dataset_validate: DataLoader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataset_train:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # total_loss_validate = 0
        # for X_batch, y_batch in dataset_validate:
        #     with torch.no_grad():
        #         preds = model(X_batch)
        #         loss = criterion(preds, y_batch)
        #         total_loss_validate += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {total_loss/len(dataset_train):.4f}')
        # print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss_validate/len(dataset_validate):.4f}')
