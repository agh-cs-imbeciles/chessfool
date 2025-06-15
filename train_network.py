
from sklearn.model_selection import train_test_split
import sys
from utils import load_and_encode_csv
from nnue_network import SimpleNNUE
from torch.utils.data import DataLoader
from nnue_network import ChessDataset, train_model
import torch

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_network.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]

    y_min = -16000 # max,min in chessData.csv
    y_max = -y_min
    network = SimpleNNUE()
    for boards, points in load_and_encode_csv(filename, 1000000):
        y_tensor = torch.stack(points)
        y_normalized = [2 * (val - y_min) / (y_max - y_min) - 1 for val in points]

        X_train, X_val, y_train, y_val = train_test_split(
            boards,
            y_normalized,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
        train_dataset = ChessDataset(X_train, y_train)
        val_dataset = ChessDataset(X_val, y_val)
        train_model(network, train_dataset, val_dataset, batch_size=8086, device='cuda')

    torch.save(network.state_dict(), "nnue_model_weights.pth")
