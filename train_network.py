
from sklearn.model_selection import train_test_split
import sys
from utils import load_and_encode_csv
from nnue_network import SimpleNNUE
from torch.utils.data import DataLoader
from nnue_network import ChessDataset, train
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_network.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    boards, points = load_and_encode_csv(filename)

    X_train, X_val, y_train, y_val = train_test_split(
        boards,
        points,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    train_dataset = ChessDataset(X_train, y_train)
    val_dataset = ChessDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    network = SimpleNNUE()
    train(network, train_loader, val_loader, epochs=10, lr=0.1)
