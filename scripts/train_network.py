
import sys
import csv

import torch
from sklearn.model_selection import train_test_split

from engine.utils import encode_board
from engine.nnue_network import SimpleNNUE
from engine.nnue_network import ChessDataset, train_model

MATE_VALUE = 8191


def load_and_encode_csv(
    filepath: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float, float]:
    boards_all: list[torch.Tensor] = []
    points_all: list[torch.Tensor] = []
    min_point = float("inf")
    max_point = float("-inf")

    with open(filepath) as input_csv_file:
        reader = csv.reader(input_csv_file)
        next(reader)  # Skip the header row

        for fen, centipawn_loss in reader:
            if centipawn_loss.startswith("#"):
                mate_score = int(centipawn_loss[1:].lstrip("-"))
                sign = -1 if "-" in centipawn_loss else 1
                loss = sign * (MATE_VALUE - mate_score)
            else:
                loss = int(centipawn_loss)

            boards_all.append(encode_board(fen))
            points_all.append(torch.tensor([loss], dtype=torch.float32))

            if loss < min_point:
                min_point = loss
            if loss > max_point:
                max_point = loss

    return boards_all, points_all, min_point, max_point


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <training_data>")
        sys.exit(1)

    training_data_path = sys.argv[1]

    network = SimpleNNUE()
    boards, points, y_min, y_max = load_and_encode_csv(training_data_path)
    y_tensor = torch.stack(points)
    y_normalized = [(val - y_min) / (y_max - y_min) for val in points]

    X_train, X_val, y_train, y_val = train_test_split(
        boards,
        y_normalized,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    train_dataset = ChessDataset(X_train, y_train)
    val_dataset = ChessDataset(X_val, y_val)

    train_model(network, train_dataset, val_dataset, batch_size=8192, epochs=10)

    torch.save(network.state_dict(), "nnue_model_weights.pth")
