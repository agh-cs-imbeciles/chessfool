import sys
import csv
import torch
from engine.utils import encode_board

MATE_VALUE = 8191
BATCH_SIZE = 100_000


def process_csv(input_file: str):
    with open(input_file) as input_csv_file:
        reader = csv.reader(input_csv_file)
        next(reader)  # Skip header

        batch_boards: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []

        for idx, (fen, centipawn_loss) in enumerate(reader, 1):
            if centipawn_loss.startswith("#"):
                mate_score = int(centipawn_loss[1:].lstrip("-"))
                sign = -1 if "-" in centipawn_loss else 1
                loss = sign * (MATE_VALUE - mate_score)
            else:
                loss = int(centipawn_loss)

            batch_boards.append(encode_board(fen))
            batch_labels.append(torch.tensor([loss], dtype=torch.float32))

            if idx % BATCH_SIZE == 0:
                yield batch_boards, batch_labels
                batch_boards = []
                batch_labels = []

        if batch_boards and batch_labels:
            yield batch_boards, batch_labels


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: "
            f"python {sys.argv[0]} <input_csv_path> <output_path_prefix>"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_prefix = sys.argv[2]

    for i, (boards, labels) in enumerate(process_csv(input_file)):
        output_file = f"{output_prefix}_{i + 1}.pt"
        torch.save({
            "boards": torch.stack(boards),
            "labels": torch.stack(labels),
        }, output_file)
        print(f"Saved {output_file}")
