import sys
import re
import csv
import chess


def process_csv(input_file: str, output_file: str) -> None:
    with (
        open(input_file) as input_csv_file,
        open(output_file, mode="w") as output_csv_file,
    ):
        reader = csv.reader(input_csv_file)
        writer = csv.writer(output_csv_file)

        # Write the CSV header to the output file, while omitting it when
        # reading it
        writer.writerow(next(reader))

        for fen, centipawn_loss in reader:
            board = chess.Board(fen)

            if board.turn == chess.BLACK:
                board.apply_mirror()
                centipawn_loss = re.sub(
                    r"[+-]",
                    lambda m: "-" if m[0] == "+" else "+",
                    centipawn_loss,
                )

            writer.writerow([board.fen(), centipawn_loss])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_csv> <output_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process_csv(input_file, output_file)
