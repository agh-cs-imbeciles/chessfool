
import chess
import pandas as pd
import torch
from typing import Iterator
import tracemalloc
import csv

tracemalloc.start()

def encode_board(board: chess.Board | str) -> torch.Tensor:
    if isinstance(board, str):
        board_real = chess.Board(board)
    else:
        board_real = board

    board_vector = torch.zeros(768)
    for square, piece in board_real.piece_map().items():
        if piece is not None:
            piece_type = piece.piece_type
            piece_color = 0 if piece.color else 1
            index = piece_color * 64 * 6 + (piece_type-1) * 64 + square
            board_vector[index] = 1.0
    del board_real
    return board_vector


def load_and_encode_csv(
    filepath: str, batch_size: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    boards_all = []
    points_all = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip the header row
        currnum = 0
        for row in reader:
            if len(boards_all) % batch_size == 0:
                print(f"Start loading of entries with indexes {currnum} to {currnum+batch_size}")
                currnum += batch_size
            board_str = row[0]
            point = int(float(row[1]))
            board_tensor = encode_board(board_str)
            point_tensor = torch.tensor([point], dtype=torch.float32)
            boards_all.append(board_tensor)
            points_all.append(point_tensor)



    return boards_all, points_all

