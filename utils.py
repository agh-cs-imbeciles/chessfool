
import chess
import pandas as pd
import torch
from typing import Iterator


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
    chunk_iter = pd.read_csv(filepath, chunksize=batch_size)
    currnum = 0
    for chunk in chunk_iter:
        print(f"Start loading of entries with indexes {currnum} to {currnum+len(chunk)}")
        boards = chunk.iloc[:, 0].apply(encode_board).tolist()
        points = chunk.iloc[:, 1].astype(int).tolist()
        points = [torch.tensor([p], dtype=torch.float32) for p in points]
        boards_all.extend(boards)
        points_all.extend(points)
        print(f"Loaded {len(chunk)} entries")
        currnum += len(chunk)
    return boards_all, points_all

