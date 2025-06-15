
import chess
import pandas as pd
import torch
from typing import Iterator

max_val = 16000 # max,min in chessData.csv
min_val = -max_val

# def find_min_max_in_csv(filepath: str) -> tuple[int, int]:
#     min_value = float('inf')
#     max_value = float('-inf')
#     for chunk in pd.read_csv(filepath, chunksize=1000):
#         values = chunk.iloc[:, 1].astype(str).apply(points_decode)
#         min_chunk = values.min()
#         max_chunk = values.max()
#         if min_chunk < min_value:
#             min_value = min_chunk
#         if max_chunk > max_value:
#             max_value = max_chunk
#     return min_value, max_value

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

def points_decode(val: str) -> int:
    if val[0] == '#':
        return max_val if val[1] != 0 else min_val
    return int(val)

def load_and_encode_csv(
    filepath: str, batch_size: int
) -> Iterator[tuple[list[torch.Tensor], list[torch.Tensor]]]:
    chunk_iter = pd.read_csv(filepath, chunksize=batch_size)
    for chunk in chunk_iter:
        boards = chunk.iloc[:, 0].apply(encode_board).tolist()
        points = chunk.iloc[:, 1].astype(str).apply(points_decode).tolist()
        points = [torch.tensor([p], dtype=torch.float32) for p in points]
        yield boards, points

