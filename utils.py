
import chess
import pandas as pd
import torch
from typing import Iterator
import csv


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

def encode_fen_to_nnue(fen: str) -> torch.Tensor:
    """
    Encodes a FEN string into a torch tensor representing the NNUE input vector,
    without using the chess.Board object.
    """
    nnue_vector = torch.zeros(768)
    piece_map = {
        'P': (0, 0), 'N': (0, 1), 'B': (0, 2), 'R': (0, 3), 'Q': (0, 4), 'K': (0, 5),
        'p': (1, 0), 'n': (1, 1), 'b': (1, 2), 'r': (1, 3), 'q': (1, 4), 'k': (1, 5)
    }
    rows = fen.split(' ')[0].split('/')
    for rank, row in enumerate(rows):
        file = 0
        for char in row:
            if char.isdigit():
                file += int(char)
            elif char in piece_map:
                color, piece_type = piece_map[char]
                square = (7 - rank) * 8 + file
                idx = color * 6 * 64 + piece_type * 64 + square
                nnue_vector[idx] = 1.0
                file += 1
    return nnue_vector


def load_and_encode_csv(
    filepath: str
) -> tuple[list[torch.Tensor], list[torch.Tensor], float, float]:
    boards_all = []
    points_all = []
    min_point = float('inf')
    max_point = float('-inf')
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            fen = row[0]
            point = float(row[1])
            boards_all.append(encode_board(fen))
            points_all.append(torch.tensor([point], dtype=torch.float32))
            if point < min_point:
                min_point = point
            if point > max_point:
                max_point = point
    return boards_all, points_all, min_point, max_point
