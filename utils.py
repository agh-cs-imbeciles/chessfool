
import pandas as pd
from nnue_network import encode_board
import torch

def load_and_encode_csv(filepath: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    df = pd.read_csv(filepath)
    boards = df['FEN'].apply(encode_board).tolist()
    points = df['Analysis'].tolist()
    points = [torch.tensor([p], dtype=torch.float32) for p in points]
    return boards, points

