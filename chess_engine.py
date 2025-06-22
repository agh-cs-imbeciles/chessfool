#!/home/kpiotr6/Documents/stuida/IWISUM/proj/chessfool/.venv/bin/python
import sys
import os
from math import inf
from typing import Optional, Tuple

import chess
import torch

from utils import encode_board
from nnue_network import SimpleNNUE

nnue = SimpleNNUE()
board = chess.Board()
DEPTH = 4

WHITE_PAWN_TABLE = [
     0,  0,  0,   0,  0,  0,  0,  0,
     5, 10, 10,  15, 15, 10, 10,  5,
     5,  8, 12,  20, 20, 12,  8,  5,
     5, 10, 15,  25, 25, 15, 10,  5,
     5, 10, 15,  30, 30, 15, 10,  5,
     8, 12, 20,  35, 35, 20, 12,  8,
    10, 15, 20,  30, 30, 20, 15, 10,
     0,  0,  0,   0,  0,  0,  0,  0,
]

BLACK_PAWN_TABLE = WHITE_PAWN_TABLE[::-1]

WHITE_KNIGHT_TABLE = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]

BLACK_KNIGHT_TABLE = WHITE_KNIGHT_TABLE[::-1]

WHITE_BISHOP_TABLE = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]

BLACK_BISHOP_TABLE = WHITE_BISHOP_TABLE[::-1]

WHITE_ROOK_TABLE = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

BLACK_ROOK_TABLE = WHITE_ROOK_TABLE[::-1]

WHITE_QUEEN_TABLE = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]

BLACK_QUEEN_TABLE = WHITE_QUEEN_TABLE[::-1]

WHITE_KING_MID_TABLE = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]

BLACK_KING_MID_TABLE = WHITE_KING_MID_TABLE[::-1]

WHITE_KING_END_TABLE = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
]

BLACK_KING_END_TABLE = WHITE_KING_END_TABLE[::-1]

PIECE_SQUARE_TABLES = {
    (chess.PAWN, chess.WHITE): WHITE_PAWN_TABLE,
    (chess.PAWN, chess.BLACK): BLACK_PAWN_TABLE,

    (chess.KNIGHT, chess.WHITE): WHITE_KNIGHT_TABLE,
    (chess.KNIGHT, chess.BLACK): BLACK_KNIGHT_TABLE,

    (chess.BISHOP, chess.WHITE): WHITE_BISHOP_TABLE,
    (chess.BISHOP, chess.BLACK): BLACK_BISHOP_TABLE,

    (chess.ROOK, chess.WHITE): WHITE_ROOK_TABLE,
    (chess.ROOK, chess.BLACK): BLACK_ROOK_TABLE,

    (chess.QUEEN, chess.WHITE): WHITE_QUEEN_TABLE,
    (chess.QUEEN, chess.BLACK): BLACK_QUEEN_TABLE,

    # or WHITE_KING_END_TABLE depending on phase
    (chess.KING, chess.WHITE): WHITE_KING_MID_TABLE,

    # or BLACK_KING_END_TABLE depending on phase
    (chess.KING, chess.BLACK): BLACK_KING_MID_TABLE,
}


def get_input(prompt: Optional[str] = None) -> Optional[str]:
    prompt = prompt if prompt else ""
    try:
        return input(prompt)
    except EOFError:
        return None


def evaluate_board(board: chess.Board) -> int:
    """
    Basic evaluation function: returns a static evaluation of the position.
    Positive for White, negative for Black.
    If the game is over for one of the players returns infinity or -infinity for
    white or black respectively or 0 in the case of draw.
    """
    outcome = board.outcome()
    if outcome:
        if outcome.winner == chess.WHITE:
            return inf
        if outcome.winner == chess.BLACK:
            return -inf
        return 0
    # if board.turn == chess.BLACK:
    #     board = board.mirror()
    encoded_board = encode_board(board)
    score = nnue(encoded_board)
    score = score[0]
    # if board.turn == chess.BLACK:
    #     score = -score

    # values: dict[chess.PieceType, float] = {
    #     chess.PAWN: 100,
    #     chess.KNIGHT: 320,
    #     chess.BISHOP: 330,
    #     chess.ROOK: 500,
    #     chess.QUEEN: 900,
    #     chess.KING: 0,
    # }

    # score: int = 0

    # for piece_type, value in values.items():
    #     score += len(board.pieces(piece_type, chess.WHITE)) * value
    #     score -= len(board.pieces(piece_type, chess.BLACK)) * value

    # for square, piece in board.piece_map().items():
    #     score += PIECE_SQUARE_TABLES[(piece.piece_type, piece.color)][square]

    return score


def ordered_legal_moves(board: chess.Board) -> list[chess.Move]:
    """
    Prioritise captures over quiet moves (MVV-LVA).
    """
    def score(move: chess.Move) -> int:
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                return 10 * victim.piece_type - attacker.piece_type
        return 0

    return sorted(board.legal_moves, key=score, reverse=True)


def alpha_beta(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    maximising_player: bool
) -> Tuple[float, Optional[chess.Move]]:
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    best_move: Optional[chess.Move] = None

    if maximising_player:
        max_eval: float = -inf
        for move in ordered_legal_moves(board):
            board.push(move)
            eval_score, _ = alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move

    else:
        min_eval: float = inf
        for move in ordered_legal_moves(board):
            board.push(move)
            eval_score, _ = alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move


def play_game(depth: int = 3, user_plays_white: bool = True) -> None:
    print(chess)
    board = chess.Board()
    print("Welcome to CLI Chess vs Alpha-Beta Bot!")
    print("Enter your moves in UCI format (e.g., e2e4). Type 'quit' to exit.")
    print(board)

    while not board.is_game_over():
        if board.turn == chess.WHITE and user_plays_white or board.turn == chess.BLACK and not user_plays_white:
            move_uci = input("\nYour move: ").strip().lower()
            if move_uci == 'quit':
                print("Game exited.")
                return
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move. Try again.")
                    continue
            except ValueError:
                print("Invalid input format. Use UCI like e2e4.")
                continue
        else:
            print("\nAI is thinking...")
            eval_score, best_move = alpha_beta(board, depth, -inf, inf, board.turn == chess.WHITE)
            print(f"AI plays {best_move} (eval: {eval_score:.2f})")
            if best_move:
                board.push(best_move)
            else:
                break

        print("\n" + str(board))

    print("\nGame over!")
    result = board.result()
    print(f"Result: {result} ({board.outcome().termination.name})")


def uci() -> None:
    print("id name Chessfool 0.0.0")
    print("id author Piotr Kuchta and Jakub Szaredko")
    print()
    print("uciok")

    while True:
        raw_command = get_input()
        if raw_command is None:
            break
        command = raw_command.strip()

        if command == "isready":
            nnue.load_state_dict(torch.load("nnue_model_weights.pth", map_location="cpu"))
            nnue.eval()
            print("readyok")
        if command.startswith("position"):
            set_position(command)
        if command.startswith("go"):
            _, best_move = alpha_beta(
                board,
                DEPTH,
                alpha=-inf,
                beta=inf,
                maximising_player=board.turn
            )
            print(f"bestmove {best_move.uci()}")
        if command == "quit":
            break


def set_position(command: str) -> None:
    global board
    tokens = command.split()

    if "startpos" in tokens:
        board = chess.Board()
        moves_index = tokens.index("moves") if "moves" in tokens else None
    elif "fen" in tokens:
        fen_index = tokens.index("fen") + 1
        fen = " ".join(tokens[fen_index:fen_index + 6])
        board = chess.Board(fen)
        moves_index = tokens.index("moves") if "moves" in tokens else None
    else:
        return

    if moves_index:
        for move in tokens[moves_index + 1:]:
            board.push_uci(move)


if __name__ == "__main__":
    line_width = 52
    header_char = "="
    # print(line_width * "=")
    # print(f"={'Chessfool v0.0.0'.center(line_width - 2)}=")
    # print(line_width * "=")

    # print("To continue type 'uci' or 'cli'.")
    # print("Type 'quit' to exit the program.")
    raw_command = get_input()
    command = raw_command.strip().lower() if raw_command is not None else "quit"

    match command:
        case "uci":
            uci()
        case "cli":
            nnue.load_state_dict(torch.load("nnue_model_weights.pth", map_location="cpu"))
            nnue.eval()
            play_game(DEPTH, user_plays_white=True)
        case "quit":
            exit(os.EX_OK)
        case _:
            print(f"Unknown command '{raw_command.strip()}'.", file=sys.stderr)
