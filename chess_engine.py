import math
from typing import Optional, Tuple

import chess


def evaluate_board(board: chess.Board) -> float:
    """
    Basic evaluation function: returns a static evaluation of the position.
    Positive for White, negative for Black.
    """
    values: dict[chess.PieceType, float] = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.2,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0
    }

    score: float = 0.0
    for piece_type, value in values.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def order_moves(board: chess.Board) -> list[chess.Move]:
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
        max_eval: float = -math.inf
        for move in board.legal_moves:
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
        min_eval: float = math.inf
        for move in order_moves(board):
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
            eval_score, best_move = alpha_beta(board, depth, -math.inf, math.inf, board.turn == chess.WHITE)
            print(f"AI plays {best_move} (eval: {eval_score:.2f})")
            if best_move:
                board.push(best_move)
            else:
                break

        print("\n" + str(board))

    print("\nGame over!")
    result = board.result()
    print(f"Result: {result} ({board.outcome().termination.name})")


if __name__ == "__main__":
    play_game(depth=6, user_plays_white=True)  # Set to False to play Black
