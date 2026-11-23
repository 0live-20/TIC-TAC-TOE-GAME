
# =====================================================
# PROJECT 3: AI PLAYER FOR TIC-TAC-TOE
# File: ai.py
# Algorithm: Minimax
# Human: X
# AI: O
# =====================================================

from game import check_winner, check_draw


# STEP 1: MINIMAX ALGORITHM

def minimax(board, depth, is_maximizing):
    """
    Minimax algorithm evaluates possible future moves
    and returns the best score for the AI.

    Parameters:
        board: Current Tic-Tac-Toe board
        depth: Number of moves explored
        is_maximizing: True when AI is playing

    Returns:
        Best possible score
    """

    winner = check_winner(board)

    # AI wins
    if winner == "O":
        return 10 - depth

    # Human wins
    elif winner == "X":
        return depth - 10

    # Game ends in a draw
    elif check_draw(board):
        return 0

    # AI maximizes the score
    if is_maximizing:

        best_score = float("-inf")

        for i in range(9):

            if board[i] == " ":

                # Try an AI move
                board[i] = "O"

                # Explore the opponent's response
                score = minimax(board, depth + 1, False)

                # Undo the move
                board[i] = " "

                best_score = max(best_score, score)

        return best_score

    # Human minimizes the AI's score
    else:

        best_score = float("inf")

        for i in range(9):

            if board[i] == " ":

                # Try a human move
                board[i] = "X"

                # Explore the AI's response
                score = minimax(board, depth + 1, True)

                # Undo the move
                board[i] = " "

                best_score = min(best_score, score)

        return best_score


# STEP 2: FIND THE BEST AI MOVE

def get_best_move(board):
    """
    Evaluates all available positions and returns
    the position with the highest Minimax score.

    Returns:
        Index of the best move, or None if no moves exist.
    """

    best_score = float("-inf")
    best_move = None

    for i in range(9):

        if board[i] == " ":

            # Temporarily place AI symbol
            board[i] = "O"

            # Evaluate the move
            score = minimax(board, 0, False)

            # Restore the original board
            board[i] = " "

            # Keep the best-scoring move
            if score > best_score:

                best_score = score
                best_move = i

    return best_move


# STEP 3: EXECUTE THE AI MOVE

def ai_move(board):
    """
    Makes the AI's selected move on the board.

    Returns:
        Position selected by the AI, or None if full.
    """

    best_move = get_best_move(board)

    if best_move is not None:

        board[best_move] = "O"

        return best_move

    return None
