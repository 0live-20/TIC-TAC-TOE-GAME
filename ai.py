import random
from game import check_winner, get_legal_moves, make_move

# Counts how many board positions were examined.
# Partner B can use this in experiments.py to compare minimax vs alpha-beta.
nodes_expanded = 0


def other(player):
    return 'O' if player == 'X' else 'X'


def minimax(board, current, ai, depth):
    """Plain minimax. Scores are from the AI's point of view."""
    global nodes_expanded
    nodes_expanded += 1

    result = check_winner(board)
    if result == ai:
        return 10 + depth          # win sooner = better
    if result == other(ai):
        return -10 - depth         # lose later = better
    if result == 'draw' or depth == 0:
        return 0

    scores = []
    for move in get_legal_moves(board):
        new_board = make_move(board, move, current)
        scores.append(minimax(new_board, other(current), ai, depth - 1))

    return max(scores) if current == ai else min(scores)


def alphabeta(board, current, ai, depth, alpha, beta):
    """Minimax with alpha-beta pruning. Gives the same result, but faster."""
    global nodes_expanded
    nodes_expanded += 1

    result = check_winner(board)
    if result == ai:
        return 10 + depth
    if result == other(ai):
        return -10 - depth
    if result == 'draw' or depth == 0:
        return 0

    if current == ai:                       # maximizing
        best = float('-inf')
        for move in get_legal_moves(board):
            new_board = make_move(board, move, current)
            best = max(best, alphabeta(new_board, other(current), ai,
                                       depth - 1, alpha, beta))
            alpha = max(alpha, best)
            if beta <= alpha:
                break                       # prune
        return best
    else:                                   # minimizing
        best = float('inf')
        for move in get_legal_moves(board):
            new_board = make_move(board, move, current)
            best = min(best, alphabeta(new_board, other(current), ai,
                                       depth - 1, alpha, beta))
            beta = min(beta, best)
            if beta <= alpha:
                break                       # prune
        return best


def get_best_move(board, player, depth, use_alpha_beta=True):
    """Returns the best cell index (0-8) for `player` within `depth` moves.

    main.py calls this as get_best_move(board, ai_symbol, depth).
    """
    best_score = float('-inf')
    best_moves = []

    for move in get_legal_moves(board):
        new_board = make_move(board, move, player)
        if use_alpha_beta:
            score = alphabeta(new_board, other(player), player, depth - 1,
                              float('-inf'), float('inf'))
        else:
            score = minimax(new_board, other(player), player, depth - 1)

        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    # Pick randomly among equally good moves so the AI isn't predictable
    return random.choice(best_moves)