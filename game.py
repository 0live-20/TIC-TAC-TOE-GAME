WIN_LINES = [(0,1,2), (3,4,5), (6,7,8),   # rows
             (0,3,6), (1,4,7), (2,5,8),   # columns
             (0,4,8), (2,4,6)]            # diagonals

def create_board():
    return [' '] * 9

def print_board(board):
    for i in range(0, 9, 3):
        print(' ' + ' | '.join(board[i:i+3]))
        if i < 6:
            print('---+---+---')

def check_winner(board):
    """Returns 'X', 'O', 'draw', or None if the game is still going."""
    for a, b, c in WIN_LINES:
        if board[a] != ' ' and board[a] == board[b] == board[c]:
            return board[a]
    if ' ' not in board:
        return 'draw'
    return None

def get_legal_moves(board):
    return [i for i in range(9) if board[i] == ' ']

def make_move(board, position, player):
    """Returns a NEW board and leaves the original unchanged."""
    new_board = board.copy()
    new_board[position] = player
    return new_board
