import random
from game import (create_board, print_board, check_winner,
                  get_legal_moves, make_move)

# Partner B's minimax lives in ai.py. Until it is ready, we fall back to a
# random AI so you can test the game loop on your own.
try:
    from ai import get_best_move
    AI_READY = True
except ImportError:
    AI_READY = False

    def get_best_move(board, player, depth):
        return random.choice(get_legal_moves(board))

# Difficulty = depth limit passed to minimax
DIFFICULTY = {
    '1': ('Easy', 1),
    '2': ('Medium', 3),
    '3': ('Hard', 9),   # 9 = full depth, so the AI plays perfectly
}


def choose_difficulty():
    print("\nChoose difficulty:")
    for key, (name, depth) in DIFFICULTY.items():
        print(f"  {key}. {name} (depth {depth})")
    while True:
        choice = input("Enter 1, 2 or 3: ").strip()
        if choice in DIFFICULTY:
            return DIFFICULTY[choice]
        print("Invalid choice. Try again.")


def choose_first():
    """Returns (human_symbol, ai_symbol). X always moves first."""
    while True:
        choice = input("Do you want to go first? (y/n): ").strip().lower()
        if choice == 'y':
            return 'X', 'O'
        if choice == 'n':
            return 'O', 'X'
        print("Please enter y or n.")


def get_human_move(board):
    legal = get_legal_moves(board)
    while True:
        raw = input("Your move (1-9): ").strip()
        if not raw.isdigit():
            print("Please enter a number from 1 to 9.")
            continue
        position = int(raw) - 1
        if position < 0 or position > 8:
            print("Number must be between 1 and 9.")
        elif position not in legal:
            print("That cell is already taken.")
        else:
            return position


def show_position_guide():
    print("\nCells are numbered like this:")
    guide = [str(i) for i in range(1, 10)]
    print_board(guide)


def play_game():
    diff_name, depth = choose_difficulty()
    human, ai = choose_first()
    board = create_board()
    current = 'X'

    print(f"\nYou are {human}, the AI is {ai}. Difficulty: {diff_name}")
    show_position_guide()

    while True:
        print()
        print_board(board)
        print()

        if current == human:
            move = get_human_move(board)
        else:
            print("AI is thinking...")
            move = get_best_move(board, ai, depth)
            print(f"AI plays cell {move + 1}")

        board = make_move(board, move, current)

        result = check_winner(board)
        if result is not None:
            print()
            print_board(board)
            print()
            if result == 'draw':
                print("It's a draw!")
            elif result == human:
                print("You win!")
            else:
                print("The AI wins!")
            return

        current = 'O' if current == 'X' else 'X'


def main():
    print("=== Tic-Tac-Toe vs AI ===")
    if not AI_READY:
        print("(ai.py not found yet - the AI will play randomly)")

    while True:
        play_game()
        again = input("\nPlay again? (y/n): ").strip().lower()
        if again != 'y':
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()