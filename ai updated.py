"""ai.py - Tic-tac-toe agents, from beginner-friendly to provably perfect.

Agents (all expose ``choose_move(board) -> cell``)
--------------------------------------------------
RandomAgent    uniformly random legal move.
RuleAgent      classic rules of thumb (win, block, centre, corner, edge) with an
               optional blunder rate. No fork awareness -> beatable by a human.
MCTSAgent      Monte-Carlo tree search (UCT) with random rollouts. Strength is
               tunable via the iteration budget.
MinimaxAgent   perfect play from a ``Solver``. It never loses, and wins whenever
               the opponent gives it a chance. Tie-breaks between equally
               optimal moves are either random (varied play) or *swindling*
               (pick the move that sets the most traps for a fallible opponent).

Solver
------
Negamax with every classic optimisation, each independently switchable so that
``experiments.py`` can measure what each one buys:

  * alpha-beta pruning
  * move ordering (immediate wins first, then centre > corners > edges)
  * transposition table with exact / lower-bound / upper-bound entries
  * symmetry reduction (8 board symmetries share one table entry)

Scores are from the side-to-move's point of view and *depth aware*: a win at
total move ``T`` scores ``WIN_SCORE - T`` (faster is better) and a loss scores
``-(WIN_SCORE - T)`` (slower is better). That makes the AI finish games
decisively and resist for as long as possible when it is lost.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Optional

from game import FULL_MASK, N_CELLS, WON, Board, canonical_key

CENTER = 4
CORNERS = (0, 2, 6, 8)
EDGES = (1, 3, 5, 7)
MOVE_ORDER = (4, 0, 2, 6, 8, 1, 3, 5, 7)
OPPOSITE_CORNER = {0: 8, 8: 0, 2: 6, 6: 2}

WIN_SCORE = 100
INF = 10_000
EXACT, LOWER, UPPER = 0, 1, 2

_ORDERED_BITS = tuple(1 << c for c in MOVE_ORDER)
_NATURAL_BITS = tuple(1 << c for c in range(N_CELLS))


def _cell_of(bit: int) -> int:
    return bit.bit_length() - 1


def describe_score(score: int, moves_played: int) -> str:
    """Human-readable result of a move: 'win in 2', 'draw', 'lose in 3'.

    ``score`` is the mover's score *after* the move, ``moves_played`` the number
    of pieces on the board *before* it.
    """
    if score == 0:
        return "draw"
    end = WIN_SCORE - abs(score)          # pieces on the board when the game ends
    if score > 0:
        k = (end - moves_played + 1) // 2  # own moves needed, this one included
        return "win now" if k == 1 else f"win in {k}"
    k = (end - moves_played) // 2
    return f"lose in {k}"


# =========================================================================== #
# Agents
# =========================================================================== #
class Agent(ABC):
    """Base class: holds a private RNG so every agent is reproducible by seed."""

    name = "agent"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    @abstractmethod
    def choose_move(self, board: Board) -> int:
        """Return a legal cell index (0-8) for the side to move."""

    @staticmethod
    def _legal(board: Board) -> list[int]:
        moves = board.legal_moves()
        if not moves:
            raise ValueError("no legal moves: the game is over")
        return moves

    def __repr__(self) -> str:
        return self.name


class RandomAgent(Agent):
    name = "random"

    def choose_move(self, board: Board) -> int:
        return self.rng.choice(self._legal(board))


class RuleAgent(Agent):
    """Win > block > centre > opposite corner > corner > edge, with random blunders.

    It deliberately has no concept of forks, so a human who builds a double
    threat will beat it.
    """

    def __init__(self, use_positional: bool = True, blunder_rate: float = 0.0,
                 name: str = "rules", seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.use_positional = use_positional
        self.blunder_rate = blunder_rate
        self.name = name

    def choose_move(self, board: Board) -> int:
        moves = self._legal(board)
        if self.rng.random() < self.blunder_rate:
            return self.rng.choice(moves)
        mine, theirs = board.masks()
        for m in moves:                               # 1. win
            if WON[mine | 1 << m]:
                return m
        for m in moves:                               # 2. block
            if WON[theirs | 1 << m]:
                return m
        if self.use_positional:
            free = set(moves)
            if CENTER in free:                        # 3. centre
                return CENTER
            for c in CORNERS:                         # 4. opposite corner
                if theirs >> c & 1 and OPPOSITE_CORNER[c] in free:
                    return OPPOSITE_CORNER[c]
            corners = [c for c in CORNERS if c in free]
            if corners:                               # 5. any corner
                return self.rng.choice(corners)
            return self.rng.choice([e for e in EDGES if e in free])   # 6. edge
        return self.rng.choice(moves)


class _Node:
    """MCTS tree node. ``reward`` is from the view of the player who moved INTO it."""

    __slots__ = ("mine", "theirs", "parent", "move", "children", "untried", "visits", "reward")

    def __init__(self, mine: int, theirs: int, parent: Optional["_Node"], move: Optional[int]) -> None:
        self.mine, self.theirs, self.parent, self.move = mine, theirs, parent, move
        self.children: list[_Node] = []
        self.visits = 0
        self.reward = 0.0
        occ = mine | theirs
        terminal = WON[theirs] or occ == FULL_MASK
        self.untried = [] if terminal else [i for i in range(N_CELLS) if not occ >> i & 1]


class MCTSAgent(Agent):
    """Monte-Carlo tree search with UCB1 selection and uniformly random rollouts."""

    def __init__(self, iterations: int = 500, exploration: float = math.sqrt(2),
                 seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.iterations = iterations
        self.exploration = exploration
        self.name = f"mcts-{iterations}"

    def choose_move(self, board: Board) -> int:
        moves = self._legal(board)
        if len(moves) == 1:
            return moves[0]
        mine, theirs = board.masks()
        root = _Node(mine, theirs, None, None)
        for _ in range(self.iterations):
            node = root
            while not node.untried and node.children:             # selection
                node = self._select(node)
            if node.untried:                                      # expansion
                m = node.untried.pop(self.rng.randrange(len(node.untried)))
                child = _Node(node.theirs, node.mine | 1 << m, node, m)
                node.children.append(child)
                node = child
            reward = self._rollout(node)                          # simulation
            while node is not None:                               # backpropagation
                node.visits += 1
                node.reward += reward
                reward = 1.0 - reward
                node = node.parent
        return max(root.children, key=lambda c: c.visits).move

    def _select(self, node: _Node) -> _Node:
        log_n = math.log(node.visits)
        c = self.exploration
        return max(node.children, key=lambda ch: ch.reward / ch.visits + c * math.sqrt(log_n / ch.visits))

    def _rollout(self, node: _Node) -> float:
        """Play randomly to the end; return the result for the player who moved into ``node``."""
        if WON[node.theirs]:
            return 1.0
        occ = node.mine | node.theirs
        if occ == FULL_MASK:
            return 0.5
        empties = [i for i in range(N_CELLS) if not occ >> i & 1]
        self.rng.shuffle(empties)
        a, b = node.mine, node.theirs
        a_is_node_mover = True
        for m in empties:
            a |= 1 << m
            if WON[a]:
                return 0.0 if a_is_node_mover else 1.0
            a, b = b, a
            a_is_node_mover = not a_is_node_mover
        return 0.5


# =========================================================================== #
# Perfect-play solver
# =========================================================================== #
class Solver:
    """Negamax solver with switchable optimisations and node statistics."""

    def __init__(self, *, alpha_beta: bool = True, ordering: bool = True,
                 transposition: bool = True, symmetry: bool = True) -> None:
        self.alpha_beta = alpha_beta
        self.ordering = ordering
        self.transposition = transposition
        self.symmetry = symmetry
        self._tt: dict[int, tuple[int, int]] = {}
        self.nodes = 0
        self.tt_hits = 0

    # ---- public API ------------------------------------------------------- #
    @property
    def table_size(self) -> int:
        return len(self._tt)

    def reset_stats(self) -> None:
        self.nodes = 0
        self.tt_hits = 0

    def clear(self) -> None:
        self._tt.clear()
        self.reset_stats()

    def score(self, board: Board) -> int:
        """Exact minimax score of ``board`` for the side to move."""
        mine, theirs = board.masks()
        return self._search(mine, theirs, -INF, INF)

    def move_scores(self, mine: int, theirs: int) -> dict[int, int]:
        """Exact score (mover's view) of every legal move, each searched with a full window."""
        occ = mine | theirs
        return {
            c: -self._search(theirs, mine | 1 << c, -INF, INF)
            for c in range(N_CELLS) if not occ >> c & 1
        }

    def analyze(self, board: Board) -> dict[int, int]:
        if board.is_over:
            raise ValueError("cannot analyze a finished game")
        return self.move_scores(*board.masks())

    # ---- search ----------------------------------------------------------- #
    def _search(self, mine: int, theirs: int, alpha: int, beta: int) -> int:
        self.nodes += 1
        occ = mine | theirs
        if WON[theirs]:                      # the previous mover completed a line
            return -(WIN_SCORE - occ.bit_count())
        if occ == FULL_MASK:
            return 0

        key = 0
        if self.transposition:
            key = canonical_key(mine, theirs) if self.symmetry else mine << N_CELLS | theirs
            entry = self._tt.get(key)
            if entry is not None:
                flag, value = entry
                if flag == EXACT:
                    self.tt_hits += 1
                    return value
                if flag == LOWER:
                    alpha = max(alpha, value)
                else:
                    beta = min(beta, value)
                if alpha >= beta:
                    self.tt_hits += 1
                    return value
        alpha_orig = alpha

        bits = [b for b in (_ORDERED_BITS if self.ordering else _NATURAL_BITS) if not occ & b]
        n = occ.bit_count()
        best = -INF
        won_now = False
        if self.ordering:                    # an immediate win is always the best move
            for b in bits:
                if WON[mine | b]:
                    best, won_now = WIN_SCORE - (n + 1), True
                    break
        if not won_now:
            for b in bits:
                value = -self._search(theirs, mine | b, -beta, -alpha)
                if value > best:
                    best = value
                if self.alpha_beta:
                    if value > alpha:
                        alpha = value
                    if alpha >= beta:
                        break

        if self.transposition:
            if won_now or not self.alpha_beta:
                flag = EXACT
            elif best <= alpha_orig:
                flag = UPPER
            elif best >= beta:
                flag = LOWER
            else:
                flag = EXACT
            self._tt[key] = (flag, best)
        return best


_SHARED: Optional[Solver] = None


def shared_solver() -> Solver:
    """A process-wide solver so every agent reuses one warm transposition table."""
    global _SHARED
    if _SHARED is None:
        _SHARED = Solver()
    return _SHARED


class MinimaxAgent(Agent):
    """Perfect play. ``tie_break`` decides among equally optimal moves:

    ``random``  uniformly random (varied games)
    ``first``   deterministic, centre > corners > edges
    ``swindle`` maximise expected result against an opponent who sometimes errs
                (see ``_swindle``); never gives up any guaranteed outcome.
    """

    def __init__(self, tie_break: str = "random", solver: Optional[Solver] = None,
                 swindle_epsilon: float = 0.2, name: Optional[str] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__(seed)
        if tie_break not in ("random", "first", "swindle"):
            raise ValueError(f"unknown tie_break {tie_break!r}")
        self.tie_break = tie_break
        self.solver = solver or shared_solver()
        self.epsilon = swindle_epsilon
        self.name = name or f"minimax-{tie_break}"
        self._cand_cache: dict[Board, tuple[int, ...]] = {}
        self._swindle_cache: dict[tuple[int, int, bool], float] = {}

    def candidate_moves(self, board: Board) -> list[int]:
        """Every move this agent could possibly play here (used for exhaustive proofs)."""
        cached = self._cand_cache.get(board)
        if cached is None:
            scores = self.solver.analyze(board)
            best = max(scores.values())
            cands = [m for m in MOVE_ORDER if scores.get(m) == best]
            if self.tie_break == "first":
                cands = cands[:1]
            elif self.tie_break == "swindle" and len(cands) > 1:
                mine, theirs = board.masks()
                values = {m: self._swindle(theirs, mine | 1 << m, trapper_to_move=False) for m in cands}
                top = max(values.values())
                cands = [m for m in cands if values[m] > top - 1e-12]
            cached = self._cand_cache[board] = tuple(cands)
        return list(cached)

    def choose_move(self, board: Board) -> int:
        self._legal(board)
        cands = self.candidate_moves(board)
        return cands[0] if len(cands) == 1 else self.rng.choice(cands)

    def _swindle(self, mine: int, theirs: int, trapper_to_move: bool) -> float:
        """Expected result (1 win, .5 draw, 0 loss) for the *trapper* from this position.

        Model: the trapper plays outcome-optimal moves, choosing the best by this
        very recursion. The opponent plays an outcome-optimal move with
        probability 1-eps and a uniformly random move with probability eps.
        """
        if WON[theirs]:
            return 0.0 if trapper_to_move else 1.0
        if (mine | theirs) == FULL_MASK:
            return 0.5
        key = (mine, theirs, trapper_to_move)
        hit = self._swindle_cache.get(key)
        if hit is not None:
            return hit
        scores = self.solver.move_scores(mine, theirs)
        outcome = {m: (s > 0) - (s < 0) for m, s in scores.items()}
        best = max(outcome.values())
        child = {m: self._swindle(theirs, mine | 1 << m, not trapper_to_move) for m in outcome}
        optimal = [child[m] for m, o in outcome.items() if o == best]
        if trapper_to_move:
            value = max(optimal)
        else:
            value = ((1 - self.epsilon) * sum(optimal) / len(optimal)
                     + self.epsilon * sum(child.values()) / len(child))
        self._swindle_cache[key] = value
        return value


# =========================================================================== #
# Difficulty levels
# =========================================================================== #
LEVEL_DESCRIPTIONS = {
    "easy": "Plays randomly, and only sometimes remembers to win or block.",
    "medium": "Wins, blocks and takes good squares - but can't see forks. Beatable.",
    "hard": "Perfect play. It never loses; the best you can do is draw.",
    "expert": "Perfect play that also sets traps, punishing any slip you make.",
}


def make_agent(level: str = "hard", seed: Optional[int] = None) -> Agent:
    """Build an agent from a level name: random, easy, medium, hard, expert, mcts[:N]."""
    level = level.lower().strip()
    if level == "random":
        return RandomAgent(seed)
    if level == "easy":
        return RuleAgent(use_positional=False, blunder_rate=0.45, name="easy", seed=seed)
    if level == "medium":
        return RuleAgent(use_positional=True, blunder_rate=0.05, name="medium", seed=seed)
    if level == "hard":
        return MinimaxAgent("random", name="hard", seed=seed)
    if level == "expert":
        return MinimaxAgent("swindle", name="expert", seed=seed)
    if level.startswith("mcts"):
        _, _, n = level.partition(":")
        return MCTSAgent(int(n) if n else 500, seed=seed)
    raise ValueError(f"unknown level {level!r}; choose from random, easy, medium, hard, expert, mcts[:N]")
