import numpy as np


class OthelloEnv:
    """
    Environment for Othello/Reversi.

    Internal representation board_abs:
        1  = player 1
       -1  = player -1
        0  = empty cell

    Observation["board"]:
        always relative to the current player:
        1  = my pieces
       -1  = opponent pieces
        0  = empty cell
    """

    def __init__(self, board_size: int = 5):
        if board_size < 4:
            raise ValueError("board_size has to be at least 4.")
        if board_size >8:
            raise ValueError("board_size has to be at most 8, to be able compute it in our laptops")

        self.board_size = board_size
        self.pass_action = board_size * board_size

        self.board_abs = None
        self.current_player = 1
        self.turn_count = 0
        self.done = False

        self.reset()

    def reset(self):
        """
        Initialize a new game.
        Returns the observation for the current player.
        """
        n = self.board_size
        self.board_abs = np.zeros((n, n), dtype=np.int8)

        # Starting position for even board sizes is the classic center setup.
        # For odd board sizes like 5x5 we use four stones around the center.
        mid1 = n // 2 - 1
        mid2 = n // 2

        self.board_abs[mid1, mid1] = 1
        self.board_abs[mid2, mid2] = 1
        self.board_abs[mid1, mid2] = -1
        self.board_abs[mid2, mid1] = -1

        self.current_player = 1
        self.turn_count = 0
        self.done = False

        return self.get_observation()

    def get_observation(self):
        """
        Return the observation dictionary in a task-compatible format.
        """
        board_relative = self.board_abs * self.current_player
        legal_actions = self.get_legal_actions(self.current_player)

        return {
            "board": board_relative.copy(),
            "board_abs": self.board_abs.copy(),
            "current_player": self.current_player,
            "legal_actions": legal_actions,
            "pass_action": self.pass_action,
            "board_size": self.board_size,
            "turn_count": self.turn_count,
        }

    def get_legal_actions(self, player: int):
        """
        Return a list of legal actions for the given player.

        Actions:
            action = row * board_size + col

        If the player has no regular move, the only legal action is pass_action.
        """
        legal_actions = []

        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_legal_move(row, col, player):
                    action = row * self.board_size + col
                    legal_actions.append(action)

        if len(legal_actions) == 0:
            legal_actions.append(self.pass_action)

        return legal_actions

    def is_legal_move(self, row: int, col: int, player: int):
        """
        A move is legal if:
        - the cell is empty,
        - in some direction it encloses at least one opponent piece.
        """
        if not self.is_on_board(row, col):
            return False

        if self.board_abs[row, col] != 0:
            return False

        directions = self.get_directions()

        for dr, dc in directions:
            if self.would_flip_in_direction(row, col, dr, dc, player):
                return True

        return False

    def would_flip_in_direction(self, row: int, col: int, dr: int, dc: int, player: int):
        """
        Check whether the move would flip pieces in a specific direction.
        """
        opponent = -player
        r = row + dr
        c = col + dc

        found_opponent = False

        while self.is_on_board(r, c):
            value = self.board_abs[r, c]

            if value == opponent:
                found_opponent = True

            elif value == player:
                return found_opponent

            else:
                return False

            r += dr
            c += dc

        return False


    def step(self, action: int):
        """
        Execute the current player's move.

        Returns:
            observation, reward, done, info

        Reward is currently simple:
            during the game: 0
            end of the game:
                +1 if the player who moved won
                -1 if they lost
                 0 if a draw
    
        """
        if self.done:
            raise RuntimeError("The game is already over. Call reset().")

        legal_actions = self.get_legal_actions(self.current_player)
        player_who_moved = self.current_player

        if action not in legal_actions:
            self.done = True

            info = {
                "illegal_move": True,
                "winner": -player_who_moved,
                "reward_player": player_who_moved,
                "disc_diff": self.get_disc_difference(player_who_moved),
            }

            return self.get_observation(), -1.0, True, info

        if action != self.pass_action:
            row = action // self.board_size
            col = action % self.board_size
            self.apply_move(row, col, player_who_moved)

        self.turn_count += 1

        # Další observation bude z pohledu soupeře.
        self.current_player *= -1

        if self.is_game_over():
            self.done = True
            winner = self.get_winner()

            if winner == player_who_moved:
                reward = 1.0
            elif winner == 0:
                reward = 0.0
            else:
                reward = -1.0

            info = {
                "illegal_move": False,
                "winner": winner,
                "reward_player": player_who_moved,
                "disc_diff": self.get_disc_difference(player_who_moved),
            }

            return self.get_observation(), reward, True, info

        info = {
            "illegal_move": False,
            "winner": None,
            "reward_player": player_who_moved,
            "disc_diff": self.get_disc_difference(player_who_moved),
        }

        return self.get_observation(), 0.0, False, info

    def apply_move(self, row: int, col: int, player: int):
        """
        Place a piece and flip all opponent pieces that are enclosed by this move.
        """
        self.board_abs[row, col] = player

        directions = self.get_directions()

        for dr, dc in directions:
            if self.would_flip_in_direction(row, col, dr, dc, player):
                self.flip_in_direction(row, col, dr, dc, player)

    def flip_in_direction(self, row: int, col: int, dr: int, dc: int, player: int):
        """
        Flip pieces in the given direction.
        Assumes the direction is valid.
        """
        opponent = -player
        r = row + dr
        c = col + dc

        while self.is_on_board(r, c) and self.board_abs[r, c] == opponent:
            self.board_abs[r, c] = player
            r += dr
            c += dc

    def is_game_over(self):
        """
        The game is over when neither player has a normal move.
        """
        current_has_move = self.has_any_normal_move(self.current_player)
        opponent_has_move = self.has_any_normal_move(-self.current_player)

        return not current_has_move and not opponent_has_move

    def has_any_normal_move(self, player: int):
        """
        True if the player has at least one normal move,
        not just pass.
        """
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_legal_move(row, col, player):
                    return True

        return False

    def get_winner(self):
        """
        Returns:
             1 if player 1 won
            -1 if player -1 won
             0 if a draw
        """
        count_player_1 = np.sum(self.board_abs == 1)
        count_player_minus_1 = np.sum(self.board_abs == -1)

        if count_player_1 > count_player_minus_1:
            return 1

        if count_player_minus_1 > count_player_1:
            return -1

        return 0

    def get_disc_difference(self, player: int):
        """
        Difference in disc count from the given player's perspective.
        """
        player_count = np.sum(self.board_abs == player)
        opponent_count = np.sum(self.board_abs == -player)

        return int(player_count - opponent_count)

    def count_corners(self, player: int) -> int:
        """Number of corners owned by `player`."""
        n = self.board_size
        last = n - 1
        corners = [
            self.board_abs[0, 0],
            self.board_abs[0, last],
            self.board_abs[last, 0],
            self.board_abs[last, last],
        ]
        return int(sum(1 for v in corners if v == player))

    def count_legal_actions(self, player: int) -> int:
        """Number of legal (non-pass) moves for `player`."""
        legal = self.get_legal_actions(player)
        return int(sum(1 for a in legal if a != self.pass_action))

    def is_on_board(self, row: int, col: int):
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    @staticmethod
    def get_directions():
        return [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    def render(self):
        """
        Simple text rendering of the board.
        """
        symbols = {
            1: "X",
            -1: "O",
            0: "."
        }

        print()
        print("Current player:", "X" if self.current_player == 1 else "O")
        print("Turn:", self.turn_count)
        print()

        header = "   " + " ".join(str(i) for i in range(self.board_size))
        print(header)

        for row in range(self.board_size):
            line = str(row) + "  "
            for col in range(self.board_size):
                line += symbols[int(self.board_abs[row, col])] + " "
            print(line)

        print()