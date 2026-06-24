import random

import numpy as np


class HeuristicAgent:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.pass_action = board_size * board_size

    def select_action(self, observation: dict) -> int:
        legal_actions = observation["legal_actions"]

        if legal_actions == [observation["pass_action"]]:
            return observation["pass_action"]

        board = observation["board"]

        best_action = legal_actions[0]
        best_score = -np.inf
        best_actions = []

        for action in legal_actions:
            if action == observation["pass_action"]:
                continue

            score = self.evaluate_action(board, action)

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions) if best_actions else best_action

    def evaluate_action(self, board, action: int) -> float:
        row = action // self.board_size
        col = action % self.board_size

        score = 0

        score += 1.0 * self.count_flipped_discs(board, row, col)
        score += self.position_score(row, col)

        return score

    def position_score(self, row: int, col: int) -> float:
        n = self.board_size
        last = n - 1

        corners = [(0, 0), (0, last), (last, 0), (last, last)]

        if (row, col) in corners:
            return 30

        # Pole vedle rohů bývají nebezpečná, pokud roh ještě není obsazený.
        dangerous = [
            (0, 1), (1, 0), (1, 1),
            (0, last - 1), (1, last), (1, last - 1),
            (last - 1, 0), (last, 1), (last - 1, 1),
            (last, last - 1), (last - 1, last), (last - 1, last - 1),
        ]

        if (row, col) in dangerous:
            return -8

        if row == 0 or row == last or col == 0 or col == last:
            return 5

        return 0

    def count_flipped_discs(self, board, row: int, col: int) -> int:
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        total = 0

        for dr, dc in directions:
            total += self.count_flipped_in_direction(board, row, col, dr, dc)

        return total

    def count_flipped_in_direction(self, board, row: int, col: int, dr: int, dc: int) -> int:
        r = row + dr
        c = col + dc
        count = 0

        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if board[r, c] == -1:
                count += 1
            elif board[r, c] == 1:
                return count
            else:
                return 0

            r += dr
            c += dc

        return 0