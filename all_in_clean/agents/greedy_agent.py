import random

import numpy as np


class GreedyAgent:
    def __init__(self, board_size: int):
        self.board_size = board_size

    def select_action(self, observation: dict) -> int:
        legal_actions = observation["legal_actions"]
        pass_action = observation["pass_action"]

        if legal_actions == [pass_action]:
            return pass_action

        board = observation["board"]

        best_action = legal_actions[0]
        best_score = -np.inf
        best_actions = []

        for action in legal_actions:
            if action == pass_action:
                continue

            score = self.evaluate_action(board, action)

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions) if best_actions else best_action

    def evaluate_action(self, board, action: int) -> float:
        """
        Jednoduchá heuristika:
        - odhadne, kolik soupeřových kamenů by se otočilo
        - přidá bonus za rohy
        - menší bonus za hrany
        """
        row = action // self.board_size
        col = action % self.board_size

        flipped_count = self.count_flipped_discs(board, row, col)

        score = flipped_count

        if self.is_corner(row, col):
            score += 10

        elif self.is_edge(row, col):
            score += 2

        return score

    def count_flipped_discs(self, board, row: int, col: int) -> int:
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        total_flipped = 0

        for dr, dc in directions:
            total_flipped += self.count_flipped_in_direction(board, row, col, dr, dc)

        return total_flipped

    def count_flipped_in_direction(self, board, row: int, col: int, dr: int, dc: int) -> int:
        r = row + dr
        c = col + dc

        count = 0

        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            value = board[r, c]

            if value == -1:
                count += 1

            elif value == 1:
                return count

            else:
                return 0

            r += dr
            c += dc

        return 0

    def is_corner(self, row: int, col: int) -> bool:
        last = self.board_size - 1
        return (row, col) in [
            (0, 0),
            (0, last),
            (last, 0),
            (last, last),
        ]

    def is_edge(self, row: int, col: int) -> bool:
        last = self.board_size - 1
        return row == 0 or row == last or col == 0 or col == last