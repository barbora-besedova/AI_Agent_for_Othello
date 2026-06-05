import random


class RandomAgent:
    def __init__(self, board_size: int):
        self.board_size = board_size

    def select_action(self, observation: dict) -> int:
        legal_actions = observation["legal_actions"]
        return random.choice(legal_actions)