import pygame
import sys

from environments.environment import OthelloEnv


class OthelloGUI:
    def __init__(self, board_size=5, cell_size=90):
        pygame.init()

        self.env = OthelloEnv(board_size=board_size)

        self.board_size = board_size
        self.cell_size = cell_size
        self.margin_top = 80
        self.window_width = board_size * cell_size
        self.window_height = board_size * cell_size + self.margin_top

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )

        pygame.display.set_caption("Othello / Reversi")

        self.font = pygame.font.SysFont(None, 32)
        self.small_font = pygame.font.SysFont(None, 24)

        self.obs = self.env.reset()
        self.done = False
        self.last_info = None
        self.last_reward = 0

    def run(self):
        clock = pygame.time.Clock()

        while True:
            self.handle_events()
            self.draw()
            pygame.display.flip()
            clock.tick(30)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_game()

                if event.key == pygame.K_p:
                    self.try_pass()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_click(event.pos)

    def handle_click(self, pos):
        if self.done:
            return

        x, y = pos

        if y < self.margin_top:
            return

        col = x // self.cell_size
        row = (y - self.margin_top) // self.cell_size

        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return

        action = row * self.board_size + col

        if action not in self.obs["legal_actions"]:
            return

        self.obs, self.last_reward, self.done, self.last_info = self.env.step(action)

    def try_pass(self):
        if self.done:
            return

        pass_action = self.obs["pass_action"]

        if pass_action in self.obs["legal_actions"]:
            self.obs, self.last_reward, self.done, self.last_info = self.env.step(pass_action)

    def reset_game(self):
        self.obs = self.env.reset()
        self.done = False
        self.last_info = None
        self.last_reward = 0

    def draw(self):
        self.screen.fill((240, 240, 240))
        self.draw_status()
        self.draw_board()
        self.draw_pieces()
        self.draw_legal_moves()

    def draw_status(self):
        if self.done:
            winner = self.last_info["winner"] if self.last_info else self.env.get_winner()

            if winner == 1:
                text = "Game over: X won"
            elif winner == -1:
                text = "Game over: O won"
            else:
                text = "Game over: draw"

            restart_text = "Press R to restart"

        else:
            player = "X" if self.env.current_player == 1 else "O"
            text = f"Current player: {player}"
            restart_text = "Click a legal move | P = pass | R = restart"

        label = self.font.render(text, True, (20, 20, 20))
        label2 = self.small_font.render(restart_text, True, (70, 70, 70))

        self.screen.blit(label, (15, 15))
        self.screen.blit(label2, (15, 45))

    def draw_board(self):
        for row in range(self.board_size):
            for col in range(self.board_size):
                x = col * self.cell_size
                y = self.margin_top + row * self.cell_size

                pygame.draw.rect(
                    self.screen,
                    (34, 139, 34),
                    (x, y, self.cell_size, self.cell_size)
                )

                pygame.draw.rect(
                    self.screen,
                    (0, 80, 0),
                    (x, y, self.cell_size, self.cell_size),
                    2
                )

    def draw_pieces(self):
        board = self.env.board_abs

        for row in range(self.board_size):
            for col in range(self.board_size):
                value = board[row, col]

                if value == 0:
                    continue

                center_x = col * self.cell_size + self.cell_size // 2
                center_y = self.margin_top + row * self.cell_size + self.cell_size // 2
                radius = self.cell_size // 3

                if value == 1:
                    color = (20, 20, 20)
                else:
                    color = (245, 245, 245)

                pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
                pygame.draw.circle(self.screen, (0, 0, 0), (center_x, center_y), radius, 2)

    def draw_legal_moves(self):
        if self.done:
            return

        legal_actions = self.obs["legal_actions"]
        pass_action = self.obs["pass_action"]

        for action in legal_actions:
            if action == pass_action:
                continue

            row = action // self.board_size
            col = action % self.board_size

            center_x = col * self.cell_size + self.cell_size // 2
            center_y = self.margin_top + row * self.cell_size + self.cell_size // 2

            pygame.draw.circle(
                self.screen,
                (255, 215, 0),
                (center_x, center_y),
                10
            )


if __name__ == "__main__":
    gui = OthelloGUI(board_size=5, cell_size=90)
    gui.run()