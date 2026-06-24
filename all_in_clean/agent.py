"""
Unified DQN agent for Othello/Reversi.

Supports three orthogonal features that can be combined freely:

    use_per           — Prioritized Experience Replay (PER)
    heuristic_weight  — Guided action selection (HDQN); 0.0 = disabled
    double_dqn        — Double DQN target (always enabled by default)

Typical configurations
----------------------
Classic DQN:
    DQNAgent(board_size, use_per=False, heuristic_weight=0.0)

Guided DQN (HDQN):
    DQNAgent(board_size, use_per=False, heuristic_weight=0.2)

Prioritized DQN (PDQN):
    DQNAgent(board_size, use_per=True, heuristic_weight=0.0)

Guided + Prioritized:
    DQNAgent(board_size, use_per=True, heuristic_weight=0.2)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from q_network import QNetwork
from replay_buffer import ReplayBuffer
from prioritized_replay_buffer import PrioritizedReplayBuffer


class DQNAgent:
    """
    Double-DQN agent with optional Prioritized Experience Replay
    and optional heuristic-guided action selection.

    Args:
        board_size:          Side length of the Othello board.
        learning_rate:       Adam learning rate.
        gamma:               Discount factor.
        batch_size:          Mini-batch size for training.
        buffer_capacity:     Maximum replay-buffer size.
        target_update_freq:  Hard-update the target network every N train steps.
        device:              'cpu' | 'cuda' | None (auto-detect).
        learning_starts:     Minimum transitions before training begins.
        double_dqn:          Use Double-DQN target (recommended: True).

        heuristic_weight:    Weight of the heuristic bonus added to Q-values
                             during action selection. 0.0 disables guided mode.
        heuristic_weight_end:      Final heuristic weight after annealing.
        heuristic_decay_episodes:  Episodes over which heuristic weight anneals.

        use_per:             Use Prioritized Experience Replay.
        per_alpha:           PER priority exponent (0 = uniform, 1 = full).
        per_beta_start:      Initial IS-correction exponent.
        per_beta_frames:     Frames over which beta anneals to 1.0.

        tau:                 Polyak soft-update coefficient. 0.0 = hard update.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        board_size: int,
        # --- training hyper-parameters ---
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_update_freq: int = 500,
        device: Optional[str] = None,
        learning_starts: int = 1_000,
        double_dqn: bool = True,
        # --- guided / heuristic ---
        heuristic_weight: float = 0.0,
        heuristic_weight_end: float = 0.0,
        heuristic_decay_episodes: int = 50_000,
        # --- prioritized replay ---
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        # --- Polyak soft target update ---
        tau: float = 0.005,
    ):
        self.board_size = board_size
        self.action_size = board_size * board_size + 1

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.double_dqn = double_dqn
        self.tau = tau

        self.heuristic_weight = heuristic_weight
        self.heuristic_weight_start = heuristic_weight
        self.heuristic_weight_end = heuristic_weight_end
        self.heuristic_decay_episodes = heuristic_decay_episodes

        self.use_per = use_per
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames

        # ----- device -----
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # ----- networks -----
        self.q_net = QNetwork(board_size).to(self.device)
        self.target_net = QNetwork(board_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # ----- optimizer -----
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # Per-sample loss is required for PER; no harm using it always.
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        # ----- replay buffer -----
        if use_per:
            self.replay_buffer: ReplayBuffer | PrioritizedReplayBuffer = (
                PrioritizedReplayBuffer(capacity=buffer_capacity, alpha=per_alpha)
            )
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.train_steps = 0
        self.episodes_seen = 0

    # ------------------------------------------------------------------ #
    #  State encoding                                                      #
    # ------------------------------------------------------------------ #

    def encode_state(self, observation: Dict[str, Any]) -> np.ndarray:
        """Board (H, W) → (1, H, W) float32 array."""
        board = observation["board"].astype(np.float32)
        return np.expand_dims(board, axis=0)

    # ------------------------------------------------------------------ #
    #  Heuristic-weight annealing                                          #
    # ------------------------------------------------------------------ #

    def current_heuristic_weight(self) -> float:
        """Linearly anneal heuristic_weight from start to end over training."""
        if self.heuristic_decay_episodes <= 0:
            return self.heuristic_weight_end
        progress = min(1.0, self.episodes_seen / self.heuristic_decay_episodes)
        return (
            self.heuristic_weight_start
            + progress * (self.heuristic_weight_end - self.heuristic_weight_start)
        )

    def notify_episode_done(self) -> None:
        """Called by the training loop at the end of each episode."""
        self.episodes_seen += 1

    # ------------------------------------------------------------------ #
    #  Action selection                                                    #
    # ------------------------------------------------------------------ #

    def select_action(
        self,
        observation: Dict[str, Any],
        epsilon: float = 0.0,
    ) -> int:
        legal_actions: List[int] = observation["legal_actions"]

        if len(legal_actions) == 0:
            return int(observation["pass_action"])

        # ε-greedy exploration
        if random.random() < epsilon:
            return int(random.choice(legal_actions))

        # --- Q-values from online network ---
        state = self.encode_state(observation)
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze(0).cpu().numpy()

        # Mask illegal actions
        masked_q = np.full(self.action_size, -np.inf, dtype=np.float32)
        masked_q[legal_actions] = q_values[legal_actions]

        # Optional heuristic bonus (guided mode)
        hw = self.current_heuristic_weight()
        if hw > 0.0:
            best_action = self._guided_argmax(observation, masked_q, legal_actions, hw)
        else:
            max_q = np.max(masked_q[legal_actions])
            best_actions = [a for a in legal_actions if masked_q[a] == max_q]
            best_action = int(random.choice(best_actions))

        # Safety fallback
        if best_action not in legal_actions:
            best_action = legal_actions[0]

        return best_action

    def _guided_argmax(
        self,
        observation: Dict[str, Any],
        masked_q: np.ndarray,
        legal_actions: List[int],
        hw: float,
    ) -> int:
        """Return the legal action maximising Q + hw * heuristic.
        Ties are broken randomly to inject diversity."""
        best_actions = []
        best_score = -float("inf")

        for action in legal_actions:
            score = masked_q[action] + hw * self._heuristic_score(
                observation, action
            )
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return int(random.choice(best_actions))

    def _heuristic_score(
        self,
        observation: Dict[str, Any],
        action: int,
    ) -> float:
        """
        Board-position heuristic for 5×5 (and larger) Othello:
            +5  corner
            +1  edge
            −4  cell adjacent to an empty corner (dangerous square)
            −1  pass action
        """
        if action == observation["pass_action"]:
            return -1.0

        n = self.board_size
        last = n - 1
        row, col = divmod(action, n)
        board = observation["board"]

        score = 0.0

        corners = {(0, 0), (0, last), (last, 0), (last, last)}
        if (row, col) in corners:
            score += 5.0

        if row == 0 or row == last or col == 0 or col == last:
            score += 1.0

        # Cells that give the opponent a corner if empty
        dangerous: Dict[Tuple[int, int], Tuple[int, int]] = {
            (0, 1): (0, 0),   (1, 0): (0, 0),   (1, 1): (0, 0),
            (0, last - 1): (0, last), (1, last): (0, last), (1, last - 1): (0, last),
            (last - 1, 0): (last, 0), (last, 1): (last, 0), (last - 1, 1): (last, 0),
            (last - 1, last): (last, last), (last, last - 1): (last, last),
            (last - 1, last - 1): (last, last),
        }
        if (row, col) in dangerous:
            cr, cc = dangerous[(row, col)]
            if board[cr][cc] == 0:
                score -= 4.0

        return score

    # ------------------------------------------------------------------ #
    #  Replay buffer interaction                                           #
    # ------------------------------------------------------------------ #

    def store_transition(
        self,
        observation: Dict[str, Any],
        action: int,
        reward: float,
        next_observation: Optional[Dict[str, Any]],
        done: bool,
    ) -> None:
        state = self.encode_state(observation)

        if next_observation is None:
            next_state = np.zeros_like(state, dtype=np.float32)
            next_legal_actions: List[int] = []
        else:
            next_state = self.encode_state(next_observation)
            next_legal_actions = next_observation["legal_actions"]

        self.replay_buffer.push(
            state, action, reward, next_state, done, next_legal_actions
        )

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def _get_beta(self) -> float:
        """Linearly anneal β from per_beta_start to 1.0."""
        progress = min(1.0, self.train_steps / self.per_beta_frames)
        return self.per_beta_start + progress * (1.0 - self.per_beta_start)

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Sample a mini-batch and perform one gradient-descent step.

        Returns a dict with training diagnostics, or None if training
        has not yet started (buffer too small).
        """
        min_size = max(self.learning_starts, self.batch_size)
        if len(self.replay_buffer) < min_size:
            return None

        # ----- sample -----
        if self.use_per:
            beta = self._get_beta()
            batch, per_indices, is_weights = self.replay_buffer.sample(
                self.batch_size, beta=beta
            )
        else:
            beta = 1.0
            batch = self.replay_buffer.sample(self.batch_size)
            per_indices = None
            is_weights = np.ones(self.batch_size, dtype=np.float32)

        # ----- unpack -----
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        next_legal_batch = [t[5] for t in batch]

        # ----- tensors -----
        s = torch.tensor(states, device=self.device)
        a = torch.tensor(actions, device=self.device).unsqueeze(1)
        r = torch.tensor(rewards, device=self.device)
        ns = torch.tensor(next_states, device=self.device)
        d = torch.tensor(dones, device=self.device)
        w = torch.tensor(is_weights, device=self.device)

        # ----- current Q -----
        current_q = self.q_net(s).gather(1, a).squeeze(1)

        # ----- target Q (Double DQN) -----
        with torch.no_grad():
            online_next_q = self.q_net(ns)
            target_next_q = self.target_net(ns) if self.double_dqn else online_next_q

            next_q_list = []
            for i, legal in enumerate(next_legal_batch):
                if dones[i] or len(legal) == 0:
                    next_q_list.append(
                        torch.tensor(0.0, dtype=torch.float32, device=self.device)
                    )
                    continue

                legal_t = torch.tensor(legal, dtype=torch.long, device=self.device)

                if self.double_dqn:
                    best_local = torch.argmax(online_next_q[i, legal_t])
                else:
                    best_local = torch.argmax(target_next_q[i, legal_t])

                best_action = legal_t[best_local]
                next_q_list.append(target_next_q[i, best_action])

            max_next_q = torch.stack(next_q_list)
            target_q = r + self.gamma * (1.0 - d) * max_next_q

        # ----- loss -----
        td_errors = target_q - current_q
        element_loss = self.loss_fn(current_q, target_q)   # per-sample Huber
        loss = (w * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ----- PER priority update -----
        if self.use_per and per_indices is not None:
            self.replay_buffer.update_priorities(
                per_indices, td_errors.detach().cpu().numpy()
            )

        self.train_steps += 1
        if self.tau > 0.0:
            self.update_target_network()
        elif self.train_steps % self.target_update_freq == 0:
            self.update_target_network()

        return {
            "loss": float(loss.item()),
            "beta": float(beta),
            "mean_td_error": float(td_errors.abs().mean().item()),
            "mean_is_weight": float(w.mean().item()),
            "mean_q": float(current_q.mean().item()),
            "grad_norm": float(grad_norm.item()),
        }

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def update_target_network(self) -> None:
        if self.tau > 0.0:
            # Polyak soft-update: θ_target ← τ·θ_online + (1−τ)·θ_target
            with torch.no_grad():
                for p, t in zip(self.q_net.parameters(), self.target_net.parameters()):
                    t.data.copy_(self.tau * p.data + (1.0 - self.tau) * t.data)
        else:
            # Hard copy (classic DQN)
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "board_size": self.board_size,
                "model_state_dict": self.q_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
                "episodes_seen": self.episodes_seen,
                "config": {
                    "heuristic_weight": self.heuristic_weight,
                    "heuristic_weight_end": self.heuristic_weight_end,
                    "heuristic_decay_episodes": self.heuristic_decay_episodes,
                    "use_per": self.use_per,
                    "per_beta_start": self.per_beta_start,
                    "per_beta_frames": self.per_beta_frames,
                    "double_dqn": self.double_dqn,
                    "tau": self.tau,
                },
                "replay_buffer": self.replay_buffer.get_state(),
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = False) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        if checkpoint["board_size"] != self.board_size:
            raise ValueError(
                f"Checkpoint board_size={checkpoint['board_size']} "
                f"!= agent board_size={self.board_size}"
            )

        self.q_net.load_state_dict(checkpoint["model_state_dict"])

        target_key = "target_state_dict" if "target_state_dict" in checkpoint else "model_state_dict"
        self.target_net.load_state_dict(checkpoint[target_key])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.train_steps = checkpoint.get("train_steps", 0)
        self.episodes_seen = checkpoint.get("episodes_seen", 0)

        if "replay_buffer" in checkpoint:
            self.replay_buffer.set_state(checkpoint["replay_buffer"])

        self.q_net.eval()
        self.target_net.eval()