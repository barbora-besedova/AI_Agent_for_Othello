"""
train_vs_minimax.py — Train a DQN agent against the fast C++ minimax solver.

Uses the same DQNAgent infrastructure as train.py but with the new
FastMinimaxAgent as the primary opponent.  Supports all DQN variants:

    Classic DQN:    python train_vs_minimax.py
    Guided (HDQN):  python train_vs_minimax.py --heuristic_weight 0.2
    Prioritized:    python train_vs_minimax.py --use_per
    Guided+PER:     python train_vs_minimax.py --use_per --heuristic_weight 0.2

The opponent curriculum starts weak (random/greedy) and progressively
introduces the fast bitboard minimax so the agent learns incrementally.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time as time_module
from typing import Dict, Optional

import numpy as np
import torch

from environment import OthelloEnv
from evaluation import evaluate_fair, play_game
from agent import DQNAgent
from diagnostics import MetricsLogger, record_game, save_transcript


# ------------------------------------------------------------------ #
#  Opponents                                                          #
# ------------------------------------------------------------------ #

def _import_opponents():
    from agents.random_agent import RandomAgent
    from agents.greedy_agent import GreedyAgent
    from agents.heuristic_agent import HeuristicAgent
    from agents.cpp_minimax_agent import FastMinimaxAgent

    return {
        "random":        RandomAgent,
        "greedy":        GreedyAgent,
        "heuristic":     HeuristicAgent,
        "fast_minimax":  FastMinimaxAgent,
    }


def _minimax_factory(board_size: int, time_limit: float,
                      max_depth: int | None = None):
    """Factory that returns a FastMinimaxAgent instance (matches the
    ``agent_b_class(board_size)`` signature used by evaluate_fair)."""
    from agents.cpp_minimax_agent import FastMinimaxAgent
    return FastMinimaxAgent(board_size=board_size, time_limit=time_limit,
                            max_depth=max_depth)


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_final_reward(winner: int, agent_player: int) -> float:
    if winner == agent_player:
        return 1.0
    if winner == 0:
        return 0.0
    return -1.0


def _make_opponent(opponent_type: str, board_size: int,
                   opponent_classes: Dict, minimax_time_limit: float,
                   max_depth: int | None = None):
    cls = opponent_classes[opponent_type]
    if opponent_type == "fast_minimax":
        return cls(board_size=board_size, time_limit=minimax_time_limit,
                   max_depth=max_depth)
    return cls(board_size=board_size)


def _choose_opponent_from_curriculum(
    episode: int,
    num_episodes: int,
    final_minimax_weight: float = 0.65,
    minimax_start_progress: float = 0.25,
    minimax_full_progress: float = 0.75,
) -> str:
    progress = episode / num_episodes
    mm_mid_progress = (minimax_start_progress + minimax_full_progress) / 2

    # Scale factors for each stage
    mm_scale_lo = 0.0  # below start
    mm_mid_fraction = (25 / 65)  # ~0.385 — midway minimax fraction
    # Interpolate the "mid" scale so it sits proportionally along the ramp
    # e.g., mm_scale_mid at the midpoint of the linear ramp
    # The original mid point (0.385 of final) sat at progress 0.50.
    # We linearly interpolate: scale goes from 0 at start to 1 at full_progress.
    # At midpoint (start+full)/2, scale should be 0.5.
    # But to preserve the original "one stage before full" feel, we use
    # a 3-stage ramp with 0 → 0.385 → 1 at start/mid/full.
    mm_scale_mid = 0.385
    mm_scale_full = 1.0

    def _stage_weights(mm_scale: float):
        r, g, h, mm = 0.05, 0.10, 0.20, final_minimax_weight
        mm *= mm_scale
        leftover = (r + g + h) * (1 - mm_scale)
        return [r + leftover * r / (r + g + h),
                g + leftover * g / (r + g + h),
                h + leftover * h / (r + g + h),
                mm]

    if progress < minimax_start_progress:
        names =   ["random", "greedy", "heuristic"]
        w = _stage_weights(mm_scale_lo)
        weights = [w[0] + w[3] / 3, w[1] + w[3] / 3, w[2] + w[3] / 3]
    elif progress < mm_mid_progress:
        names =   ["random", "greedy", "heuristic", "fast_minimax"]
        weights = _stage_weights(mm_scale_mid)
    elif progress < minimax_full_progress:
        names =   ["random", "greedy", "heuristic", "fast_minimax"]
        weights = _stage_weights(0.45 / 0.65)
    else:
        names =   ["random", "greedy", "heuristic", "fast_minimax"]
        weights = [0.05, 0.10, 0.20, final_minimax_weight]

    return random.choices(names, weights=weights, k=1)[0]


def _epsilon_for_opponent(
    base_epsilon: float,
    opponent_type: str,
) -> float:
    floors = {
        "greedy": 0.15,
        "heuristic": 0.15,
        "fast_minimax": 0.20,
    }
    return max(base_epsilon, floors.get(opponent_type, 0.0))


# ------------------------------------------------------------------ #
#  Main training function                                             #
# ------------------------------------------------------------------ #

def train(
    # --- environment ---
    board_size: int = 6,
    # --- episodes ---
    num_episodes: int = 3_000,
    start_episode: int = 1,
    # --- epsilon schedule ---
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.999,
    # --- opponent ---
    opponent_type: str = "curriculum",
    minimax_time_limit: float = 1.0,
    minimax_max_depth: int | None = None,
    final_minimax_weight: float = 0.65,
    minimax_start_progress: float = 0.25,
    minimax_full_progress: float = 0.75,
    # --- agent hyper-parameters ---
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 50_000,
    target_update_freq: int = 500,
    learning_starts: int = 1_000,
    double_dqn: bool = True,
    heuristic_weight: float = 0.0,
    use_per: bool = False,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_frames: int = 100_000,
    tau: float = 0.005,
    # --- I/O ---
    model_path: str = "models/minimax_trained.pth",
    load_model_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    best_model_path: Optional[str] = None,
    # --- time limit ---
    max_minutes: Optional[float] = None,
    # --- logging ---
    print_every: int = 50,
    eval_every: int = 200,
    save_every: int = 500,
    seed: int = 42,
    profile: bool = False,
    # --- diagnostics ---
    record_game_eps: str = "1200,3000",
    n_record_games: int = 3,
) -> Dict:
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    set_seed(seed)

    opponent_classes = _import_opponents()

    log_path = os.path.splitext(model_path)[0] + "_train.csv"
    train_log = MetricsLogger(log_path, [
        "episode", "epsilon", "avg_reward", "win_rate",
        "loss", "mean_q", "grad_norm", "mean_td_error", "beta",
    ])

    record_eps_set = set()
    for s in record_game_eps.split(","):
        s = s.strip()
        if s:
            try:
                record_eps_set.add(int(s))
            except ValueError:
                pass

    env = OthelloEnv(board_size=board_size)

    start_wall = time_module.perf_counter()

    if profile:
        prof_agent_select = 0.0
        prof_opponent_select = 0.0
        prof_env_step = 0.0
        prof_train_step = 0.0
        prof_store = 0.0
        prof_count = 0

    agent = DQNAgent(
        board_size=board_size,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
        learning_starts=learning_starts,
        double_dqn=double_dqn,
        heuristic_weight=heuristic_weight,
        use_per=use_per,
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        per_beta_frames=per_beta_frames,
        tau=tau,
    )

    if load_model_path is not None:
        agent.load(load_model_path, load_optimizer=True)
        # Override to the current learning rate (may differ from checkpoint)
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = learning_rate
        print(f"Loaded weights + optimizer from: {load_model_path}")
        print(f"  replay buffer size: {len(agent.replay_buffer)}")
        print(f"  train steps: {agent.train_steps}")

    agent_label = (
        f"{'guided_' if heuristic_weight > 0 else ''}"
        f"{'per_' if use_per else ''}"
        f"dqn"
    )
    print(f"Training: {agent_label} | board={board_size}x{board_size} | "
          f"opponent={opponent_type} | episodes={num_episodes} | "
          f"minimax_time_limit={minimax_time_limit}s | "
          f"minimax_max_depth={minimax_max_depth}")

    epsilon = epsilon_start
    rewards_history, win_history, loss_history = [], [], []
    beta_history, td_error_history = [], []
    last_train_info = None
    wins = draws = losses = 0

    opponent_stats = {
        name: {"games": 0, "wins": 0, "draws": 0, "losses": 0}
        for name in opponent_classes
    }

    best_minimax_score = -1.0

    total_game_time = 0.0
    total_games = 0

    if start_episode > 1:
        print(f"Resuming from episode {start_episode} (epsilon={epsilon:.4f})")

    # ============================================================== #
    #  Episode loop                                                    #
    # ============================================================== #
    for episode in range(start_episode, num_episodes + 1):
        # --- time limit check (graceful exit) ---
        if max_minutes is not None:
            elapsed = (time_module.perf_counter() - start_wall) / 60.0
            if elapsed >= max_minutes:
                print(f"\nTime limit {max_minutes} min reached ({elapsed:.1f} min); saving and stopping.")
                agent.save(model_path)
                train_log.close()
                print(f"Model saved -> {model_path}")
                break

        obs = env.reset()
        done = False

        if opponent_type == "curriculum":
            cur_opp_type = _choose_opponent_from_curriculum(
                episode, num_episodes, final_minimax_weight,
                minimax_start_progress, minimax_full_progress)
        else:
            cur_opp_type = opponent_type

        opponent = _make_opponent(cur_opp_type, board_size,
                                  opponent_classes, minimax_time_limit,
                                  minimax_max_depth)

        cur_epsilon = _epsilon_for_opponent(epsilon, cur_opp_type)
        opponent_stats[cur_opp_type]["games"] += 1

        agent_player = 1 if episode % 2 == 0 else -1

        if env.current_player != agent_player:
            t_opp = time_module.perf_counter()
            opp_action = opponent.select_action(obs)
            if profile: prof_opponent_select += time_module.perf_counter() - t_opp
            t_env = time_module.perf_counter()
            obs, _, done, _ = env.step(opp_action)
            if profile: prof_env_step += time_module.perf_counter() - t_env

        episode_reward = 0.0
        t0 = time_module.perf_counter()

        while not done:
            state_obs = obs

            t_agt = time_module.perf_counter()
            action = agent.select_action(state_obs, epsilon=cur_epsilon)
            if profile: prof_agent_select += time_module.perf_counter() - t_agt

            t_env = time_module.perf_counter()
            obs_after_agent, _, done, info = env.step(action)
            if profile: prof_env_step += time_module.perf_counter() - t_env

            if done:
                reward = compute_final_reward(info["winner"], agent_player)
                t_st = time_module.perf_counter()
                agent.store_transition(
                    state_obs, action, reward, obs_after_agent, True)
                if profile: prof_store += time_module.perf_counter() - t_st
                t_tr = time_module.perf_counter()
                train_info = agent.train_step()
                if profile: prof_train_step += time_module.perf_counter() - t_tr
                if train_info is not None:
                    last_train_info = train_info
                    loss_history.append(train_info["loss"])
                    if use_per:
                        beta_history.append(train_info["beta"])
                        td_error_history.append(train_info["mean_td_error"])
                episode_reward = reward
                break

            t_opp = time_module.perf_counter()
            opp_action = opponent.select_action(obs_after_agent)
            if profile: prof_opponent_select += time_module.perf_counter() - t_opp

            t_env = time_module.perf_counter()
            obs_after_opp, _, done, info = env.step(opp_action)
            if profile: prof_env_step += time_module.perf_counter() - t_env

            if done:
                reward = compute_final_reward(info["winner"], agent_player)
            else:
                reward = 0.0

            t_st = time_module.perf_counter()
            agent.store_transition(
                state_obs, action, reward, obs_after_opp, done)
            if profile: prof_store += time_module.perf_counter() - t_st

            t_tr = time_module.perf_counter()
            train_info = agent.train_step()
            if profile: prof_train_step += time_module.perf_counter() - t_tr
            if train_info is not None:
                last_train_info = train_info
                loss_history.append(train_info["loss"])
                if use_per:
                    beta_history.append(train_info["beta"])
                    td_error_history.append(train_info["mean_td_error"])

            obs = obs_after_opp
            episode_reward = reward

        game_time = time_module.perf_counter() - t0
        total_game_time += game_time
        total_games += 1
        if profile: prof_count += 1

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        rewards_history.append(episode_reward)
        if episode_reward > 0:
            wins += 1
            win_history.append(1.0)
            opponent_stats[cur_opp_type]["wins"] += 1
        elif episode_reward < 0:
            losses += 1
            win_history.append(0.0)
            opponent_stats[cur_opp_type]["losses"] += 1
        else:
            draws += 1
            win_history.append(0.0)
            opponent_stats[cur_opp_type]["draws"] += 1

        # --- console log ---
        if episode % print_every == 0 or episode == 1:
            avg_r = np.mean(rewards_history[-100:])
            avg_w = np.mean(win_history[-100:])
            avg_time = total_game_time / max(total_games, 1)

            if last_train_info is None:
                extra = "learning not started"
            else:
                extra = (
                    f"loss={last_train_info['loss']:.4f}"
                    f" q={last_train_info['mean_q']:.3f}"
                    f" gn={last_train_info['grad_norm']:.2f}"
                )
                if use_per:
                    extra += (
                        f" | beta={last_train_info['beta']:.3f}"
                        f" | td={last_train_info['mean_td_error']:.4f}"
                    )

            extra_line = (
                f"[{episode:>5}/{num_episodes}] "
                f"opp={cur_opp_type:<13} eps={epsilon:.3f} | "
                f"avg_r={avg_r:.3f} win%={avg_w:.3f} | "
                f"{extra} | W/D/L={wins}/{draws}/{losses} | "
                f"time={avg_time:.1f}s/game"
            )

            if profile and prof_count > 0:
                total = prof_agent_select + prof_opponent_select + prof_env_step + prof_train_step + prof_store
                extra_line += (
                    f"\n         prof: agent={prof_agent_select/total*100:.0f}% "
                    f"opp={prof_opponent_select/total*100:.0f}% "
                    f"env={prof_env_step/total*100:.0f}% "
                    f"train={prof_train_step/total*100:.0f}% "
                    f"store={prof_store/total*100:.0f}%"
                )

            print(extra_line)

            train_log.log(
                episode=episode,
                epsilon=f"{epsilon:.4f}",
                avg_reward=f"{avg_r:.3f}",
                win_rate=f"{avg_w:.3f}",
                loss=last_train_info["loss"] if last_train_info else "",
                mean_q=last_train_info["mean_q"] if last_train_info else "",
                grad_norm=last_train_info["grad_norm"] if last_train_info else "",
                mean_td_error=last_train_info["mean_td_error"]
                    if last_train_info and use_per else "",
                beta=last_train_info["beta"] if last_train_info and use_per else "",
            )

        # --- periodic evaluation ---
        if episode % eval_every == 0:
            print("\n-- Evaluation (no exploration) --")
            agent.q_net.eval()

            eval_log = MetricsLogger(
                os.path.splitext(model_path)[0] + f"_eval_ep{episode}.csv",
                ["opponent", "role", "wins", "draws", "losses",
                 "win_rate", "score"],
            )
            for opp_name, opp_class in opponent_classes.items():
                if opp_name == "fast_minimax":
                    mm_fac = lambda bs: _minimax_factory(
                        bs, max(0.25, minimax_time_limit * 0.5),
                        minimax_max_depth)
                    # aggregate
                    n_games = 50
                    result = evaluate_fair(
                        agent_a=agent, agent_b_class=mm_fac,
                        board_size=board_size, n_games=n_games,
                    )
                    win_rate = result['win_rate']
                    print(
                        f"  {opp_name:<13} score={result['score']:.3f} "
                        f"win={win_rate:.3f} "
                        f"W/D/L={result['wins']}/{result['draws']}/{result['losses']}"
                    )
                    eval_log.log(opponent=opp_name, role="aggregate",
                                 wins=result['wins'], draws=result['draws'],
                                 losses=result['losses'],
                                 win_rate=f"{win_rate:.3f}",
                                 score=f"{result['score']:.3f}")
                    # per-color
                    n_per = n_games // 2
                    wins1 = draws1 = losses1 = 0
                    for _ in range(n_per):
                        opp = mm_fac(board_size)
                        w = play_game(agent_1=agent, agent_2=opp,
                                      board_size=board_size,
                                      random_opening_plies=2)
                        if w == 1: wins1 += 1
                        elif w == 0: draws1 += 1
                        else: losses1 += 1
                    wr1 = wins1 / n_per
                    score1 = (wins1 + 0.5 * draws1) / n_per
                    print(f"    as black:  win={wr1:.3f} W/D/L={wins1}/{draws1}/{losses1}")
                    eval_log.log(opponent=opp_name, role="as_black",
                                 wins=wins1, draws=draws1, losses=losses1,
                                 win_rate=f"{wr1:.3f}", score=f"{score1:.3f}")

                    wins2 = draws2 = losses2 = 0
                    for _ in range(n_per):
                        opp = mm_fac(board_size)
                        w = play_game(agent_1=opp, agent_2=agent,
                                      board_size=board_size,
                                      random_opening_plies=2)
                        if w == -1: wins2 += 1
                        elif w == 0: draws2 += 1
                        else: losses2 += 1
                    wr2 = wins2 / n_per
                    score2 = (wins2 + 0.5 * draws2) / n_per
                    print(f"    as white:  win={wr2:.3f} W/D/L={wins2}/{draws2}/{losses2}")
                    eval_log.log(opponent=opp_name, role="as_white",
                                 wins=wins2, draws=draws2, losses=losses2,
                                 win_rate=f"{wr2:.3f}", score=f"{score2:.3f}")

                    # combined
                    comb_w = wins1 + wins2
                    comb_d = draws1 + draws2
                    comb_l = losses1 + losses2
                    comb_n = comb_w + comb_d + comb_l
                    comb_score = (comb_w + 0.5 * comb_d) / comb_n
                    print(f"    combined: win={comb_w/comb_n:.3f} W/D/L={comb_w}/{comb_d}/{comb_l}")

                    # best model tracking (based on minimax combined score)
                    if best_model_path is not None and comb_score > best_minimax_score:
                        best_minimax_score = comb_score
                        agent.save(best_model_path)
                        print(f"  [new best model -> {best_model_path}] (score={comb_score:.3f})")
                else:
                    n_games = 100
                    result = evaluate_fair(
                        agent_a=agent, agent_b_class=opp_class,
                        board_size=board_size, n_games=n_games,
                    )
                    win_rate = result['win_rate']
                    print(
                        f"  {opp_name:<13} score={result['score']:.3f} "
                        f"win={win_rate:.3f} "
                        f"W/D/L={result['wins']}/{result['draws']}/{result['losses']}"
                    )
                    eval_log.log(opponent=opp_name, role="aggregate",
                                 wins=result['wins'], draws=result['draws'],
                                 losses=result['losses'],
                                 win_rate=f"{win_rate:.3f}",
                                 score=f"{result['score']:.3f}")
            eval_log.close()

            # --- record full games at specified episodes ---
            if episode in record_eps_set:
                games_dir = os.path.join(
                    os.path.dirname(model_path) or ".",
                    "games")
                print(f"  Recording {n_record_games} games vs minimax to {games_dir}/ ...")
                mm = _minimax_factory(
                    board_size, max(0.25, minimax_time_limit * 0.5),
                    minimax_max_depth)
                for gi in range(n_record_games):
                    transcript = record_game(
                        agent, mm, board_size=board_size,
                        record_q=True, device=agent.device)
                    path = os.path.join(
                        games_dir, f"ep{episode}_game{gi}.json")
                    save_transcript(transcript, path)
                    transcript2 = record_game(
                        mm, agent, board_size=board_size,
                        record_q=True, device=agent.device)
                    path2 = os.path.join(
                        games_dir, f"ep{episode}_game{gi}_swapped.json")
                    save_transcript(transcript2, path2)

            agent.q_net.train()
            print()

        # --- periodic save ---
        if episode % save_every == 0:
            agent.save(model_path)
            print(f"  [saved -> {model_path}]")
            if checkpoint_dir is not None:
                cp_path = os.path.join(checkpoint_dir, f"model_ep{episode}.pth")
                agent.save(cp_path)
                print(f"  [checkpoint -> {cp_path}]")

            # resume state (episode + epsilon for continuing later)
            resume_state = {
                "episode": episode,
                "epsilon": epsilon,
                "model_path": model_path,
            }
            resume_path = os.path.join(
                os.path.dirname(model_path) or ".", "resume_state.json")
            with open(resume_path, "w", encoding="utf-8") as f:
                json.dump(resume_state, f, indent=2)

    # --- close log ---
    train_log.close()

    # --- final save ---
    agent.save(model_path)
    print(f"\nModel saved -> {model_path}")

    print("\nResults by opponent type:")
    for opp_name, stats in opponent_stats.items():
        if stats["games"] > 0:
            print(
                f"  {opp_name:<13} games={stats['games']:>5}  "
                f"W/D/L={stats['wins']}/{stats['draws']}/{stats['losses']}"
            )

    return {
        "agent": agent,
        "rewards": rewards_history,
        "wins": win_history,
        "losses": loss_history,
        "betas": beta_history,
        "td_errors": td_error_history,
    }


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a DQN agent against the fast C++ minimax solver.")

    p.add_argument("--board_size",          type=int,   default=6)
    p.add_argument("--num_episodes",        type=int,   default=3_000)
    p.add_argument("--epsilon_start",       type=float, default=1.0)
    p.add_argument("--epsilon_end",         type=float, default=0.05)
    p.add_argument("--epsilon_decay",       type=float, default=0.999)
    p.add_argument("--opponent_type",       type=str,   default="curriculum",
                   choices=["curriculum", "random", "greedy",
                            "heuristic", "fast_minimax"])
    p.add_argument("--minimax_time_limit",  type=float, default=1.0,
                   help="Seconds per move for FastMinimaxAgent.")
    p.add_argument("--minimax_max_depth",   type=int,   default=None,
                   help="Max search depth for minimax (None = no limit).")
    p.add_argument("--final_minimax_weight", type=float, default=0.65,
                   help="Minimax proportion in final curriculum stage (default 0.65).")
    p.add_argument("--minimax_start_progress", type=float, default=0.25,
                   help="Progress fraction where minimax first appears (default 0.25).")
    p.add_argument("--minimax_full_progress", type=float, default=0.75,
                   help="Progress fraction where minimax reaches full weight (default 0.75).")
    p.add_argument("--learning_rate",       type=float, default=1e-3)
    p.add_argument("--gamma",               type=float, default=0.99)
    p.add_argument("--batch_size",          type=int,   default=64)
    p.add_argument("--buffer_capacity",     type=int,   default=50_000)
    p.add_argument("--target_update_freq",  type=int,   default=500)
    p.add_argument("--learning_starts",     type=int,   default=1_000)
    p.add_argument("--heuristic_weight",    type=float, default=0.0,
                   help="Heuristic bonus weight. 0 = classic DQN.")
    p.add_argument("--use_per",             action="store_true",
                   help="Prioritized Experience Replay (PDQN).")
    p.add_argument("--per_alpha",           type=float, default=0.6)
    p.add_argument("--per_beta_start",      type=float, default=0.4)
    p.add_argument("--per_beta_frames",     type=int,   default=100_000)
    p.add_argument("--model_path",          type=str,
                   default="models/minimax_trained.pth")
    p.add_argument("--load_model_path",     type=str,   default=None)
    p.add_argument("--checkpoint_dir",      type=str,   default=None,
                   help="Dir for episode-numbered checkpoints.")
    p.add_argument("--best_model_path",     type=str,   default=None,
                   help="Path to save best model (by minimax eval score).")
    p.add_argument("--max_minutes",         type=float, default=None,
                   help="Stop training after this many minutes (graceful save).")
    p.add_argument("--print_every",         type=int,   default=50)
    p.add_argument("--eval_every",          type=int,   default=200)
    p.add_argument("--save_every",          type=int,   default=500)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--profile",             action="store_true",
                   help="Print timing breakdown of agent/opponent/env/train.")
    p.add_argument("--record_game_eps",     type=str,   default="1200,3000",
                   help="Comma-separated episode numbers to record full games.")
    p.add_argument("--n_record_games",      type=int,   default=3,
                   help="Number of full games to record at each checkpoint.")
    p.add_argument("--no_double_dqn",       action="store_true",
                   help="Disable Double DQN.")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        board_size=args.board_size,
        num_episodes=args.num_episodes,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        opponent_type=args.opponent_type,
        minimax_time_limit=args.minimax_time_limit,
        minimax_max_depth=args.minimax_max_depth,
        final_minimax_weight=args.final_minimax_weight,
        minimax_start_progress=args.minimax_start_progress,
        minimax_full_progress=args.minimax_full_progress,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update_freq,
        learning_starts=args.learning_starts,
        double_dqn=not args.no_double_dqn,
        heuristic_weight=args.heuristic_weight,
        use_per=args.use_per,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_frames=args.per_beta_frames,
        model_path=args.model_path,
        load_model_path=args.load_model_path,
        checkpoint_dir=args.checkpoint_dir,
        best_model_path=args.best_model_path,
        max_minutes=args.max_minutes,
        print_every=args.print_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
        profile=args.profile,
        record_game_eps=args.record_game_eps,
        n_record_games=args.n_record_games,
    )
