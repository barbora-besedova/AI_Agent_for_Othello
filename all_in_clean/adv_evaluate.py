"""
adv_evaluate.py — Evaluate checkpoints or plot progress from existing CSV results.

Default mode: reads latest_eval_ep*.csv files from results/ and plots
winrate progress over training episodes — no GPU/time needed.

Use --eval to run actual evaluation against scripted opponents.
"""

import sys, os, glob, re
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import argparse

DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "Training", "experiments_002", "results",
)
DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(__file__),
    "Training", "experiments_002", "model_checkpoints",
)


def plot_from_csv(results_dir):
    csv_files = sorted(
        glob.glob(os.path.join(results_dir, "latest_eval_ep*.csv")),
        key=lambda p: int(
            re.search(r"latest_eval_ep(\d+)\.csv", os.path.basename(p)).group(1)
        ),
    )

    if not csv_files:
        print(f"No latest_eval_ep*.csv files found in {results_dir}")
        return

    episodes = []
    data = {}

    for csv_file in csv_files:
        ep = int(
            re.search(r"latest_eval_ep(\d+)\.csv", os.path.basename(csv_file)).group(1)
        )
        episodes.append(ep)

        with open(csv_file) as f:
            f.readline()
            for line in f:
                parts = line.strip().split(",")
                opp, role, wins, draws, losses, wr, score = parts
                key = opp if role == "aggregate" else f"{opp}_{role}"
                data.setdefault(key, []).append(float(score))

    header = f"{'Episode':<10}" + "".join(f"{k:<14s}" for k in data)
    print("\n" + "=" * (10 + 14 * len(data)))
    print(header)
    print("-" * (10 + 14 * len(data)))
    for i, ep in enumerate(episodes):
        row = f"{ep:<10d}" + "".join(f"{data[k][i]:<14.3f}" for k in data)
        print(row)
    print("=" * (10 + 14 * len(data)))

    plt.figure(figsize=(10, 6))
    for key, scores in data.items():
        plt.plot(episodes, scores, marker="o", label=key)
    plt.xlabel("Training Episodes")
    plt.ylabel("Score (win_rate)")
    plt.title(f"Win Rate Progress — {os.path.basename(os.path.dirname(results_dir))}")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(results_dir, "..", "eval_winrate_vs_episodes.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {out_path}")


def run_eval(models_dir, n_games=100, depth=5):
    from evaluation import evaluate_fair
    from agent import DQNAgent
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent
    from agents.cpp_minimax_agent import FastMinimaxAgent

    opponents = {
        "random": RandomAgent,
        "heuristic": HeuristicAgent,
        f"minimax_d{depth}": lambda bs: FastMinimaxAgent(board_size=bs, max_depth=depth),
    }

    ckpts = sorted(
        glob.glob(os.path.join(models_dir, "model_ep*.pth")),
        key=lambda p: int(
            re.search(r"model_ep(\d+)\.pth", os.path.basename(p)).group(1)
        ),
    )

    if not ckpts:
        print(f"No model_ep*.pth files found in {models_dir}")
        return

    episodes = []
    results = {name: [] for name in opponents}

    for ckpt in ckpts:
        ep = int(re.search(r"model_ep(\d+)\.pth", os.path.basename(ckpt)).group(1))
        episodes.append(ep)

        agent = DQNAgent(board_size=6, use_per=True, heuristic_weight=0.2)
        agent.load(ckpt)
        agent.q_net.eval()

        print(f"\n--- Episode {ep} ({os.path.basename(ckpt)}) ---")
        for opp_name, opp_fn in opponents.items():
            res = evaluate_fair(
                agent, opp_fn,
                board_size=6,
                n_games=n_games,
                random_opening_plies=2,
            )
            score = res["score"]
            results[opp_name].append(score)
            print(f"  vs {opp_name:<14s}  score={score:.3f}  "
                  f"({res['wins']}W/{res['draws']}D/{res['losses']}L)")

    header = f"{'Episode':<10}" + "".join(f"{name:<14s}" for name in opponents)
    print("\n" + "=" * (10 + 14 * len(opponents)))
    print(header)
    print("-" * (10 + 14 * len(opponents)))
    for i, ep in enumerate(episodes):
        row = f"{ep:<10d}" + "".join(f"{results[name][i]:<14.3f}" for name in opponents)
        print(row)
    print("=" * (10 + 14 * len(opponents)))

    plt.figure(figsize=(10, 6))
    for opp_name in opponents:
        plt.plot(episodes, results[opp_name], marker="o", label=opp_name)
    plt.xlabel("Training Episodes")
    plt.ylabel("Score (win_rate)")
    plt.title(f"Model Win Rate vs Opponents — {os.path.basename(models_dir)}")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(models_dir, "..", "eval_winrate_vs_episodes.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot evaluation progress from CSV results, or re-evaluate models."
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run actual evaluation (default: plot from existing CSVs)",
    )
    parser.add_argument(
        "--models-dir", default=DEFAULT_MODELS_DIR,
        help="Directory with model_ep*.pth checkpoints",
    )
    parser.add_argument(
        "--results-dir", default=DEFAULT_RESULTS_DIR,
        help="Directory with latest_eval_ep*.csv files",
    )
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--depth", type=int, default=5,
                        help="Minimax search depth for --eval")

    args = parser.parse_args()

    if args.eval:
        run_eval(args.models_dir, args.n_games, args.depth)
    else:
        plot_from_csv(args.results_dir)
