"""
server.py — Web UI backend for the Othello project.

Place this file in the SAME directory as play.py (so `environment` and `agent`
import cleanly), put index.html next to it, then:

    python server.py                       # auto-detects a model in ./ or ./models
    python server.py --model models/dqn.pth
    python server.py --board_size 6 --port 8000

Open http://127.0.0.1:8000 in your browser.

No external dependencies beyond what play.py already needs (numpy, torch,
your environment.py / agent.py). Uses only the Python standard library for
the web layer.

Inferred engine API (matches play.py):
  env = OthelloEnv(board_size)
  env.reset() -> obs ; env.step(a) -> (obs, reward, done, info) ; info["winner"]
  env.board_abs[r,c] in {1,-1,0} ; env.current_player in {1,-1}
  env.turn_count ; env.pass_action ; env.get_legal_actions(player) -> [int,...]
  agent.select_action(obs[, epsilon=..]) -> action
If your method names differ, adjust the small wrappers below.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from environment import OthelloEnv  # noqa: E402

# ── Global mutable game state (guarded by LOCK) ─────────────────────────────
LOCK = threading.RLock()

ENV: OthelloEnv | None = None
BOARD_SIZE = 6
EPSILON = 0.0

# move_history is the full played line (list of int actions, incl. pass_action).
MOVE_HISTORY: list[int] = []
VIEW_CURSOR = 0  # 0..len(MOVE_HISTORY); position currently being viewed

# Per-side agent objects and labels. None => human.
AGENTS: dict[int, object | None] = {1: None, -1: None}
PLAYER_SPEC: dict[int, dict] = {1: {"type": "human"}, -1: {"type": "human"}}

PRESET_MODEL: str | None = None


# ── Agent loading ───────────────────────────────────────────────────────────

def _load_model_agent(path: str):
    from agent import DQNAgent
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    a = DQNAgent(board_size=BOARD_SIZE)
    a.load(path)
    try:
        a.q_net.eval()
    except Exception:
        pass
    return a


def _load_rule_agent(name: str):
    mods = {
        "random":    ("agents.random_agent",    "RandomAgent"),
        "greedy":    ("agents.greedy_agent",     "GreedyAgent"),
        "heuristic": ("agents.heuristic_agent",  "HeuristicAgent"),
        "minimax":   ("agents.minimax_agent",    "MinimaxAgent"),
    }
    if name not in mods:
        raise ValueError(f"Unknown rule agent '{name}'")
    mod, cls = mods[name]
    module = __import__(mod, fromlist=[cls])
    return getattr(module, cls)(BOARD_SIZE)


def _build_agent(spec: dict):
    t = spec.get("type")
    if t == "human":
        return None
    if t == "model":
        return _load_model_agent(spec["path"])
    if t == "rule":
        return _load_rule_agent(spec["name"])
    raise ValueError(f"Bad player spec: {spec}")


def _agent_act(agent, obs, legal: list[int]) -> int:
    """Call select_action robustly; fall back to a legal move if it misbehaves."""
    fn = agent.select_action
    action = None
    try:
        code = getattr(fn, "__code__", None)
        if code is not None and "epsilon" in code.co_varnames:
            action = fn(obs, epsilon=EPSILON)
        else:
            action = fn(obs)
    except TypeError:
        action = fn(obs)
    if action not in legal:
        # Defensive: never apply an illegal action.
        action = legal[0] if len(legal) == 1 else int(np.random.choice(legal))
    return int(action)


# ── Engine helpers (call only while holding LOCK) ───────────────────────────

def _rebuild(cursor: int):
    """Reset env and replay MOVE_HISTORY[:cursor]. Returns (obs, last, done, info)."""
    obs = ENV.reset()
    last = {1: None, -1: None}  # last *placing* action per side
    done, info = False, {}
    pa = ENV.pass_action
    for a in MOVE_HISTORY[:cursor]:
        player = ENV.current_player
        if a != pa:
            last[player] = a
        obs, _r, done, info = ENV.step(a)
    return obs, last, done, info


def _cell(action):
    if action is None or action == ENV.pass_action:
        return None
    r, c = divmod(int(action), ENV.board_size)
    return {"action": int(action), "row": r, "col": c}


def state_dict(extra: dict | None = None) -> dict:
    """Full UI state at VIEW_CURSOR. Call while holding LOCK."""
    obs, last, done, info = _rebuild(VIEW_CURSOR)
    n = ENV.board_size
    pa = ENV.pass_action
    cp = ENV.current_player

    legal = list(ENV.get_legal_actions(cp))
    legal_cells, can_pass = [], False
    for a in legal:
        if a == pa:
            can_pass = True
        else:
            r, c = divmod(int(a), n)
            legal_cells.append({"action": int(a), "row": r, "col": c})
    must_pass = can_pass and not legal_cells

    board = np.asarray(ENV.board_abs).astype(int).tolist()
    p1 = int(np.sum(np.asarray(ENV.board_abs) == 1))
    p2 = int(np.sum(np.asarray(ENV.board_abs) == -1))

    winner = None
    if done:
        winner = int(info.get("winner", 0)) if isinstance(info, dict) else 0

    spec_label = {}
    for side in (1, -1):
        sp = PLAYER_SPEC[side]
        if sp["type"] == "human":
            spec_label[side] = "Human"
        elif sp["type"] == "model":
            spec_label[side] = "Model: " + os.path.basename(sp["path"])
        else:
            spec_label[side] = "Rule: " + sp["name"]

    is_agent_turn = AGENTS[cp] is not None

    out = {
        "board_size": n,
        "board": board,
        "current_player": int(cp),
        "legal_moves": legal_cells,
        "can_pass": can_pass,
        "must_pass": must_pass,
        "scores": {"p1": p1, "p2": p2},
        "turn_count": int(getattr(ENV, "turn_count", VIEW_CURSOR)),
        "last_move": {"p1": _cell(last[1]), "p2": _cell(last[-1])},
        "most_recent": _cell(MOVE_HISTORY[VIEW_CURSOR - 1]) if VIEW_CURSOR > 0 else None,
        "most_recent_player": None,
        "done": done,
        "winner": winner,
        "history_len": len(MOVE_HISTORY),
        "view_cursor": VIEW_CURSOR,
        "at_latest": VIEW_CURSOR == len(MOVE_HISTORY),
        "is_agent_turn": is_agent_turn and not done,
        "players": {"1": {"type": PLAYER_SPEC[1]["type"], "label": spec_label[1]},
                    "-1": {"type": PLAYER_SPEC[-1]["type"], "label": spec_label[-1]}},
        "epsilon": EPSILON,
        "history": [_history_entry(i) for i in range(len(MOVE_HISTORY))],
    }
    if VIEW_CURSOR > 0:
        # player who made the most recent (viewed) move
        obs2, _l, _d, _i = _rebuild(VIEW_CURSOR - 1)
        out["most_recent_player"] = int(ENV.current_player)
    if extra:
        out.update(extra)
    return out


def _history_entry(i: int) -> dict:
    """Describe move i (player + cell/pass) by replaying up to i."""
    _o, _l, _d, _inf = _rebuild(i)
    player = int(ENV.current_player)
    a = MOVE_HISTORY[i]
    if a == ENV.pass_action:
        return {"index": i, "player": player, "pass": True}
    r, c = divmod(int(a), ENV.board_size)
    return {"index": i, "player": player, "row": r, "col": c, "pass": False}


def _hint_agent(cp: int):
    """Agent to use for a suggestion at current player's position."""
    if AGENTS[cp] is not None:
        return AGENTS[cp]
    if AGENTS[-cp] is not None:
        return AGENTS[-cp]
    return None


# ── Config / actions ────────────────────────────────────────────────────────

def apply_config(cfg: dict) -> dict:
    global ENV, BOARD_SIZE, EPSILON, MOVE_HISTORY, VIEW_CURSOR, AGENTS, PLAYER_SPEC
    with LOCK:
        BOARD_SIZE = int(cfg.get("board_size", BOARD_SIZE))
        EPSILON = float(cfg.get("epsilon", EPSILON))
        players = cfg.get("players", {"1": {"type": "human"}, "-1": {"type": "human"}})

        ENV = OthelloEnv(board_size=BOARD_SIZE)
        MOVE_HISTORY = []
        VIEW_CURSOR = 0

        new_spec, new_agents, errors = {}, {}, []
        for side_key, spec in players.items():
            side = int(side_key)
            try:
                new_agents[side] = _build_agent(spec)
                new_spec[side] = spec
            except Exception as e:  # noqa: BLE001
                new_agents[side] = None
                new_spec[side] = {"type": "human"}
                errors.append(f"side {side}: {e}")
        AGENTS = new_agents
        PLAYER_SPEC = new_spec
        return state_dict({"errors": errors})


def do_move(action: int) -> dict:
    global MOVE_HISTORY, VIEW_CURSOR
    with LOCK:
        if ENV is None:
            raise RuntimeError("not configured")
        _o, _l, done, _i = _rebuild(VIEW_CURSOR)
        if done:
            return state_dict({"error": "game over"})
        legal = list(ENV.get_legal_actions(ENV.current_player))
        if action not in legal:
            return state_dict({"error": "illegal move"})
        # Branch if viewing the past.
        MOVE_HISTORY = MOVE_HISTORY[:VIEW_CURSOR]
        MOVE_HISTORY.append(int(action))
        VIEW_CURSOR += 1
        return state_dict()


def do_agent_step() -> dict:
    global MOVE_HISTORY, VIEW_CURSOR
    with LOCK:
        if VIEW_CURSOR != len(MOVE_HISTORY):
            return state_dict({"error": "not at latest"})
        obs, _l, done, _i = _rebuild(VIEW_CURSOR)
        if done:
            return state_dict({"error": "game over"})
        cp = ENV.current_player
        agent = AGENTS[cp]
        if agent is None:
            return state_dict({"error": "current player is human"})
        legal = list(ENV.get_legal_actions(cp))
        action = _agent_act(agent, obs, legal)
        MOVE_HISTORY.append(int(action))
        VIEW_CURSOR += 1
        return state_dict({"played": _cell(action), "played_pass": action == ENV.pass_action})


def do_preview() -> dict:
    with LOCK:
        obs, _l, done, _i = _rebuild(VIEW_CURSOR)
        if done:
            return {"error": "game over"}
        cp = ENV.current_player
        agent = _hint_agent(cp)
        if agent is None:
            return {"error": "no model/agent available for a hint"}
        legal = list(ENV.get_legal_actions(cp))
        action = _agent_act(agent, obs, legal)
        if action == ENV.pass_action:
            return {"pass": True, "by": PLAYER_SPEC[cp]["type"]}
        r, c = divmod(int(action), ENV.board_size)
        return {"action": int(action), "row": r, "col": c, "pass": False}


def do_nav(payload: dict) -> dict:
    global VIEW_CURSOR
    with LOCK:
        to = payload.get("to")
        n = len(MOVE_HISTORY)
        if to == "back":
            VIEW_CURSOR = max(0, VIEW_CURSOR - 1)
        elif to == "forward":
            VIEW_CURSOR = min(n, VIEW_CURSOR + 1)
        elif to == "start":
            VIEW_CURSOR = 0
        elif to == "latest":
            VIEW_CURSOR = n
        elif to == "goto":
            VIEW_CURSOR = max(0, min(n, int(payload.get("cursor", VIEW_CURSOR))))
        return state_dict()


def list_options() -> dict:
    models = []
    for pat in ("*.pth", os.path.join("models", "*.pth"), os.path.join("model", "*.pth")):
        for p in glob.glob(os.path.join(HERE, pat)):
            rel = os.path.relpath(p, HERE)
            if rel not in models:
                models.append(rel)
    models.sort()
    rules = []
    for name in ("random", "greedy", "heuristic", "minimax"):
        try:
            _load_rule_agent(name)
            rules.append(name)
        except Exception:
            pass
    return {"board_size": BOARD_SIZE, "models": models, "rules": rules,
            "preset_model": PRESET_MODEL}


# ── HTTP layer ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            fp = os.path.join(HERE, "index.html")
            if not os.path.exists(fp):
                self.send_error(404, "index.html not found next to server.py")
                return
            with open(fp, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/state":
            with LOCK:
                self._send_json(state_dict())
            return
        if path == "/api/options":
            self._send_json(list_options())
            return
        self.send_error(404)

    def do_POST(self):
        path = self.path.split("?")[0]
        payload = self._read_json()
        try:
            if path == "/api/config":
                self._send_json(apply_config(payload)); return
            if path == "/api/move":
                self._send_json(do_move(int(payload["action"]))); return
            if path == "/api/agent_step":
                self._send_json(do_agent_step()); return
            if path == "/api/preview":
                self._send_json(do_preview()); return
            if path == "/api/nav":
                self._send_json(do_nav(payload)); return
            if path == "/api/reset":
                with LOCK:
                    self._send_json(apply_config({
                        "board_size": BOARD_SIZE, "epsilon": EPSILON,
                        "players": {"1": PLAYER_SPEC[1], "-1": PLAYER_SPEC[-1]},
                    }))
                return
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": str(e)}, code=500)
            return
        self.send_error(404)


def initial_config(model_path: str | None, board_size: int):
    """Sensible default: Human (X) vs best-available agent (O)."""
    opp = {"type": "rule", "name": "greedy"}
    if model_path:
        opp = {"type": "model", "path": model_path}
    else:
        opts = list_options()
        if opts["models"]:
            opp = {"type": "model", "path": opts["models"][0]}
        elif "greedy" not in opts["rules"] and opts["rules"]:
            opp = {"type": "rule", "name": opts["rules"][0]}
        elif not opts["rules"]:
            opp = {"type": "human"}
    apply_config({"board_size": board_size, "epsilon": 0.0,
                  "players": {"1": {"type": "human"}, "-1": opp}})


def main():
    global PRESET_MODEL, BOARD_SIZE
    ap = argparse.ArgumentParser(description="Othello web UI server")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--board_size", type=int, default=6)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    BOARD_SIZE = args.board_size
    PRESET_MODEL = args.model
    initial_config(args.model, args.board_size)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Othello UI running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
