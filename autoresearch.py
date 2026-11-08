#!/usr/bin/env python3
"""
Sugarscape Minimal-Viable-Model Research Loop

An LLM (Gemma) proposes mechanism-level changes to strategy.py. Each candidate
is scored by the judge's canonical compressed complexity (including all executed
Mesa code) and accepted only if its EMERGENT FEATURES are statistically
equivalent to canonical Sugarscape over the pre-registered seeds and
input domain (judge/): the same features, and the same response of every
feature to every input, at a fresh random sample of the domain in both trade
worlds. Accepted models enter a Pareto archive; periodic holdout audits roll
back unconfirmed changes.

Multi-Trajectory Compression & Plateau Resets:
  The system continually compresses complexity C along a trajectory until a
  scientifically grounded plateau criterion is met (frontier stagnation across
  move-portfolio cycles, diminishing marginal compression rate < 0.2%, or
  mechanism-ledger saturation). Upon plateauing, the trajectory champion is
  certified and archived (`archive/<profile>/trajectories.json` and
  `archive/<profile>/champions/`), a clear `PLATEAU_RESET` separator is logged
  in `results.tsv`, and `strategy.py` resets to `canonical_strategy.py` (the
  original flattened Mesa Sugarscape) to explore a fresh ablation path.
  Every attempt's behavioral and structural vectors are logged and clustered
  (`cluster_analysis.py` -> `reports/<profile>/`) to identify cross-trajectory
  failure attractors and load-bearing mechanisms.

Architecture:
  - judge/             (FIXED)   contract, features, equivalence tests, complexity, design
  - calibrate_judge.py (FIXED)   reference ensemble, margins, control calibration
  - canonical_strategy.py (FIXED) original flattened Mesa Sugarscape for resets
  - strategy.py        (MUTABLE) current trajectory's best model
  - program.md         (HUMAN)   research agenda for the LLM
  - mechanisms.json    (LEDGER)  rule statuses, evidence, and cross-trajectory history
  - cluster_analysis.py          vector extraction, clustering & attractor reports
  - certify.py                   knockout certificate for the final model

Usage:
    python calibrate_judge.py --profile full
    python autoresearch.py --profile full
"""

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from cluster_analysis import analyze_clusters, extract_attempt_vector
from judge import judge_hash
from judge import complexity as cx
from judge.contract import code_hash
from judge.design import load_profile
from judge.parallel import shutdown as shutdown_pool
from judge.pipeline import (audit, calibration_status, evaluate_candidate, identity_digests,
                            load_reference)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = os.environ.get("ABLATION_MODEL", "gemma-4-31b-it")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "gemma-4-26b-a4b-it")
VARIANTS_PER_GENERATION = int(os.environ.get("VARIANTS_PER_GENERATION", 4))
DEFAULT_PLATEAU_GENERATIONS = int(os.environ.get("PLATEAU_GENERATIONS", 6))
DEFAULT_PLATEAU_ATTEMPTS = int(os.environ.get("PLATEAU_ATTEMPTS", 20))

ROOT = Path(__file__).resolve().parent
STRATEGY_FILE = ROOT / "strategy.py"
CANONICAL_FILE = ROOT / "canonical_strategy.py"
AGENT_REPO = Path(__file__).parent / "agent_repo"
AGENT_STRATEGY = AGENT_REPO / "strategy.py"
RESULTS_FILE = Path(__file__).parent / "results.tsv"
AGENT_RESULTS = AGENT_REPO / "results.tsv"
PROGRAM_FILE = ROOT / "program.md"
LEDGER_FILE = ROOT / "mechanisms.json"
AGENT_LEDGER = AGENT_REPO / "mechanisms.json"
ARCHIVE_ROOT = ROOT / "archive"
REPORTS_ROOT = ROOT / "reports"

# Edit harness configuration ('batch' for fast single-turn surgical edits, 'agent' for interactive multi-turn)
HARNESS_MODE = os.environ.get("HARNESS_MODE", "batch")

API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{ts}] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[{ts}] {msg.encode('ascii', 'replace').decode()}", flush=True)


# ---------------------------------------------------------------------------
# Git & Remote push helpers (operate on agent_repo)
# ---------------------------------------------------------------------------
def get_github_token() -> str:
    """Retrieve GitHub token from environment (.env), Colab Secrets, or Windows Credential Manager."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_PAT", "")
    if token:
        return token.strip().strip("'\"")

    # Check Google Colab Secrets (google.colab.userdata)
    try:
        from google.colab import userdata
        colab_token = userdata.get("GITHUB_TOKEN") or userdata.get("GH_TOKEN")
        if colab_token:
            return colab_token.strip().strip("'\"")
    except Exception:
        pass

    # Fallback: Query Windows Credential Manager via git-credential-wincred
    wincred_candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Git/mingw64/libexec/git-core/git-credential-wincred.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Git/mingw64/libexec/git-core/git-credential-wincred.exe",
    ]
    for wincred in wincred_candidates:
        if wincred.exists():
            try:
                proc = subprocess.Popen(
                    [str(wincred), "get"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                out, _ = proc.communicate(input="protocol=https\nhost=github.com\n\n", timeout=3)
                for line in out.splitlines():
                    if line.startswith("password="):
                        return line.split("=", 1)[1].strip()
            except Exception:
                pass
    return ""


def get_remote_repo_url(token: str | None = None) -> str:
    """Determine the remote GitHub URL, optionally embedding authentication token."""
    remote = os.environ.get("GITHUB_REMOTE_URL") or os.environ.get("AUTORESEARCH_REMOTE_URL", "")
    if not remote:
        try:
            res = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True, text=True, cwd=str(Path(__file__).parent),
                stdin=subprocess.DEVNULL
            )
            remote = res.stdout.strip()
        except Exception:
            remote = ""

    if not remote:
        remote = "https://github.com/NobodyKnowNothing/Sugarscape-Auto-Ablation.git"

    clean_remote = re.sub(r'https://[^@]+@', 'https://', remote)
    if not clean_remote.endswith(".git"):
        clean_remote += ".git"

    if token:
        return clean_remote.replace("https://", f"https://{token}@")
    return clean_remote


def git(cmd: str, cwd: str | None = None, timeout: int = 30) -> str:
    """Run a git command in agent_repo (or specified cwd) in non-interactive mode."""
    try:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(
            f"git {cmd}", shell=True, capture_output=True, text=True,
            cwd=cwd or str(AGENT_REPO), stdin=subprocess.DEVNULL,
            env=env, timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        log(f"⚠️  Git command 'git {cmd}' timed out after {timeout}s")
        return ""
    except Exception as e:
        log(f"⚠️  Git command 'git {cmd}' failed: {e}")
        return ""


def merge_results_tsv(file_a: Path, file_b: Path):
    """
    Ensure both file_a and file_b contain a unified superset of results rows,
    preserving row order and avoiding duplicate entries.
    """
    if not file_a.exists() and not file_b.exists():
        return
    if not file_a.exists():
        file_a.write_text(file_b.read_text(encoding="utf-8"), encoding="utf-8")
        return
    if not file_b.exists():
        file_b.write_text(file_a.read_text(encoding="utf-8"), encoding="utf-8")
        return

    lines_a = [l.strip() for l in file_a.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines_b = [l.strip() for l in file_b.read_text(encoding="utf-8").splitlines() if l.strip()]

    if not lines_a and not lines_b:
        return

    seen = set(lines_a)
    combined = list(lines_a)
    for line in lines_b[1:]:
        if line not in seen:
            seen.add(line)
            combined.append(line)

    content = "\n".join(combined) + "\n"
    if content != file_a.read_text(encoding="utf-8"):
        file_a.write_text(content, encoding="utf-8")
    if content != file_b.read_text(encoding="utf-8"):
        file_b.write_text(content, encoding="utf-8")


def sync_logs_to_agent_repo():
    """Ensure results.tsv, mechanisms.json, archive/, reports/, and .gitignore are mirrored into agent_repo."""
    if not AGENT_REPO.exists():
        return
    try:
        if RESULTS_FILE.exists() or AGENT_RESULTS.exists():
            merge_results_tsv(RESULTS_FILE, AGENT_RESULTS)
        if LEDGER_FILE.exists():
            shutil.copy2(LEDGER_FILE, AGENT_LEDGER)
        for folder in ("archive", "reports"):
            src_dir = ROOT / folder
            dst_dir = AGENT_REPO / folder
            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        parent_gitignore = Path(__file__).parent / ".gitignore"
        if parent_gitignore.exists():
            shutil.copy2(parent_gitignore, AGENT_REPO / ".gitignore")
    except Exception as e:
        log(f"⚠️  Failed to sync logs to agent_repo: {e}")


def git_commit(message: str):
    sync_logs_to_agent_repo()
    git("add strategy.py")
    if AGENT_RESULTS.exists():
        git("add results.tsv")
    if AGENT_LEDGER.exists():
        git("add mechanisms.json")
    if (AGENT_REPO / "archive").exists():
        git("add archive")
    if (AGENT_REPO / "reports").exists():
        git("add reports")
    git(f"commit -m {shlex.quote(message)}")


def git_revert_to(commit_hash: str):
    # Revert strategy.py to commit_hash without discarding test logs in results.tsv
    git(f"checkout {commit_hash} -- strategy.py")
    if AGENT_STRATEGY.exists():
        shutil.copy2(AGENT_STRATEGY, STRATEGY_FILE)


def git_current_hash() -> str:
    return git("rev-parse --short HEAD")


def git_current_branch() -> str:
    branch = git("rev-parse --abbrev-ref HEAD")
    return branch if branch and "fatal:" not in branch else f"ablation/{datetime.now(timezone.utc).strftime('%b%d').lower()}"


def git_create_branch(name: str):
    existing = git("branch --list " + name)
    if existing.strip():
        git(f"checkout {name}")
    else:
        git(f"checkout -b {name}")


def git_push(branch: str | None = None, remote: str = "origin") -> bool:
    """Autonomously push changes from agent_repo to the remote GitHub repository."""
    token = get_github_token()
    if not token:
        log("   ⚠️  Git push skipped: No GITHUB_TOKEN configured.")
        log("       To enable push in Colab, add 'GITHUB_TOKEN' to Secrets (🔑 icon)")
        log("       or run: os.environ['GITHUB_TOKEN'] = 'your_token_here'")
        return False

    target_branch = branch or git_current_branch()
    log(f"   Pushing {target_branch} to GitHub...")
    try:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(
            f"git push -u {remote} {target_branch}",
            shell=True, capture_output=True, text=True,
            cwd=str(AGENT_REPO), stdin=subprocess.DEVNULL,
            env=env, timeout=45
        )
        if result.returncode == 0:
            log(f"   🚀 Successfully pushed {target_branch} to GitHub")
            return True
        else:
            err = result.stderr.strip() or result.stdout.strip()
            cleaned_err = re.sub(r'https://[^@]+@', 'https://***@', err)
            log(f"   ⚠️  Git push returned code {result.returncode}: {cleaned_err}")
            return False
    except subprocess.TimeoutExpired:
        log("   ⚠️  Git push timed out after 45s (network delay)")
        return False
    except Exception as e:
        log(f"   ⚠️  Git push error: {e}")
        return False


def init_agent_repo():
    """Initialize agent_repo with remote origin and target ablation branch."""
    token = get_github_token()
    remote_url = get_remote_repo_url(token)
    safe_remote_url = get_remote_repo_url(token=None)
    branch = os.environ.get("ABLATION_BRANCH") or f"ablation/{datetime.now(timezone.utc).strftime('%b%d').lower()}"
    parent_repo = str(Path(__file__).parent)

    AGENT_REPO.mkdir(exist_ok=True)

    if not token:
        log("⚠️  NOTICE: No GITHUB_TOKEN detected!")
        log("   Subrepo changes will be committed locally, but pushing to GitHub is disabled.")
        log("   In Google Colab, add 'GITHUB_TOKEN' to Secrets (🔑 icon on left sidebar)")
        log("   or set: os.environ['GITHUB_TOKEN'] = 'gho_...' before running.")

    is_new = not (AGENT_REPO / ".git").exists()
    if is_new:
        log(f"Initializing autonomous agent subrepo in {AGENT_REPO.name}...")
        subprocess.run(["git", "init"], cwd=str(AGENT_REPO), capture_output=True, stdin=subprocess.DEVNULL)

        git('config credential.helper ""')
        if not git("config user.name"):
            git('config user.name "Autoresearch Agent"')
        if not git("config user.email"):
            git('config user.email "agent@autoresearch.local"')

        git(f'remote add origin "{remote_url}"')

        remote_synced = False
        if token:
            log(f"Checking if remote branch '{branch}' exists on GitHub...")
            fetch_remote = subprocess.run(
                ["git", "fetch", "origin", f"{branch}:{branch}"],
                cwd=str(AGENT_REPO), capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30
            )
            if fetch_remote.returncode == 0:
                log(f"Synced with existing remote branch '{branch}' from GitHub.")
                subprocess.run(
                    ["git", "checkout", branch],
                    cwd=str(AGENT_REPO), capture_output=True, text=True, stdin=subprocess.DEVNULL
                )
                remote_synced = True
                if not AGENT_STRATEGY.exists():
                    shutil.copy2(STRATEGY_FILE, AGENT_STRATEGY)
                if AGENT_RESULTS.exists() or RESULTS_FILE.exists():
                    merge_results_tsv(RESULTS_FILE, AGENT_RESULTS)
                sync_logs_to_agent_repo()

        if not remote_synced:
            log("Aligning commit history with repository main branch...")
            fetch_res = subprocess.run(
                ["git", "fetch", parent_repo, "main"],
                cwd=str(AGENT_REPO), capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30
            )
            if fetch_res.returncode == 0:
                subprocess.run(
                    ["git", "checkout", "-B", branch, "FETCH_HEAD"],
                    cwd=str(AGENT_REPO), capture_output=True, text=True, stdin=subprocess.DEVNULL
                )
                shutil.copy2(STRATEGY_FILE, AGENT_STRATEGY)
                sync_logs_to_agent_repo()
                status = git("status --porcelain")
                if status:
                    git_commit("Baseline Sugarscape strategy and logs for ablation")
            else:
                git_create_branch(branch)
                shutil.copy2(STRATEGY_FILE, AGENT_STRATEGY)
                sync_logs_to_agent_repo()
                git_commit("Initial Sugarscape strategy and logs")
    else:
        git('config credential.helper ""')
        if not git("config user.name"):
            git('config user.name "Autoresearch Agent"')
        if not git("config user.email"):
            git('config user.email "agent@autoresearch.local"')

        existing_remotes = git("remote")
        if "origin" in existing_remotes:
            git(f'remote set-url origin "{remote_url}"')
        else:
            git(f'remote add origin "{remote_url}"')
        if token:
            subprocess.run(
                ["git", "fetch", "origin", f"{branch}:{branch}"],
                cwd=str(AGENT_REPO), capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30
            )
        git_create_branch(branch)
        if token:
            subprocess.run(
                ["git", "merge", "--ff-only", f"origin/{branch}"],
                cwd=str(AGENT_REPO), capture_output=True, text=True, stdin=subprocess.DEVNULL
            )
        if AGENT_STRATEGY.exists():
            shutil.copy2(AGENT_STRATEGY, STRATEGY_FILE)
        elif not AGENT_STRATEGY.exists():
            shutil.copy2(STRATEGY_FILE, AGENT_STRATEGY)
        if AGENT_RESULTS.exists() or RESULTS_FILE.exists():
            merge_results_tsv(RESULTS_FILE, AGENT_RESULTS)
        sync_logs_to_agent_repo()

    log(f"Subrepo ready on branch '{branch}' -> {safe_remote_url}")
    if token:
        git_push(branch)


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------
RESULTS_COLUMNS = [
    "trajectory", "generation", "variant", "parent", "candidate", "status", "stage", "label",
    "move", "mechanism", "action", "C", "delta_C", "eq_distance", "E1", "E2", "E3", "E4", "E5",
    "nearest_control", "ncd_canonical", "seeds_per_point", "seconds", "judge_hash", "reason", "description",
]


def init_results():
    header = "\t".join(RESULTS_COLUMNS) + "\n"
    if RESULTS_FILE.exists():
        lines = [l for l in RESULTS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines and lines[0] != header.rstrip("\n"):
            old_cols = lines[0].split("\t")
            if "generation" in old_cols and len(lines) > 1:
                # Migrate existing rows cleanly by filling trajectory=1 and new columns
                migrated = [header.rstrip("\n")]
                for raw_line in lines[1:]:
                    vals = raw_line.split("\t")
                    row_map = dict(zip(old_cols, vals))
                    row_map.setdefault("trajectory", "1")
                    row_map.setdefault("nearest_control", "")
                    row_map.setdefault("ncd_canonical", "")
                    migrated.append("\t".join(row_map.get(c, "") for c in RESULTS_COLUMNS))
                RESULTS_FILE.write_text("\n".join(migrated) + "\n", encoding="utf-8")
                log(f"Migrated {len(lines) - 1} existing rows in {RESULTS_FILE.name} to include trajectory & vector columns")
            else:
                RESULTS_FILE.write_text(header, encoding="utf-8")
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(header, encoding="utf-8")
    sync_logs_to_agent_repo()


def _clean(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if math.isnan(v) else (f"{v:.3f}" if not math.isinf(v) else "inf")
    return str(v).replace("\t", " ").replace("\n", " ")


def append_result(row: dict):
    try:
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write("\t".join(_clean(row.get(c)) for c in RESULTS_COLUMNS) + "\n")
        sync_logs_to_agent_repo()
    except Exception as e:
        log(f"⚠️ Warning: Failed to append result row to {RESULTS_FILE}: {e}")


# ---------------------------------------------------------------------------
# Archive (Pareto front over complexity x equivalence distance), multi-trajectory
# ---------------------------------------------------------------------------
class Archive:
    def __init__(self, profile_name: str):
        self.profile_name = profile_name
        self.dir = ARCHIVE_ROOT / profile_name
        self.models = self.dir / "models"
        self.champions = self.dir / "champions"
        self.file = self.dir / "archive.json"
        self.attempts_file = self.dir / "attempts.jsonl"
        self.trajectories_file = self.dir / "trajectories.json"
        self.models.mkdir(parents=True, exist_ok=True)
        self.champions.mkdir(parents=True, exist_ok=True)
        if self.file.exists():
            self.data = json.loads(self.file.read_text(encoding="utf-8"))
        else:
            self.data = {
                "entries": [],
                "audits": [],
                "generation": 0,
                "trajectory": 1,
                "trajectory_start_gen": 0,
                " trajectory_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "last_best_improvement_gen": 0,
                "attempts_since_improvement": 0,
                "trajectory_attempts": 0,
            }
        self.data.setdefault("trajectory", 1)
        self.data.setdefault("trajectory_start_gen", 0)
        self.data.setdefault("trajectory_started_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.data.setdefault("last_best_improvement_gen", self.data.get("trajectory_start_gen", 0))
        self.data.setdefault("attempts_since_improvement", 0)
        self.data.setdefault("trajectory_attempts", 0)
        for e in self.data.get("entries", []):
            e.setdefault("trajectory", 1)

    @property
    def current_trajectory(self) -> int:
        return int(self.data.get("trajectory", 1))

    @property
    def entries(self) -> list[dict]:
        return self.data["entries"]

    def save(self):
        self.file.write_text(json.dumps(self.data, indent=1), encoding="utf-8")

    def code(self, entry_id: str) -> str:
        return (self.models / f"{entry_id}.py").read_text(encoding="utf-8")

    def get(self, entry_id: str, trajectory: int | None = None) -> dict:
        if trajectory is not None:
            for e in reversed(self.entries):
                if e["id"] == entry_id and int(e.get("trajectory", 1)) == trajectory:
                    return e
        return next(e for e in reversed(self.entries) if e["id"] == entry_id)

    def add(self, code: str, info: dict) -> dict:
        info.setdefault("trajectory", self.current_trajectory)
        (self.models / f"{info['id']}.py").write_text(code, encoding="utf-8")
        self.entries.append(info)
        self.save()
        return info

    def live(self, trajectory: int | None = None) -> list[dict]:
        target_t = self.current_trajectory if trajectory is None else trajectory
        return [
            e for e in self.entries
            if not e.get("excluded") and int(e.get("trajectory", 1)) == target_t
        ]

    def all_live(self) -> list[dict]:
        return [e for e in self.entries if not e.get("excluded")]

    def front(self, trajectory: int | None = None) -> list[dict]:
        live = self.live(trajectory=trajectory)
        return [e for e in live if not any(
            o is not e and o["C"] <= e["C"] and o["eq_distance"] <= e["eq_distance"]
            and (o["C"] < e["C"] or o["eq_distance"] < e["eq_distance"]) for o in live)]

    def best(self, trajectory: int | None = None) -> dict:
        return min(self.live(trajectory=trajectory), key=lambda e: (e["C"], e["eq_distance"]))

    def origin(self, trajectory: int | None = None) -> dict:
        target_t = self.current_trajectory if trajectory is None else trajectory
        traj_entries = [e for e in self.entries if int(e.get("trajectory", 1)) == target_t]
        return traj_entries[0] if traj_entries else self.entries[0]

    def choose_parent(self, rng) -> dict:
        front = sorted(self.front(), key=lambda e: e["C"])
        if rng.random() < 0.6 or len(front) == 1:
            return front[0]
        return front[int(rng.integers(len(front)))]

    def lineage(self, entry: dict) -> list[dict]:
        target_t = int(entry.get("trajectory", self.current_trajectory))
        chain = [entry]
        visited = {entry["id"]}
        while chain[-1].get("parent"):
            pid = chain[-1]["parent"]
            if pid in visited:
                break
            visited.add(pid)
            chain.append(self.get(pid, trajectory=target_t))
        return chain

    def log_attempt(self, rec: dict):
        rec.setdefault("trajectory", self.current_trajectory)
        rec.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with open(self.attempts_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def trajectory_attempts_list(self, trajectory: int | None = None) -> list[dict]:
        if not self.attempts_file.exists():
            return []
        target_t = self.current_trajectory if trajectory is None else trajectory
        out = []
        for l in self.attempts_file.read_text(encoding="utf-8").strip().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if int(r.get("trajectory", 1)) == target_t and r.get("outcome") != "plateau_reset":
                out.append(r)
        return out

    def recent_attempts(self, n: int = 12) -> list[dict]:
        if not self.attempts_file.exists():
            return []
        lines = [l for l in self.attempts_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
        parsed = [json.loads(l) for l in lines if json.loads(l).get("outcome") != "plateau_reset"]
        return parsed[-n:]

    def load_trajectories(self) -> list[dict]:
        if not self.trajectories_file.exists():
            return []
        return json.loads(self.trajectories_file.read_text(encoding="utf-8"))

    def record_completed_trajectory(self, summary: dict) -> None:
        trajs = self.load_trajectories()
        trajs.append(summary)
        self.trajectories_file.write_text(json.dumps(trajs, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Mechanism ledger
# ---------------------------------------------------------------------------
def load_ledger() -> dict:
    return json.loads(LEDGER_FILE.read_text(encoding="utf-8"))


def update_ledger(ledger: dict, variant: dict, res, generation: int, trajectory: int = 1) -> None:
    mid = (variant.get("mechanism") or "").strip().lower()
    action = (variant.get("action") or "").strip().lower()
    mechs = ledger["mechanisms"]
    if mid not in mechs or res.label != "mechanism":
        return
    status = {"remove": "removed", "substitute": "replaced", "merge": "merged"}.get(action)
    entry = mechs[mid]
    entry.setdefault("evidence", []).append({
        "trajectory": trajectory,
        "generation": generation,
        "candidate": res.code_hash,
        "action": action or "modify",
        "description": variant.get("desc"),
        "hypothesis": variant.get("hypothesis"),
        "judge_hash": res.judge_hash,
        "features": res.features,
        "eq_distance": res.eq_distance,
    })
    if status:
        entry["status"] = status
    LEDGER_FILE.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def reset_ledger_for_new_trajectory(ledger: dict, completed_trajectory: int) -> None:
    """
    Snapshot current trajectory's mechanism statuses into `cross_trajectory_history`
    and reset all statuses back to `"kept"` for the fresh canonical Sugarscape run.
    """
    for mid, entry in ledger.get("mechanisms", {}).items():
        hist = entry.setdefault("cross_trajectory_history", {})
        hist[f"T{completed_trajectory}"] = entry.get("status", "kept")
        entry["status"] = "kept"
    LEDGER_FILE.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Proposers
# ---------------------------------------------------------------------------
class GeminiProposer:
    def __init__(self, model: str, fallback: str, profile_name: str = "full"):
        from google import genai
        self.client = genai.Client(api_key=API_KEY)
        self.model, self.fallback = model, fallback
        self.profile_name = profile_name

    def propose(self, parent_code, ledger, recent, complexity, moves) -> list[dict]:
        from edit_harness import generate_research_variants
        program = PROGRAM_FILE.read_text(encoding="utf-8") if PROGRAM_FILE.exists() else ""
        return generate_research_variants(self.client, parent_code, program, ledger, recent, complexity,
                                          moves, self.model, self.fallback, log_fn=log,
                                          profile_name=self.profile_name)


class ScriptedProposer:
    """Offline proposer for tests: a JSON list of generations, each a list of variants with edits."""

    def __init__(self, script: Path):
        self.gens = json.loads(Path(script).read_text(encoding="utf-8"))
        self.i = 0

    def propose(self, parent_code, ledger, recent, complexity, moves) -> list[dict]:
        if self.i >= len(self.gens):
            return []
        out = []
        for v in self.gens[self.i]:
            code = parent_code
            ok = True
            for old, new in v.get("edits", []):
                if code.count(old) != 1:
                    ok = False
                    break
                code = code.replace(old, new)
            if v.get("full_file"):
                code = (ROOT / v["full_file"]).read_text(encoding="utf-8")
            item = {k: v.get(k, "") for k in ("desc", "move", "mechanism", "action", "hypothesis")}
            out.append({**item, "code": code if ok and code != parent_code else None,
                        **({} if ok else {"parse_error": "scripted edit anchor not found"})})
        self.i += 1
        return out


# ---------------------------------------------------------------------------
# Ratchet loop, Plateau Detection & Automatic Reset
# ---------------------------------------------------------------------------
def seed_archive(archive: Archive, profile, force_canonical: bool = False) -> None:
    if archive.live() and not force_canonical:
        return
    if force_canonical and CANONICAL_FILE.exists():
        code = CANONICAL_FILE.read_text(encoding="utf-8")
        STRATEGY_FILE.write_text(code, encoding="utf-8")
    else:
        code = STRATEGY_FILE.read_text(encoding="utf-8")
    traj = archive.current_trajectory
    gen = archive.data.get("generation", 0)
    log(f"Seeding Trajectory #{traj} archive with strategy.py (measuring complexity and identity digests)...")
    comp = cx.measure(code, profile)
    digests = identity_digests(code, profile)
    archive.data["trajectory_start_gen"] = gen
    archive.data["trajectory_started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    archive.data["last_best_improvement_gen"] = gen
    archive.data["attempts_since_improvement"] = 0
    archive.data["trajectory_attempts"] = 0
    archive.add(code, {
        "id": code_hash(code),
        "trajectory": traj,
        "parent": None,
        "C": comp["C"],
        "eq_distance": 0.0,
        "label": "origin",
        "delta_C": 0,
        "digests": digests,
        "generation": gen,
        "desc": f"canonical Sugarscape origin (trajectory {traj})",
        "audited": True,
    })


def portfolio(n: int, rng) -> list[str]:
    from edit_harness import MOVE_TYPES
    kinds = list(MOVE_TYPES)
    start = int(rng.integers(len(kinds)))
    return [kinds[(start + i) % len(kinds)] for i in range(n)]


def run_generation(gen: int, archive: Archive, proposer, profile, ref, margins, calib, ledger, rng, args) -> int:
    traj = archive.current_trajectory
    parent = archive.choose_parent(rng)
    parent_code = archive.code(parent["id"])
    pcomp = cx.measure(parent_code, profile)
    moves = portfolio(args.variants, rng)
    prev_best_C = archive.best()["C"]
    log(f"\n═══ [Trajectory #{traj}] Generation {gen} ═══  parent {parent['id']} (C={parent['C']}, front size {len(archive.front())})")
    log(f"Moves: {', '.join(moves)}")
    variants = proposer.propose(parent_code, ledger, archive.recent_attempts(), pcomp, moves)
    if not variants:
        log("No variants proposed this generation.")
        return 0
    accepted = 0
    for k, v in enumerate(variants, 1):
        log(f"--- [T{traj}] Variant {k}/{len(variants)} [{v.get('move')}/{v.get('mechanism')}/{v.get('action')}]: {v.get('desc')}")
        row = {
            "trajectory": traj,
            "generation": gen,
            "variant": k,
            "parent": parent["id"],
            "move": v.get("move"),
            "mechanism": v.get("mechanism"),
            "action": v.get("action"),
            "description": v.get("desc"),
        }
        archive.data["trajectory_attempts"] = archive.data.get("trajectory_attempts", 0) + 1
        archive.data["attempts_since_improvement"] = archive.data.get("attempts_since_improvement", 0) + 1

        if not v.get("code"):
            log(f"    unusable proposal: {v.get('parse_error')}")
            append_result({**row, "status": "parse_error", "reason": v.get("parse_error")})
            archive.log_attempt({**row, "outcome": "unusable", "reason": v.get("parse_error"), "desc": v.get("desc")})
            archive.save()
            continue

        res = evaluate_candidate(v["code"], profile, parent={"C": parent["C"], "digests": parent.get("digests")},
                                 ref=ref, margins=margins, log=log)
        status = "accepted" if res.accepted else ("rejected" if res.stage >= 1 else "rejected_static")

        # Extract behavioral and structural vectors (0 ms simulation overhead via SQLite cache)
        vec_info = extract_attempt_vector(v["code"], res, parent_code, profile, ref, margins, calib)
        nearest_ctrl = (vec_info.get("nearest_control") or {}).get("name", "")
        ncd_canon = vec_info.get("ncd_canonical", 0.0)

        log(f"    {'✅ ACCEPTED' if res.accepted else '❌ rejected'} at stage {res.stage}"
            f"{' (' + res.label + ')' if res.label else ''}: {res.reason or 'all features equivalent'}"
            f" | ~{nearest_ctrl} (NCD={ncd_canon:.3f}) [{res.seconds}s]")

        append_result({
            **row,
            "candidate": res.code_hash,
            "status": status,
            "stage": res.stage,
            "label": res.label,
            "C": res.complexity.get("C"),
            "delta_C": res.delta_C,
            "eq_distance": res.eq_distance,
            **{f: res.features.get(f, "") for f in ("E1", "E2", "E3", "E4", "E5")},
            "nearest_control": nearest_ctrl,
            "ncd_canonical": ncd_canon,
            "seeds_per_point": res.seeds_per_point,
            "seconds": res.seconds,
            "judge_hash": res.judge_hash,
            "reason": res.reason,
        })
        archive.log_attempt({
            **row,
            "candidate": res.code_hash,
            "desc": v.get("desc"),
            "outcome": status,
            "stage": res.stage,
            "C": res.complexity.get("C"),
            "delta_C": res.delta_C,
            "eq_distance": res.eq_distance,
            "features": res.features,
            "reason": res.reason,
            "feedback": res.feedback,
            "label": res.label,
            "vector": vec_info,
        })
        archive.save()

        if not res.accepted:
            continue
        accepted += 1
        archive.add(v["code"], {
            "id": res.code_hash,
            "trajectory": traj,
            "parent": parent["id"],
            "C": res.complexity["C"],
            "eq_distance": res.eq_distance,
            "label": res.label,
            "delta_C": res.delta_C,
            "digests": res.identity_digests,
            "generation": gen,
            "desc": v.get("desc"),
            "move": v.get("move"),
            "mechanism": v.get("mechanism"),
            "action": v.get("action"),
            "hypothesis": v.get("hypothesis"),
            "ncd_canonical": ncd_canon,
            "audited": False,
        })
        update_ledger(ledger, v, res, gen, trajectory=traj)
        best = archive.best()
        if best["C"] < prev_best_C:
            prev_best_C = best["C"]
            archive.data["last_best_improvement_gen"] = gen
            archive.data["attempts_since_improvement"] = 0
            archive.save()
        if best["id"] == res.code_hash:
            STRATEGY_FILE.write_text(v["code"], encoding="utf-8")
            log(f"    🎉 new best model in Trajectory #{traj}: C {parent['C']} -> {res.complexity['C']} ({res.label})")
            if not args.no_git:
                AGENT_STRATEGY.write_text(v["code"], encoding="utf-8")
                git_commit(f"[T{traj}] Gen {gen}: {res.label} {v.get('desc', '')[:50]} | C -{res.delta_C:.0f}")
                if not args.no_push:
                    git_push()
    return accepted


def maybe_audit(archive: Archive, profile, ref, margins, args, force: bool = False) -> None:
    pending = [e for e in archive.live() if e.get("audited") is False and e.get("label") == "mechanism"]
    if not pending:
        return
    if not force and len(pending) < profile["audit_every"]:
        return
    best = archive.best()
    log(f"\n═══ Audit [Trajectory #{archive.current_trajectory}]: best model {best['id']} on holdout seeds at {profile['audit_points']} fresh i.i.d. domain points ═══")
    result = audit(archive.code(best["id"]), profile, ref, margins)
    archive.data["audits"].append({
        "trajectory": archive.current_trajectory,
        "model": best["id"],
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{k: result.get(k) for k in ("passed", "features", "reason", "eq_distance", "located_failures", "coverage")},
    })
    if result["passed"]:
        cov = (result.get("coverage") or {}).get("any_statistic_union")
        log("    audit passed: lineage confirmed on holdout seeds over a fresh sample of the input domain"
            + (f" (domain fraction with any gross departure <= {cov:.1%} at 95% confidence)" if cov is not None else ""))
        for e in archive.lineage(best):
            e["audited"] = True
        for e in pending:
            e["audited"] = True
    else:
        log(f"    audit FAILED: {result.get('features') or result.get('reason')} - rolling back unaudited changes")
        for e in pending:
            e["excluded"] = True
            e["audited"] = "failed"
        STRATEGY_FILE.write_text(archive.code(archive.best()["id"]), encoding="utf-8")
    archive.save()


def check_plateau(archive: Archive, ledger: dict, gen: int, args) -> tuple[bool, str]:
    """
    Scientifically grounded complexity-compression plateau detector.

    Returns (is_plateaued, reason) when any of the following criteria holds:
      1. Frontier Stagnation (`frontier_stagnation`):
         At least `args.plateau_generations` consecutive generations AND
         `args.plateau_attempts` evaluated proposals have elapsed since the
         last improvement in trajectory minimum complexity C_min.
         (For default G=6, A=20..24 across the 6 move categories, a move with
         >= 14% success rate would have improved C_min with > 95% probability.)
      2. Diminishing Marginal Compression Rate (`diminishing_returns`):
         At least `args.plateau_generations` generations and `args.plateau_attempts`
         attempts have run in this trajectory, and over the rolling window of the
         last `args.plateau_generations` generations the relative complexity reduction
         (C_window_start - C_min) / C_window_start is < 0.2% with zero accepted
         mechanism-level simplifications in the window.
      3. Mechanism-Ledger Saturation (`mechanism_saturation`):
         At least max(2, plateau_generations // 2) generations have elapsed without
         C_min improvement, and every remaining 'kept' mechanism in `mechanisms.json`
         has been targeted by >= 2 rejected remove/substitute/merge attempts in the
         current trajectory since the last mechanism acceptance.
    """
    g_thresh = max(1, int(args.plateau_generations))
    a_thresh = max(1, int(args.plateau_attempts))

    start_gen = int(archive.data.get("trajectory_start_gen", 0))
    last_imp_gen = int(archive.data.get("last_best_improvement_gen", start_gen))
    gens_run = gen - start_gen
    gens_since_imp = gen - last_imp_gen
    atts_since_imp = int(archive.data.get("attempts_since_improvement", 0))
    traj_atts = int(archive.data.get("trajectory_attempts", 0))

    # Criterion 1: Frontier Stagnation
    if gens_since_imp >= g_thresh and atts_since_imp >= a_thresh:
        p_upper = round(1.0 - (0.05 ** (1.0 / max(atts_since_imp, 1))), 3)
        return (
            True,
            f"frontier_stagnation: {gens_since_imp} generations ({atts_since_imp} attempts) "
            f"without C_min improvement (95% upper bound on per-attempt improvement rate p < {p_upper})",
        )

    # Criterion 2: Diminishing Marginal Compression Rate over rolling window
    if gens_run >= g_thresh and traj_atts >= a_thresh:
        win_start_gen = gen - g_thresh
        live = archive.live()
        earlier = [e["C"] for e in live if int(e.get("generation", 0)) <= win_start_gen]
        c_win_start = min(earlier) if earlier else archive.origin()["C"]
        c_now = archive.best()["C"]
        rel_drop = (c_win_start - c_now) / max(c_win_start, 1)
        mech_accepts_in_win = [
            e for e in live
            if int(e.get("generation", 0)) > win_start_gen and e.get("label") == "mechanism"
        ]
        if rel_drop < 0.002 and not mech_accepts_in_win:
            return (
                True,
                f"diminishing_returns: relative C reduction over last {g_thresh} generations is "
                f"{rel_drop:.4%} (< 0.20% threshold) with 0 mechanism-level acceptances",
            )

    # Criterion 3: Mechanism-Ledger Saturation
    if gens_since_imp >= max(2, g_thresh // 2):
        kept_mechs = [k for k, v in (ledger.get("mechanisms") or {}).items() if v.get("status") == "kept"]
        if kept_mechs:
            live = archive.live()
            mech_entries = [e for e in live if e.get("label") == "mechanism"]
            last_mech_gen = max((int(e.get("generation", 0)) for e in mech_entries), default=start_gen)
            t_atts = archive.trajectory_attempts_list()
            rej_counts = {m: 0 for m in kept_mechs}
            for a in t_atts:
                if int(a.get("generation", 0)) >= last_mech_gen and a.get("outcome") in ("rejected", "rejected_static"):
                    m_target = (a.get("mechanism") or "").strip().lower()
                    act = (a.get("action") or "").strip().lower()
                    if m_target in rej_counts and act in ("remove", "substitute", "merge", ""):
                        rej_counts[m_target] += 1
            if all(cnt >= 2 for cnt in rej_counts.values()):
                return (
                    True,
                    f"mechanism_saturation: all {len(kept_mechs)} remaining kept mechanisms have >= 2 "
                    f"rejected ablation attempts since gen {last_mech_gen} with no C_min drop for {gens_since_imp} gens",
                )

    return False, ""


def handle_plateau_reset(
    gen: int,
    archive: Archive,
    profile,
    ref: dict,
    margins: dict,
    ledger: dict,
    plateau_reason: str,
    args,
) -> None:
    """
    Finalize the plateaued trajectory, certify/audit its champion, archive its artifacts,
    write an explicit separator row to `results.tsv`, and reset `strategy.py` to
    `canonical_strategy.py` to begin the next independent ablation trajectory.
    """
    traj = archive.current_trajectory
    # Audit any pending mechanism changes on the trajectory champion before archiving
    maybe_audit(archive, profile, ref, margins, args, force=True)

    origin = archive.origin(trajectory=traj)
    best = archive.best(trajectory=traj)
    chain = archive.lineage(best)
    by_label = {"refactor": 0.0, "mechanism": 0.0}
    for e in chain:
        if e.get("label") in by_label:
            by_label[e["label"]] += e.get("delta_C", 0) or 0

    mechs_modified = {k: v["status"] for k, v in ledger["mechanisms"].items() if v["status"] != "kept"}
    mechs_kept = [k for k, v in ledger["mechanisms"].items() if v["status"] == "kept"]
    start_gen = int(archive.data.get("trajectory_start_gen", 0))
    gens_run = max(1, gen - start_gen)
    traj_atts = int(archive.data.get("trajectory_attempts", 0))
    total_saved = origin["C"] - best["C"]

    # Save champion code and ledger snapshot in archive/<profile>/champions/
    champ_code = archive.code(best["id"])
    champ_file = archive.champions / f"traj_{traj}_{best['id']}.py"
    champ_file.write_text(champ_code, encoding="utf-8")
    ledger_snap_file = archive.champions / f"traj_{traj}_ledger.json"
    ledger_snap_file.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    traj_summary = {
        "trajectory_id": traj,
        "started_at": archive.data.get("trajectory_started_at", ""),
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start_generation": start_gen + 1,
        "end_generation": gen,
        "generations_run": gens_run,
        "attempts_evaluated": traj_atts,
        "variants_accepted": len(archive.live(trajectory=traj)) - 1,
        "origin_id": origin["id"],
        "origin_C": origin["C"],
        "champion_id": best["id"],
        "champion_C": best["C"],
        "champion_eq_distance": best.get("eq_distance", 0.0),
        "total_bytes_saved": total_saved,
        "compression_ratio": round(best["C"] / max(origin["C"], 1), 4),
        "bytes_saved_refactor": round(by_label["refactor"], 1),
        "bytes_saved_mechanism": round(by_label["mechanism"], 1),
        "mechanisms_modified": mechs_modified,
        "mechanisms_kept": mechs_kept,
        "plateau_reason": plateau_reason,
        "lineage": [
            {
                "id": e["id"],
                "C": e["C"],
                "delta_C": e.get("delta_C", 0),
                "label": e.get("label", ""),
                "move": e.get("move", ""),
                "mechanism": e.get("mechanism", ""),
                "desc": e.get("desc", ""),
            }
            for e in reversed(chain)
        ],
    }
    archive.record_completed_trajectory(traj_summary)

    sep_desc = (
        f"=== TRAJECTORY {traj} PLATEAUED (C: {origin['C']} -> {best['C']}, -{total_saved}B) | "
        f"RESETTING TO CANONICAL SUGARSCAPE FOR TRAJECTORY {traj + 1} ==="
    )
    log(f"\n🔄 {sep_desc}")
    log(f"   Reason: {plateau_reason}")
    rel_champ = champ_file.relative_to(ROOT) if champ_file.is_relative_to(ROOT) else champ_file
    log(f"   Champion saved to: {rel_champ}")

    append_result({
        "trajectory": traj,
        "generation": gen,
        "variant": 0,
        "parent": origin["id"],
        "candidate": best["id"],
        "status": "PLATEAU_RESET",
        "stage": 4,
        "label": "reset",
        "move": "plateau_reset",
        "mechanism": ",".join(mechs_modified.keys()) or "none",
        "action": "reset_to_canonical",
        "C": best["C"],
        "delta_C": total_saved,
        "eq_distance": best.get("eq_distance", 0.0),
        "E1": "pass", "E2": "pass", "E3": "pass", "E4": "pass", "E5": "pass",
        "nearest_control": "canonical_reset",
        "ncd_canonical": best.get("ncd_canonical", 0.0),
        "seeds_per_point": profile["candidate_seeds"],
        "seconds": 0.0,
        "judge_hash": judge_hash(),
        "reason": plateau_reason,
        "description": sep_desc,
    })
    archive.log_attempt({
        "trajectory": traj,
        "generation": gen,
        "variant": 0,
        "parent": origin["id"],
        "candidate": best["id"],
        "outcome": "plateau_reset",
        "stage": 4,
        "C": best["C"],
        "delta_C": total_saved,
        "reason": plateau_reason,
        "desc": sep_desc,
    })

    # Advance to Trajectory T + 1 and reset strategy.py & ledger to canonical Sugarscape
    archive.data["trajectory"] = traj + 1
    reset_ledger_for_new_trajectory(ledger, completed_trajectory=traj)
    seed_archive(archive, profile, force_canonical=True)

    # Refresh cluster & attractor reports across all trajectories
    try:
        analyze_clusters(profile.name)
    except Exception as e:
        log(f"⚠️  Cluster analysis update warning: {e}")

    if not args.no_git:
        shutil.copy2(STRATEGY_FILE, AGENT_STRATEGY)
        git_commit(f"[T{traj}->T{traj+1}] Plateau reset: T{traj} champion {best['id']} (C={best['C']}) -> canonical")
        if not args.no_push:
            git_push()


def summarize(archive: Archive) -> None:
    traj = archive.current_trajectory
    origin = archive.origin(trajectory=traj)
    best = archive.best(trajectory=traj)
    chain = archive.lineage(best)
    by_label = {"refactor": 0.0, "mechanism": 0.0}
    for e in chain:
        if e.get("label") in by_label:
            by_label[e["label"]] += e.get("delta_C", 0) or 0
    trajs_done = archive.load_trajectories()
    log("\n═══ Summary ═══")
    log(f"  Active Trajectory: #{traj} ({len(trajs_done)} completed trajectories in trajectories.json)")
    log(f"  Current Trajectory Archive: {len(archive.live())} live entries (front {len(archive.front())})")
    log(f"  Total Archive Entries Across All Trajectories: {len(archive.entries)}")
    log(f"  Current Trajectory Complexity C: {origin['C']} -> {best['C']} bytes ({origin['C'] - best['C']} saved)")
    log(f"    from code-level refactors:   {by_label['refactor']:.0f} bytes")
    log(f"    from mechanism-level changes: {by_label['mechanism']:.0f} bytes")
    ledger = load_ledger()
    changed = {k: v["status"] for k, v in ledger["mechanisms"].items() if v["status"] != "kept"}
    log(f"  Mechanisms removed/replaced/merged in T{traj}: {changed or 'none'}")


def main():
    global MODEL, FALLBACK_MODEL
    import argparse

    ap = argparse.ArgumentParser(description="Sugarscape minimal-viable-model research loop")
    ap.add_argument("--profile", default=os.environ.get("ABLATION_PROFILE", "full"))
    ap.add_argument("--generations", type=int, default=0, help="number of generations (0 = run until interrupted)")
    ap.add_argument("--variants", type=int, default=VARIANTS_PER_GENERATION)
    ap.add_argument("--plateau-generations", type=int, default=DEFAULT_PLATEAU_GENERATIONS,
                    help="stagnant generations before declaring a complexity plateau and resetting (default: 6)")
    ap.add_argument("--plateau-attempts", type=int, default=DEFAULT_PLATEAU_ATTEMPTS,
                    help="minimum attempts without C_min drop before declaring a plateau (default: 20)")
    ap.add_argument("--max-trajectories", type=int, default=0,
                    help="stop after completing N plateaued trajectories (0 = unlimited)")
    ap.add_argument("--proposer", choices=["gemini", "scripted"], default="gemini")
    ap.add_argument("--script", help="JSON script for --proposer scripted")
    ap.add_argument("--model", "-m", default=MODEL)
    ap.add_argument("--fallback-model", default=FALLBACK_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-git", action="store_true", help="do not use the agent_repo sub-repository")
    ap.add_argument("--no-push", action="store_true", help="commit locally but never push")
    ap.add_argument("--allow-failed-calibration", action="store_true",
                    help="run even if calibration does not meet the control criteria (smoke tests only)")
    args = ap.parse_args()
    MODEL, FALLBACK_MODEL = args.model, args.fallback_model

    profile = load_profile(args.profile)
    ok, msg = calibration_status(profile.name)
    if not ok:
        stale_or_missing = "stale" in msg or "no calibration" in msg
        if stale_or_missing or not args.allow_failed_calibration:
            log(f"ERROR: {msg}")
            sys.exit(2)
        log(f"⚠️  {msg} - continuing because --allow-failed-calibration was given. Results are NOT scientifically valid.")
    ref, margins, calib = load_reference(profile.name)

    log("╔═══════════════════════════════════════════════════════════╗")
    log("║  Sugarscape Minimal-Viable-Model Research Loop            ║")
    log(f"║  Profile:   {profile.name:40s}      ║")
    log(f"║  Proposer:  {args.proposer + ' (' + MODEL + ')':40s}      ║")
    log(f"║  Judge:     {judge_hash():40s}      ║")
    log("╚═══════════════════════════════════════════════════════════╝")

    if args.proposer == "gemini":
        if not API_KEY:
            log("ERROR: No API key. Set GOOGLE_API_KEY or GEMINI_API_KEY env var.")
            sys.exit(1)
        proposer = GeminiProposer(MODEL, FALLBACK_MODEL, profile_name=profile.name)
    else:
        if not args.script:
            log("ERROR: --proposer scripted requires --script")
            sys.exit(1)
        proposer = ScriptedProposer(Path(args.script))

    init_results()
    if not args.no_git:
        init_agent_repo()
    archive = Archive(profile.name)
    seed_archive(archive, profile)
    ledger = load_ledger()
    rng = np.random.default_rng(args.seed + archive.data["generation"])

    try:
        done = 0
        trajectories_completed_this_run = 0
        while args.generations == 0 or done < args.generations:
            archive.data["generation"] += 1
            gen = archive.data["generation"]
            archive.save()
            try:
                run_generation(gen, archive, proposer, profile, ref, margins, calib, ledger, rng, args)
                maybe_audit(archive, profile, ref, margins, args)
                analyze_clusters(profile.name)

                is_plat, plat_reason = check_plateau(archive, ledger, gen, args)
                if is_plat:
                    handle_plateau_reset(gen, archive, profile, ref, margins, ledger, plat_reason, args)
                    trajectories_completed_this_run += 1
                    if args.max_trajectories > 0 and trajectories_completed_this_run >= args.max_trajectories:
                        log(f"Reached --max-trajectories={args.max_trajectories}. Stopping.")
                        break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log(f"Generation {gen} failed: {e}")
                traceback.print_exc()
            done += 1
    except KeyboardInterrupt:
        log("\nInterrupted by user.")
    finally:
        shutdown_pool()
        try:
            analyze_clusters(profile.name)
        except Exception as e:
            log(f"⚠️  Final cluster analysis warning: {e}")
        summarize(archive)
        if not args.no_git:
            sync_logs_to_agent_repo()
            if git("status --porcelain"):
                git_commit("Update experiment logs and cluster reports")
            if not args.no_push:
                git_push()


if __name__ == "__main__":
    main()
