#!/usr/bin/env python3
"""
clustering.py — Semantic clustering of Sugarscape code diffs and ablation attempts.

Key Features:
  1. Code Diff Semantic Clustering:
     Extracts and normalizes unified code diffs (modifications to `strategy.py`),
     embedding them via SentenceTransformer rather than relying solely on commit
     messages or natural-language descriptions.
  2. Flexible Input Pipeline:
     - Direct results TSV (`results.tsv`): Resolves diffs from git history using
       the commit column, or uses an existing diff column.
     - Direct Git Branch / Commit Range (`--branch origin/ablation/sep10` or `--revs`):
       Directly extracts ablation commits and code diffs across branches.
  3. Emergent Pattern Discovery & Analysis:
     - UMAP + HDBSCAN clustering to discover emergent categories of code modification.
     - Code-aware TF-IDF term extraction and medoid diff discovery.
     - Log-odds discriminative analysis identifying code tokens that correlate
       with validation success vs failure.
  4. Rich Interactive Visualisation:
     Plotly dual-panel dashboard showing semantic diff clusters with interactive
     hoverable code previews, outcomes, and log-odds vocabulary.

Dependencies:
    pip install pandas sentence-transformers umap-learn hdbscan plotly scikit-learn

Usage Examples:
    # 1. Cluster results.tsv (auto-resolves code diffs via Git)
    python clustering.py results.tsv

    # 2. Cluster directly from an ablation git branch
    python clustering.py --branch origin/ablation/sep10

    # 3. Custom output and diff format
    python clustering.py results.tsv --diff-format structured --output sep10_clusters.html
"""

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── 1. Git Diff Extraction & Preprocessing ──────────────────────────────────

class DiffProcessor:
    """Processes, cleans, and structures unified code diffs for embedding."""

    @staticmethod
    def clean_diff(raw_diff: str, max_chars: int = 3500) -> str:
        """
        Clean raw git diff for semantic embedding.
        Removes noisy metadata headers (e.g. index hashes, 'diff --git')
        while preserving hunk headers (function/class context) and modified (+/-) lines.
        """
        if not raw_diff or not raw_diff.strip():
            return ""

        cleaned_lines = []
        for line in raw_diff.splitlines():
            # Skip file header noise
            if line.startswith(("diff --git", "index ", "--- ", "+++ ", "new file mode", "deleted file mode")):
                continue
            # Keep hunk headers with context: '@@ -120,6 +120,8 @@ def move(self):' -> '@@ def move(self): @@'
            if line.startswith("@@"):
                hunk_match = re.search(r"@@.*?@@\s*(.*)", line)
                if hunk_match and hunk_match.group(1).strip():
                    cleaned_lines.append(f"@@ {hunk_match.group(1).strip()} @@")
                continue
            # Keep additions, deletions, and context lines
            if line.startswith(("+", "-")) or line.startswith(" "):
                cleaned_lines.append(line)

        result = "\n".join(cleaned_lines).strip()
        if len(result) > max_chars:
            # Keep the beginning of the diff where primary structural changes occur
            result = result[:max_chars] + "\n... [diff truncated for embedding]"
        return result

    @staticmethod
    def structured_diff(raw_diff: str, file_name: str = "strategy.py", max_chars: int = 3500) -> str:
        """
        Formats the diff into a structured representation highlighting:
        - Target scope/functions modified
        - Explicit added code blocks
        - Explicit removed code blocks
        """
        if not raw_diff or not raw_diff.strip():
            return ""

        added: List[str] = []
        removed: List[str] = []
        contexts: List[str] = []

        for line in raw_diff.splitlines():
            if line.startswith("@@"):
                hunk_match = re.search(r"@@.*?@@\s*(.*)", line)
                if hunk_match and hunk_match.group(1).strip():
                    ctx = hunk_match.group(1).strip()
                    if ctx not in contexts:
                        contexts.append(ctx)
            elif line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:].strip())
            elif line.startswith("-") and not line.startswith("---"):
                removed.append(line[1:].strip())

        ctx_str = ", ".join(contexts) if contexts else "module-level"
        parts = [f"File: {file_name}", f"Modified Context: {ctx_str}"]

        if removed:
            parts.append("REMOVED CODE:\n" + "\n".join(removed[:40]))
        if added:
            parts.append("ADDED CODE:\n" + "\n".join(added[:40]))

        full_text = "\n\n".join(parts)
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n... [truncated]"
        return full_text

    @staticmethod
    def format_diff_for_embedding(
        raw_diff: str,
        description: str = "",
        mode: str = "clean",
        file_name: str = "strategy.py",
        max_chars: int = 3500,
    ) -> str:
        """Format diff based on selected strategy."""
        if mode == "structured":
            diff_text = DiffProcessor.structured_diff(raw_diff, file_name=file_name, max_chars=max_chars)
        elif mode == "diff_and_desc":
            clean = DiffProcessor.clean_diff(raw_diff, max_chars=max_chars - 300)
            diff_text = f"Summary: {description.strip()}\n\nDiff:\n{clean}" if description else clean
        else:  # default "clean"
            diff_text = DiffProcessor.clean_diff(raw_diff, max_chars=max_chars)

        return diff_text.strip()


def run_git(args: List[str], repo_dir: Optional[Path] = None) -> Tuple[int, str, str]:
    """Execute a git command and return (returncode, stdout, stderr)."""
    cwd = str(repo_dir) if repo_dir else str(Path.cwd())
    proc = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def get_commit_diff(commit: str, target_file: str = "strategy.py", repo_dir: Optional[Path] = None) -> str:
    """
    Retrieve unified diff of `target_file` for a specific commit against its parent.
    Tries `git diff {commit}~1 {commit} -- {target_file}` first, falling back to `git show`.
    """
    commit = commit.strip()
    if not commit or commit.lower() in ("none", "nan", "null"):
        return ""

    # Primary method: diff parent vs commit
    code, stdout, _ = run_git(["diff", f"{commit}~1", commit, "--", target_file], repo_dir)
    if code == 0 and stdout.strip():
        return stdout

    # Fallback method: git show
    code, stdout, _ = run_git(["show", "--format=", commit, "--", target_file], repo_dir)
    if code == 0 and stdout.strip():
        return stdout

    return ""


def get_commits_from_branch(
    branch: str,
    target_file: str = "strategy.py",
    rev_range: Optional[str] = None,
    repo_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Extract all commits affecting `target_file` on a git branch or revision range."""
    revisions = rev_range or branch
    log_args = ["log", "--reverse", "--format=%H%x09%h%x09%s%x09%an%x09%ad", revisions, "--", target_file]
    code, stdout, stderr = run_git(log_args, repo_dir)
    if code != 0:
        print(f"Warning: git log failed for {revisions}: {stderr.strip()}")
        return []

    entries = []
    lines = [line for line in stdout.strip().splitlines() if "\t" in line]
    print(f"Discovered {len(lines)} commits affecting {target_file} in '{revisions}'")

    for line in lines:
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        full_hash, short_hash, subject, author, date = parts[0], parts[1], parts[2], parts[3], parts[4]

        # Fetch code diff
        raw_diff = get_commit_diff(full_hash, target_file=target_file, repo_dir=repo_dir)
        if not raw_diff.strip():
            continue

        # Extract generation and score reduction from commit message if available
        # e.g., 'Gen 2: Simplify the move method... | score -0.8'
        gen_match = re.search(r"Gen\s+(\d+)", subject, re.IGNORECASE)
        score_match = re.search(r"score\s+([+-]?\d+\.?\d*)", subject, re.IGNORECASE)

        gen_val = int(gen_match.group(1)) if gen_match else None
        score_delta = float(score_match.group(1)) if score_match else None

        entries.append({
            "commit": short_hash,
            "full_hash": full_hash,
            "description": subject,
            "generation": gen_val,
            "score_delta": score_delta,
            "author": author,
            "date": date,
            "raw_diff": raw_diff,
            "success": True,  # Committed ablation variants represent accepted ratchets
            "status": "pass",
        })

    return entries


def resolve_diffs_for_df(
    df: pd.DataFrame,
    target_file: str = "strategy.py",
    repo_dir: Optional[Path] = None,
    diff_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ensure every row in df has a valid code diff.
    Resolves diffs via:
      1. Existing diff column (if present)
      2. Direct commit hash lookup in git
      3. Generation / description matching against child commits in git history
    """
    df = df.copy()

    # 1. Check existing diff column
    if diff_col and diff_col in df.columns:
        df["raw_diff"] = df[diff_col].fillna("").astype(str)
        has_diff = df["raw_diff"].str.len() > 10
        if has_diff.sum() > 0:
            print(f"  Used existing diff column '{diff_col}' for {has_diff.sum()}/{len(df)} rows")
            return df

    # Prepare git history index for fallback matching
    # Autoresearch logs `current_hash` (parent) in results.tsv before committing the passing variant
    code, stdout, _ = run_git(["log", "--all", "--format=%H%x09%h%x09%s"], repo_dir)
    git_commits = []
    if code == 0:
        for line in stdout.strip().splitlines():
            if "\t" in line:
                git_commits.append(line.split("\t"))

    raw_diffs = []
    resolved_count = 0

    for idx, row in df.iterrows():
        commit = str(row.get("commit", "")).strip()
        desc = str(row.get("description", "")).strip()
        gen = row.get("generation")

        diff = ""
        # Try direct commit diff
        if commit and commit.lower() not in ("none", "nan", "null"):
            diff = get_commit_diff(commit, target_file=target_file, repo_dir=repo_dir)

        # If direct diff was empty, attempt child commit resolution
        # (autoresearch parent hash -> committed variant)
        if not diff and git_commits:
            for full_h, short_h, subj in git_commits:
                # Match by generation and keyword in description
                if gen is not None and f"Gen {gen}:" in subj:
                    child_diff = get_commit_diff(full_h, target_file=target_file, repo_dir=repo_dir)
                    if child_diff:
                        diff = child_diff
                        break
                elif desc and len(desc) > 15 and desc[:25].lower() in subj.lower():
                    child_diff = get_commit_diff(full_h, target_file=target_file, repo_dir=repo_dir)
                    if child_diff:
                        diff = child_diff
                        break

        raw_diffs.append(diff)
        if diff:
            resolved_count += 1

    df["raw_diff"] = raw_diffs
    print(f"  Successfully extracted code diffs for {resolved_count}/{len(df)} entries")
    return df


# ─── 2. TSV Loading & Column Detection ───────────────────────────────────────

def load_tsv(
    path: str,
    desc_col: Optional[str] = None,
    success_col: Optional[str] = None,
    commit_col: Optional[str] = None,
    diff_col: Optional[str] = None,
) -> pd.DataFrame:
    """Load results TSV with intelligent column auto-detection."""
    df = pd.read_csv(path, sep="\t")
    print(f"Loaded {len(df)} rows from '{path}'")
    print(f"  Columns: {list(df.columns)}")

    # Flexible description column detection: prioritize exact names over partial keywords
    if not desc_col or desc_col not in df.columns:
        for name in ["description", "desc", "summary", "message", "variant_desc", "simplification"]:
            matches = [c for c in df.columns if c.lower() == name]
            if matches:
                desc_col = matches[0]
                break
        if not desc_col:
            candidates = [
                c for c in df.columns
                if any(k in c.lower() for k in ("desc", "summary", "message", "simplif"))
            ]
            desc_col = candidates[0] if candidates else None

    if desc_col and desc_col in df.columns:
        df["description"] = df[desc_col].fillna("").astype(str).str.strip()
    else:
        df["description"] = ""

    # Flexible success column detection (supports 'status', 'passes', 'pass', 'success', 'result')
    if not success_col or success_col not in df.columns:
        candidates = [
            c for c in df.columns
            if any(k in c.lower() for k in ("status", "pass", "success", "valid", "accept", "result"))
        ]
        success_col = candidates[0] if candidates else None

    if success_col and success_col in df.columns:
        s = df[success_col].astype(str).str.strip().str.lower()
        df["success"] = s.isin({"true", "1", "yes", "pass", "passed", "success"})
        df["status"] = df[success_col].astype(str).str.strip()
    else:
        df["success"] = True
        df["status"] = "pass"

    # Flexible commit column detection
    if not commit_col or commit_col not in df.columns:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ("commit", "sha", "hash", "rev"))]
        commit_col = candidates[0] if candidates else None

    if commit_col and commit_col in df.columns:
        df["commit"] = df[commit_col].astype(str).str.strip()
    else:
        df["commit"] = ""

    # Check for direct diff column if present
    if not diff_col:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ("diff", "patch", "code_diff"))]
        diff_col = candidates[0] if candidates else None

    return df


# ─── 3. Embedding Code Diffs ──────────────────────────────────────────────────

def embed(
    diff_texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """Embed code diff representations using SentenceTransformer."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Missing dependency: pip install sentence-transformers")

    print(f"Embedding {len(diff_texts)} code diffs with {model_name}…")
    model = SentenceTransformer(model_name)
    vecs = model.encode(
        diff_texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=batch_size,
    )
    print(f"  Embedding matrix shape: {vecs.shape}\n")
    return vecs


# ─── 4. UMAP Dimensionality Reduction ────────────────────────────────────────

def reduce_umap(embeddings: np.ndarray, n_neighbors: Optional[int] = None) -> np.ndarray:
    """Project high-dimensional diff embeddings into 2D space via UMAP."""
    n = len(embeddings)
    if n < 3:
        # Trivial projection for tiny sets
        reduced = np.zeros((n, 2))
        for i in range(n):
            reduced[i, 0] = float(i)
        return reduced

    try:
        from umap import UMAP
    except ImportError:
        sys.exit("Missing dependency: pip install umap-learn")

    if n_neighbors is None:
        # Heuristic: square root of n bounded between 2 and 15
        n_neighbors = max(2, min(15, int(math.sqrt(n))))
    if n_neighbors >= n:
        n_neighbors = max(2, n - 1)

    print(f"UMAP → 2D projection (n={n}, n_neighbors={n_neighbors})…")
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )
    reduced = reducer.fit_transform(embeddings)
    print("  UMAP projection completed.\n")
    return reduced


# ─── 5. HDBSCAN Density Clustering ───────────────────────────────────────────

def cluster_hdbscan(
    embeddings_2d: np.ndarray,
    min_cluster_size: Optional[int] = None,
) -> np.ndarray:
    """Cluster 2D UMAP projections using HDBSCAN."""
    n = len(embeddings_2d)
    if n < 4:
        # Small sample size: assign single cluster
        return np.zeros(n, dtype=int)

    try:
        import hdbscan
    except ImportError:
        sys.exit("Missing dependency: pip install hdbscan")

    if min_cluster_size is None:
        min_cluster_size = 2 if n < 30 else max(3, n // 10)
    if min_cluster_size > n:
        min_cluster_size = max(2, n // 2)

    print(f"HDBSCAN clustering (min_cluster_size={min_cluster_size})…")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings_2d)
    unique = set(labels)
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = int((labels == -1).sum())
    print(f"  Discovered {n_clusters} clusters, {n_noise} noise/outlier points\n")
    return labels


# ─── 6. Cluster Labelling (Code-Aware TF-IDF & Medoids) ───────────────────────

def label_clusters(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """
    Generate semantic labels for clusters using code-aware TF-IDF n-grams
    and find the medoid diff closest to the cluster centroid.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

    cluster_ids = sorted(set(cluster_labels[cluster_labels >= 0]))
    if not cluster_ids:
        return {}

    python_keywords = {
        "self", "def", "class", "return", "if", "else", "elif", "for", "in", "while",
        "and", "or", "not", "is", "none", "true", "false", "try", "except", "import",
        "from", "as", "pass", "break", "continue", "with", "lambda", "yield", "int", "float", "str", "len",
    }
    stop_words = list(ENGLISH_STOP_WORDS.union(python_keywords))

    # Code-aware token pattern preserving Python identifiers and snake_case
    code_token_pattern = r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b"
    corpus = df["diff_text"].tolist()

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=code_token_pattern,
        max_features=600,
        stop_words=stop_words,
    )
    try:
        vec.fit(corpus)
        feature_names = vec.get_feature_names_out()
    except ValueError:
        feature_names = np.array([])

    info = {}
    for cid in cluster_ids:
        mask = cluster_labels == cid
        idxs = np.where(mask)[0]
        sub_texts = df.loc[idxs, "diff_text"].tolist()
        successes = df.loc[idxs, "success"]
        success_rate = float(successes.mean())

        # Medoid calculation
        centroid = embeddings[idxs].mean(axis=0)
        dists = np.linalg.norm(embeddings[idxs] - centroid, axis=1)
        medoid_idx = idxs[int(dists.argmin())]
        medoid_desc = df.loc[medoid_idx, "description"] or df.loc[medoid_idx, "commit"]

        # Extract top TF-IDF terms
        top_terms: List[str] = []
        if len(feature_names) > 0:
            tfidf_mat = vec.transform(sub_texts).toarray()
            mean_scores = tfidf_mat.mean(axis=0)
            top_idxs = mean_scores.argsort()[::-1][:4]
            top_terms = [feature_names[i] for i in top_idxs if mean_scores[i] > 0.01]

        term_label = ", ".join(top_terms[:2]) if top_terms else "code-refactor"
        info[cid] = {
            "n": len(idxs),
            "success_rate": success_rate,
            "medoid": medoid_desc,
            "medoid_diff": df.loc[medoid_idx, "diff_text"][:200],
            "terms": top_terms,
            "label": f"C{cid} [{term_label}] ({success_rate:.0%} pass, n={len(idxs)})",
        }

    return info


# ─── 7. Log-Odds Code Discriminative Analysis ─────────────────────────────────

def log_odds_analysis(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Computes Laplace-smoothed log-odds for code tokens appearing in
    passing vs failing code diffs.
      Positive log-odds → code syntax/pattern predicts validation success.
      Negative log-odds → code syntax/pattern predicts validation failure.
    """
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    success_texts = df[df["success"]]["diff_text"].tolist()
    fail_texts = df[~df["success"]]["diff_text"].tolist()

    if len(success_texts) < 2 or len(fail_texts) < 2:
        print("Skipping log-odds: insufficient examples on both pass and fail sides.")
        return pd.DataFrame()

    python_keywords = {
        "self", "def", "class", "return", "if", "else", "elif", "for", "in", "while",
        "and", "or", "not", "is", "none", "true", "false", "try", "except", "import",
        "from", "as", "pass", "break", "continue", "with", "lambda", "yield", "int", "float", "str", "len",
    }
    stop_words = list(ENGLISH_STOP_WORDS.union(python_keywords))

    code_token_pattern = r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b"
    vec = CountVectorizer(
        ngram_range=(1, 2),
        token_pattern=code_token_pattern,
        stop_words=stop_words,
        min_df=2,
        max_df=0.95,
    )
    all_texts = success_texts + fail_texts
    try:
        vec.fit(all_texts)
    except ValueError:
        return pd.DataFrame()

    terms = vec.get_feature_names_out()
    s_counts = vec.transform(success_texts).toarray().sum(axis=0)
    f_counts = vec.transform(fail_texts).toarray().sum(axis=0)
    n_s, n_f = len(success_texts), len(fail_texts)

    # Laplace smoothing
    log_odds = np.log((s_counts + 0.5) / (n_s + 1)) - np.log((f_counts + 0.5) / (n_f + 1))

    result = pd.DataFrame({"term": terms, "log_odds": log_odds})
    pos = result[result["log_odds"] > 0].nlargest(top_n, "log_odds")
    neg = result[result["log_odds"] < 0].nsmallest(top_n, "log_odds")
    return pd.concat([pos, neg]).drop_duplicates().reset_index(drop=True)


# ─── 8. Interactive Plotly Visualisation ──────────────────────────────────────

def visualise(
    df: pd.DataFrame,
    embeddings_2d: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_info: Dict[int, Dict[str, Any]],
    log_odds_df: pd.DataFrame,
    output_path: str,
):
    """Render interactive dual-panel Plotly visualization of code diff clusters."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        sys.exit("Missing dependency: pip install plotly")

    label_map = {cid: info["label"] for cid, info in cluster_info.items()}
    label_map[-1] = "noise / outlier"

    df_plot = df.copy()
    df_plot["x"] = embeddings_2d[:, 0]
    df_plot["y"] = embeddings_2d[:, 1]
    df_plot["cluster_label"] = [label_map.get(l, f"C{l}") for l in cluster_labels]

    # Format diff preview for HTML tooltips
    def make_tooltip_diff(d: str) -> str:
        lines = [line for line in d.splitlines() if line.startswith(("+", "-", "@@"))][:6]
        preview = "<br>".join(lines).replace('"', '&quot;').replace("'", "&#39;")
        return preview or "(empty diff)"

    df_plot["diff_preview"] = df_plot["diff_text"].apply(make_tooltip_diff)

    has_log_odds = not log_odds_df.empty
    fig = make_subplots(
        rows=1,
        cols=2 if has_log_odds else 1,
        subplot_titles=(
            "Semantic Code Diff Clusters  (● pass · ✕ fail)",
            "Discriminative Code Syntax (Log-Odds)",
        ) if has_log_odds else ("Semantic Code Diff Clusters  (● pass · ✕ fail)",),
        column_widths=[0.60, 0.40] if has_log_odds else [1.0],
        horizontal_spacing=0.08,
    )

    # ── Scatter: cluster color, pass/fail symbol ──
    for outcome, symbol, color in [("pass", "circle", "#2ecc71"), ("fail", "x", "#e74c3c")]:
        mask = df_plot["success"] == (outcome == "pass")
        sub = df_plot[mask]
        if sub.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="markers",
                name=f"{outcome} ({len(sub)})",
                marker=dict(
                    size=12 if outcome == "pass" else 10,
                    symbol=symbol,
                    color=color,
                    opacity=0.88,
                    line=dict(width=1, color="rgba(255,255,255,0.7)"),
                ),
                customdata=np.stack([
                    sub["commit"].fillna(""),
                    sub["description"].fillna(""),
                    sub["cluster_label"],
                    sub["diff_preview"],
                ], axis=-1),
                hovertemplate=(
                    "<b>Commit:</b> %{customdata[0]}<br>"
                    "<b>Description:</b> %{customdata[1]}<br>"
                    "<b>Cluster:</b> %{customdata[2]}<br>"
                    "<b>Outcome:</b> " + outcome + "<br><br>"
                    "<b>Diff Snippet:</b><br>%{customdata[3]}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # ── Log-odds bar chart ──
    if has_log_odds:
        sorted_df = log_odds_df.sort_values("log_odds")
        bar_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in sorted_df["log_odds"]]
        fig.add_trace(
            go.Bar(
                x=sorted_df["log_odds"],
                y=sorted_df["term"],
                orientation="h",
                marker_color=bar_colors,
                name="log-odds",
                hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=dict(
            text="Sugarscape Structural Ablation — Semantic Code Diff Analysis",
            font=dict(size=16, family="Inter, sans-serif"),
        ),
        height=650,
        template="plotly_white",
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
        font=dict(family="monospace", size=12),
    )
    fig.update_xaxes(title_text="UMAP Dimension 1", showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text="UMAP Dimension 2", showgrid=False, row=1, col=1)
    if has_log_odds:
        fig.update_xaxes(
            title_text="Log-Odds  (green = predicts success, red = predicts failure)",
            row=1,
            col=2,
        )
        fig.update_yaxes(tickfont=dict(size=10), row=1, col=2)

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Saved interactive visualization → {output_path}")


# ─── 9. Console Summary ──────────────────────────────────────────────────────

def print_summary(cluster_info: Dict[int, Dict[str, Any]], log_odds_df: pd.DataFrame):
    """Print structured terminal summary of discovered clusters."""
    print("\n" + "=" * 65)
    print("CODE DIFF CLUSTER SUMMARY")
    print("=" * 65)
    for cid, info in sorted(cluster_info.items()):
        sr = info["success_rate"]
        bar = "█" * round(sr * 10) + "░" * (10 - round(sr * 10))
        terms_str = ", ".join(info["terms"]) if info["terms"] else "general refactor"
        print(f"\n  Cluster {cid} (n={info['n']}, {sr:.0%} pass) [{bar}]")
        print(f"    Key code tokens: {terms_str}")
        print(f"    Representative: \"{str(info['medoid'])[:80]}\"")

    if not log_odds_df.empty:
        print("\n" + "=" * 65)
        print("DISCRIMINATIVE CODE TOKENS (Top 8 Each Direction)")
        print("=" * 65)
        pos = log_odds_df[log_odds_df["log_odds"] > 0].nlargest(8, "log_odds")
        neg = log_odds_df[log_odds_df["log_odds"] < 0].nsmallest(8, "log_odds")
        if not pos.empty:
            print("\n  ✓ Success-correlated code terms:")
            for _, row in pos.iterrows():
                print(f"      {row['term']:<32} {row['log_odds']:+.3f}")
        if not neg.empty:
            print("\n  ✕ Failure-correlated code terms:")
            for _, row in neg.iterrows():
                print(f"      {row['term']:<32} {row['log_odds']:+.3f}")
    print()


# ─── 10. Main Entrypoint ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Semantic clustering of Sugarscape code diffs using SentenceTransformer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tsv",
        nargs="?",
        default="results.tsv" if Path("results.tsv").exists() else None,
        help="Path to results TSV file (default: results.tsv if present)",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Git branch to extract ablation commits directly from (e.g. origin/ablation/sep10)",
    )
    parser.add_argument(
        "--revs",
        default=None,
        help="Git revision range to extract commits from (e.g. 22fb681..origin/ablation/sep10)",
    )
    parser.add_argument(
        "--target-file",
        default="strategy.py",
        help="Target file whose diffs should be extracted and clustered (default: strategy.py)",
    )
    parser.add_argument(
        "--diff-format",
        choices=["clean", "structured", "diff_and_desc"],
        default="clean",
        help="Diff formatting mode for SentenceTransformer embedding (default: clean)",
    )
    parser.add_argument(
        "--desc-col",
        default=None,
        help="TSV column containing natural-language description (auto-detected if omitted)",
    )
    parser.add_argument(
        "--success-col",
        default=None,
        help="TSV column containing pass/fail validation status (auto-detected if omitted)",
    )
    parser.add_argument(
        "--commit-col",
        default=None,
        help="TSV column containing git commit hash (auto-detected if omitted)",
    )
    parser.add_argument(
        "--diff-col",
        default=None,
        help="TSV column containing direct code diff/patch (auto-detected if omitted)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for embedding code diffs (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="HDBSCAN min_cluster_size (auto-configured from corpus size if omitted)",
    )
    parser.add_argument(
        "--output",
        default="ablation_clusters.html",
        help="Output HTML file for Plotly visualization (default: ablation_clusters.html)",
    )
    parser.add_argument(
        "--top-n-logodds",
        type=int,
        default=20,
        help="Number of terms on each side of the log-odds chart (default: 20)",
    )

    args = parser.parse_args()

    # Load dataset either from Git branch/revs or TSV
    if args.branch or args.revs:
        target_ref = args.branch or "HEAD"
        print(f"Extracting ablation commits and code diffs directly from Git: {args.revs or target_ref}…")
        entries = get_commits_from_branch(
            branch=target_ref,
            target_file=args.target_file,
            rev_range=args.revs,
        )
        if not entries:
            sys.exit(f"No commits affecting {args.target_file} found in {args.revs or target_ref}")
        df = pd.DataFrame(entries)
    elif args.tsv:
        tsv_path = Path(args.tsv)
        if not tsv_path.exists():
            sys.exit(f"Specified TSV file does not exist: {args.tsv}")
        df = load_tsv(
            path=str(tsv_path),
            desc_col=args.desc_col,
            success_col=args.success_col,
            commit_col=args.commit_col,
            diff_col=args.diff_col,
        )
        # Extract code diffs for each row using Git
        df = resolve_diffs_for_df(df, target_file=args.target_file, diff_col=args.diff_col)
    else:
        sys.exit(
            "No input source specified. Provide a TSV file (e.g. python clustering.py results.tsv) "
            "or specify a git branch (e.g. python clustering.py --branch origin/ablation/sep10)."
        )

    # Format diffs for embedding
    df["diff_text"] = [
        DiffProcessor.format_diff_for_embedding(
            raw_diff=row.get("raw_diff", ""),
            description=str(row.get("description", "")),
            mode=args.diff_format,
            file_name=args.target_file,
        )
        for _, row in df.iterrows()
    ]

    # Filter usable diffs
    usable_mask = df["diff_text"].str.len() > 10
    if not usable_mask.any():
        sys.exit("No non-empty code diffs found to cluster.")

    df_usable = df[usable_mask].reset_index(drop=True)
    print(f"Clustering {len(df_usable)} non-empty code diffs (format={args.diff_format})…\n")

    # Embed code diffs with SentenceTransformer
    embeddings = embed(df_usable["diff_text"].tolist(), model_name=args.model)

    # Dimensionality reduction and clustering
    reduced_2d = reduce_umap(embeddings)
    labels = cluster_hdbscan(reduced_2d, min_cluster_size=args.min_cluster_size)
    df_usable["cluster_id"] = labels

    # Label clusters and perform discriminative vocabulary analysis
    cluster_info = label_clusters(df_usable, embeddings, labels)
    log_odds_df = log_odds_analysis(df_usable, top_n=args.top_n_logodds)

    # Console summary and interactive Plotly visualization
    print_summary(cluster_info, log_odds_df)
    visualise(df_usable, reduced_2d, labels, cluster_info, log_odds_df, args.output)

    # Save labelled CSV
    csv_path = Path(args.output).stem + "_labelled.csv"
    export_cols = [c for c in ["commit", "generation", "description", "status", "success", "cluster_id", "diff_text"] if c in df_usable.columns]
    df_usable[export_cols].to_csv(csv_path, index=False)
    print(f"Exported labelled code diff clusters → {csv_path}\n")


if __name__ == "__main__":
    main()