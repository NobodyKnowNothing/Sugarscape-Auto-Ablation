#!/usr/bin/env python3
"""
Sugarscape Structural Ablation — Mini-SWE-Agent Edit Harness for Gemma

This module wraps Gemma calls in an autonomous edit harness inspired by and utilizing
mini-SWE-agent (https://github.com/swe-agent/mini-swe-agent).

Instead of requiring Gemma to output the entire 500+ line `strategy.py` file on every
ablation attempt, this harness enables surgical, pinpoint edits:
  1. Single-line replacements: `replace_line <file> <line_num> <new_content>`
  2. Block replacements:       `replace_block <file> <start> <end> <content>`
  3. Search & replace:         `str_replace <file> <old_str> <new_str>`
  4. Line insertions:          `insert <file> <after_line> <content>`
  5. Line deletions:           `delete <file> <start> <end>`

Two execution modes are supported:
  - 'agent': Full multi-turn interactive mini-swe-agent loop. Gemma explores the file,
    inspects line numbers, makes edits via CLI tools, tests syntax/metrics, and submits.
  - 'batch': Fast single-turn surgical edit. Gemma outputs only the pinpoint replacement
    block or diff (reducing token output from ~600 lines to ~5-15 lines).

Directly integrates with `autoresearch.py` via `generate_ablation_variants_harness()`.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Workspace & Configuration Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("ABLATION_MODEL", "gemma-4-31b-it")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "gemma-4-26b-a4b-it")
WORKSPACE_DIR = Path(__file__).parent.resolve()
STRATEGY_FILE = WORKSPACE_DIR / "strategy.py"
BASELINE_METRICS_FILE = WORKSPACE_DIR / "baseline_metrics.json"
RESULTS_FILE = WORKSPACE_DIR / "results.tsv"
PROGRAM_FILE = WORKSPACE_DIR / "program.md"


def is_resource_exhausted_error(exc: Exception) -> bool:
    """Check if exception represents an exhausted quota or rate-limit error (HTTP 429 / RESOURCE_EXHAUSTED)."""
    if exc is None:
        return False

    code = getattr(exc, "code", None)
    if code == 429:
        return True

    status = getattr(exc, "status", None)
    if status and ("RESOURCE_EXHAUSTED" in str(status).upper() or "429" in str(status)):
        return True

    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        d_code = details.get("code") or details.get("error", {}).get("code")
        d_status = details.get("status") or details.get("error", {}).get("status")
        if d_code == 429 or (d_status and "RESOURCE_EXHAUSTED" in str(d_status).upper()):
            return True

    msg = str(exc).upper()
    exhaustion_keywords = [
        "RESOURCE_EXHAUSTED",
        "429",
        "QUOTA EXCEEDED",
        "RATE LIMIT",
        "RESOURCE HAS BEEN EXHAUSTED",
        "TOO MANY REQUESTS",
        "EXHAUSTED",
    ]
    return any(keyword in msg for keyword in exhaustion_keywords)


class ModelFailoverManager:
    """
    Manages primary model calls with automatic failover to a fallback model
    ONLY when request quotas or rate limits are exhausted.
    """

    def __init__(
        self,
        primary_model: str = DEFAULT_MODEL,
        fallback_model: str = FALLBACK_MODEL,
        cooldown_seconds: float = 60.0,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.cooldown_seconds = cooldown_seconds
        self.primary_exhausted_until: float = 0.0
        self.last_model_used: str = primary_model

    def is_primary_exhausted(self) -> bool:
        return time.time() < self.primary_exhausted_until

    def mark_primary_exhausted(self, cooldown: Optional[float] = None) -> None:
        duration = cooldown if cooldown is not None else self.cooldown_seconds
        self.primary_exhausted_until = time.time() + duration

    def clear_primary_exhaustion(self) -> None:
        self.primary_exhausted_until = 0.0

    def generate_content(
        self,
        client: Any,
        contents: Any,
        config: Optional[Any] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        override_model: Optional[str] = None,
    ) -> Tuple[Any, str]:
        """
        Execute client.models.generate_content with strict failover on resource exhaustion.
        Returns: (response, model_name_used)
        """
        _log = log_fn or (lambda m: print(f"[ModelFailover] {m}", file=sys.stderr))
        primary = override_model or self.primary_model

        if override_model and override_model != self.primary_model:
            resp = client.models.generate_content(model=override_model, contents=contents, config=config)
            self.last_model_used = override_model
            return resp, override_model

        if self.is_primary_exhausted():
            remaining = max(1, int(self.primary_exhausted_until - time.time()))
            _log(
                f"Primary model '{primary}' is in quota cooldown ({remaining}s remaining). "
                f"Routing to fallback model '{self.fallback_model}'."
            )
            try:
                resp = client.models.generate_content(
                    model=self.fallback_model,
                    contents=contents,
                    config=config,
                )
                self.last_model_used = self.fallback_model
                return resp, self.fallback_model
            except Exception as e:
                raise e

        try:
            resp = client.models.generate_content(
                model=primary,
                contents=contents,
                config=config,
            )
            if self.primary_exhausted_until > 0:
                _log(f"Primary model '{primary}' quota recovered! Resumed as active default model.")
                self.clear_primary_exhaustion()
            self.last_model_used = primary
            return resp, primary
        except Exception as exc:
            if not is_resource_exhausted_error(exc):
                # ONLY switch when requests/quota are exhausted
                raise exc

            self.mark_primary_exhausted()
            _log(
                f"⚠️ Requests for primary model '{primary}' EXHAUSTED "
                f"(HTTP 429 / RESOURCE_EXHAUSTED). "
                f"Switching to fallback model '{self.fallback_model}'..."
            )
            resp = client.models.generate_content(
                model=self.fallback_model,
                contents=contents,
                config=config,
            )
            self.last_model_used = self.fallback_model
            return resp, self.fallback_model


GLOBAL_MODEL_FAILOVER = ModelFailoverManager()


def generate_content_with_failover(
    client: Any,
    contents: Any,
    config: Optional[Any] = None,
    primary_model: str = DEFAULT_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[Any, str]:
    """Helper function to run generate_content with automatic failover only on exhaustion."""
    global GLOBAL_MODEL_FAILOVER
    if primary_model != GLOBAL_MODEL_FAILOVER.primary_model or fallback_model != GLOBAL_MODEL_FAILOVER.fallback_model:
        GLOBAL_MODEL_FAILOVER = ModelFailoverManager(
            primary_model=primary_model,
            fallback_model=fallback_model,
        )
    return GLOBAL_MODEL_FAILOVER.generate_content(
        client=client,
        contents=contents,
        config=config,
        log_fn=log_fn,
    )



# ===========================================================================
# 1. Core Code Editor (Single-line, Block, Search/Replace, Line Insertion)
# ===========================================================================

class CodeEditor:
    """
    Robust file editor supporting surgical line and block modifications.
    Maintains an undo history stack for safe rollbacks.
    """

    def __init__(self, target_file: Path | str):
        self.target_file = Path(target_file)
        self.history: List[str] = []
        if self.target_file.exists():
            self._initial_content = self.target_file.read_text(encoding="utf-8")
        else:
            self._initial_content = ""

    def _snapshot(self):
        """Save current content to history stack."""
        if self.target_file.exists():
            self.history.append(self.target_file.read_text(encoding="utf-8"))

    def undo(self) -> str:
        """Revert to previous state in history stack."""
        if not self.history:
            return "No previous state to revert to."
        prev = self.history.pop()
        self.target_file.write_text(prev, encoding="utf-8")
        return f"Reverted {self.target_file.name} to previous state ({len(prev.splitlines())} lines)."

    def reset_to_initial(self) -> str:
        """Reset file to initial session content."""
        self.target_file.write_text(self._initial_content, encoding="utf-8")
        return f"Reset {self.target_file.name} to original state."

    def view(self, start: int = 1, end: Optional[int] = None) -> str:
        """
        Display file contents with 1-based line numbers.
        """
        if not self.target_file.exists():
            return f"Error: File {self.target_file} does not exist."

        lines = self.target_file.read_text(encoding="utf-8").splitlines()
        total = len(lines)
        if total == 0:
            return f"{self.target_file.name} is empty."

        start = max(1, start)
        end = min(total, end) if end is not None else total

        if start > total:
            return f"Start line {start} exceeds total lines ({total})."

        output = [f"--- {self.target_file.name} (lines {start} to {end} of {total}) ---"]
        for idx in range(start - 1, end):
            output.append(f"{idx + 1:4d} | {lines[idx]}")
        return "\n".join(output)

    def replace_line(self, line_num: int, new_line: str) -> str:
        """
        Replace a single line (1-indexed) in the target file.
        """
        if not self.target_file.exists():
            return f"Error: File {self.target_file} not found."

        self._snapshot()
        lines = self.target_file.read_text(encoding="utf-8").splitlines(keepends=True)
        total = len(lines)

        if line_num < 1 or line_num > total:
            return f"Error: line_num {line_num} out of bounds (1 to {total})."

        old_line = lines[line_num - 1].rstrip("\r\n")
        new_line_clean = new_line.rstrip("\r\n") + "\n"
        lines[line_num - 1] = new_line_clean

        self.target_file.write_text("".join(lines), encoding="utf-8")
        return (
            f"Successfully replaced line {line_num} in {self.target_file.name}:\n"
            f"- {line_num:4d} | {old_line}\n"
            f"+ {line_num:4d} | {new_line_clean.rstrip()}"
        )

    def replace_block(self, start_line: int, end_line: int, new_content: str) -> str:
        """
        Replace lines start_line through end_line (inclusive, 1-indexed) with new_content.
        new_content can have any number of lines (including empty to delete).
        """
        if not self.target_file.exists():
            return f"Error: File {self.target_file} not found."

        self._snapshot()
        lines = self.target_file.read_text(encoding="utf-8").splitlines(keepends=True)
        total = len(lines)

        if start_line < 1:
            return f"Error: start_line {start_line} must be >= 1."
        if end_line > total:
            return f"Error: end_line {end_line} exceeds total lines ({total})."
        if start_line > end_line:
            return f"Error: start_line {start_line} > end_line {end_line}."

        if new_content and not new_content.endswith("\n"):
            new_content += "\n"

        replacement_lines = [new_content] if new_content else []
        old_lines = lines[start_line - 1 : end_line]
        lines[start_line - 1 : end_line] = replacement_lines

        self.target_file.write_text("".join(lines), encoding="utf-8")

        return (
            f"Successfully replaced lines {start_line}-{end_line} ({len(old_lines)} lines removed, "
            f"{len(new_content.splitlines())} lines added) in {self.target_file.name}."
        )

    def insert_lines(self, after_line: int, content: str) -> str:
        """
        Insert new lines after after_line (0 to insert at very beginning of file).
        """
        if not self.target_file.exists():
            return f"Error: File {self.target_file} not found."

        self._snapshot()
        lines = self.target_file.read_text(encoding="utf-8").splitlines(keepends=True)
        total = len(lines)

        if after_line < 0 or after_line > total:
            return f"Error: after_line {after_line} out of range (0 to {total})."

        if content and not content.endswith("\n"):
            content += "\n"

        lines.insert(after_line, content)
        self.target_file.write_text("".join(lines), encoding="utf-8")
        return f"Successfully inserted {len(content.splitlines())} lines after line {after_line}."

    def delete_lines(self, start_line: int, end_line: int) -> str:
        """
        Delete lines start_line through end_line (inclusive, 1-indexed).
        """
        return self.replace_block(start_line, end_line, "")

    def str_replace(self, old_str: str, new_str: str, allow_multiple: bool = False) -> str:
        """
        Exact string replacement. Requires unique match unless allow_multiple=True.
        """
        if not self.target_file.exists():
            return f"Error: File {self.target_file} not found."

        self._snapshot()
        content = self.target_file.read_text(encoding="utf-8")

        count = content.count(old_str)
        if count == 0:
            return f"Error: Target string not found in {self.target_file.name}."
        if count > 1 and not allow_multiple:
            return (
                f"Error: Target string occurs {count} times in {self.target_file.name}. "
                f"Please provide more surrounding context to ensure uniqueness, or pass allow_multiple."
            )

        updated = content.replace(old_str, new_str) if allow_multiple else content.replace(old_str, new_str, 1)
        self.target_file.write_text(updated, encoding="utf-8")
        return f"Successfully replaced {count if allow_multiple else 1} occurrence(s) in {self.target_file.name}."

    def check_syntax(self) -> Tuple[bool, str]:
        """Verify Python AST syntax validity."""
        if not self.target_file.exists():
            return False, f"File {self.target_file} does not exist."
        try:
            code = self.target_file.read_text(encoding="utf-8")
            ast.parse(code)
            return True, "Syntax OK: Code parses cleanly as Python AST."
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}\n  {e.text}"

    def get_diff(self) -> str:
        """Return unified diff comparing current file to initial state."""
        current = self.target_file.read_text(encoding="utf-8").splitlines(keepends=True)
        initial = self._initial_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            initial, current,
            fromfile=f"a/{self.target_file.name}",
            tofile=f"b/{self.target_file.name}"
        )
        return "".join(diff) or "No changes detected."


# ===========================================================================
# 2. Evaluation & Complexity Helpers
# ===========================================================================

def compute_complexity_info(code: str) -> dict:
    """Calculate AST nodes, cyclomatic complexity, lines, and combined score."""
    from complexity import combined_complexity_score
    return combined_complexity_score(code)


def quick_evaluate(strategy_path: Path | str, steps: int = 50) -> dict:
    """
    Fast simulation test to verify strategy.py compiles and runs without crashing.
    Uses Stage 1 of the validation waterfall for exact pipeline alignment.
    """
    path = Path(strategy_path)
    if not path.exists():
        return {"success": False, "error": f"File {path} does not exist"}

    code = path.read_text(encoding="utf-8")
    try:
        from validation import stage1_fast_filter
        vr = stage1_fast_filter(code, steps=steps)
        if vr.stage1_ok:
            return {
                "success": True,
                "steps_run": steps,
                "message": f"Stage 1 Fast Filter passed ({steps} steps in {vr.stage1_time:.2f}s).",
            }
        else:
            return {
                "success": False,
                "error": vr.reject_reason or "Stage 1 Fast Filter rejected variant.",
            }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


# ===========================================================================
# 3. Model Adapter: Google GenAI (Native Gemma) conforming to Mini-SWE-Agent
# ===========================================================================

class GemmaGenAIModel:
    """
    Model adapter connecting Google GenAI SDK (default: gemma-4-31b-it with failover
    to gemma-4-26b-a4b-it on request exhaustion) to the mini-swe-agent Model protocol.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        fallback_model: str = FALLBACK_MODEL,
        client: Any = None,
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.client = client
        if self.client is None:
            key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
            if key:
                try:
                    from google import genai
                    self.client = genai.Client(api_key=key)
                except ImportError:
                    self.client = None

        self.failover_manager = ModelFailoverManager(
            primary_model=self.model_name,
            fallback_model=self.fallback_model,
        )
        self.action_regex = r"```(?:mswea_bash_command|bash)?\s*\n(.*?)\n```"
        self.cost = 0.0
        self.n_calls = 0

    def query(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Query Gemma model (with quota-exhaustion failover) and parse actions."""
        if not self.client:
            raise RuntimeError(
                "google-genai client not available. Ensure 'google-genai' is installed "
                "and GOOGLE_API_KEY is set or client is provided."
            )

        from google.genai import types

        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"[{role}]\n{content}\n")

        full_prompt = "\n".join(prompt_parts)

        self.n_calls += 1
        response, used_model = self.failover_manager.generate_content(
            client=self.client,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=kwargs.get("temperature", 0.6),
                max_output_tokens=kwargs.get("max_output_tokens", 16384),
            ),
        )
        self.model_name = used_model

        text = extract_response_text(response)
        actions = self._parse_actions(text)

        return {
            "role": "assistant",
            "content": text,
            "extra": {
                "actions": actions,
                "cost": 0.0,
                "model_used": used_model,
                "timestamp": time.time(),
            },
        }

    def _parse_actions(self, content: str) -> List[Dict[str, str]]:
        matches = re.findall(self.action_regex, content, re.DOTALL)
        if not matches:
            for line in content.splitlines():
                line_str = line.strip()
                if line_str.startswith("python edit_harness.py") or line_str.startswith("./edit_harness.py"):
                    return [{"command": line_str}]
            return []
        return [{"command": matches[0].strip()}]

    def format_message(self, **kwargs) -> Dict[str, Any]:
        return kwargs

    def format_observation_messages(
        self, message: Dict[str, Any], outputs: List[Dict[str, Any]], template_vars: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        results = []
        for out in outputs:
            text_out = out.get("output", "")
            ret_code = out.get("returncode", 0)
            content = f"<returncode>{ret_code}</returncode>\n<output>\n{text_out}\n</output>"
            results.append({
                "role": "user",
                "content": content,
                "extra": {"raw_output": text_out, "returncode": ret_code}
            })
        return results

    def get_template_vars(self, **kwargs) -> Dict[str, Any]:
        return {"model_name": self.model_name, "n_calls": self.n_calls}

    def serialize(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "n_calls": self.n_calls}


# ===========================================================================
# 4. Mini-SWE-Agent Environment & Runner (Mode = 'agent')
# ===========================================================================

SYSTEM_TEMPLATE = """\
You are an expert AI software researcher performing structural ablation on the Sugarscape model.
Your objective is to simplify `strategy.py` to reduce AST node count and cyclomatic branching while preserving emergent behaviors.

IMPORTANT: DO NOT output the entire `strategy.py` file!
Instead, use surgical single-line or block editing commands. You interact with the environment via bash commands.

### Available Editing CLI Tools:
1. `python edit_harness.py view <file> <start_line> <end_line>`
   View targeted lines with 1-based line numbers.
2. `python edit_harness.py replace_line <file> <line_num> "<new_line>"`
   Replace a single line.
3. `python edit_harness.py replace_block <file> <start_line> <end_line> << 'EOF'
<new_code>
EOF`
   Replace a range of lines with new code (or empty to delete).
4. `python edit_harness.py str_replace <file> << 'EOF'
<<<<<<< SEARCH
old exact code
=======
new code
>>>>>>> REPLACE
EOF`
   Exact search and replace.
5. `python edit_harness.py check <file>`
   Check Python syntax and show complexity delta (AST nodes, cyclomatic score).
6. `python edit_harness.py eval <file>`
   Run a fast simulation sanity check to ensure the model still runs.
7. `python edit_harness.py diff <file>`
   Show unified git diff of changes made so far.
8. `python edit_harness.py submit "<description>"`
   Finalize and submit your ablation with a 1-line description.

### Rules of Engagement:
- Each turn must include a **THOUGHT** section explaining what you're doing.
- Each turn must include **EXACTLY ONE** bash command in a ```mswea_bash_command code block.
- Always run `python edit_harness.py check strategy.py` after editing to ensure no syntax errors.
- When satisfied, finish by calling: `python edit_harness.py submit "Brief description of ablation"`.
"""

INSTANCE_TEMPLATE = """\
## Task: Perform Structural Ablation Round on strategy.py

Current Baseline & Constraints:
- Baseline Combined Complexity: {{ current_complexity.combined_score | round(1) if current_complexity else 'N/A' }}
  (AST Nodes: {{ current_complexity.ast_nodes if current_complexity else 'N/A' }}, Cyclomatic: {{ current_complexity.cyclomatic_total if current_complexity else 'N/A' }}, Lines: {{ current_complexity.lines if current_complexity else 'N/A' }})
- Target: Lower the combined complexity score while maintaining valid Python syntax and emergent properties.

Ablation History / Lessons Learned:
```
{{ results_history }}
```

Target File:
`strategy.py` is in the current working directory.

Recommended Workflow:
1. Inspect the function or class you want to simplify:
   `python edit_harness.py view strategy.py <start> <end>`
2. Execute a single-line or block replacement:
   `python edit_harness.py replace_block strategy.py <start> <end> << 'EOF'`
3. Verify syntax and complexity:
   `python edit_harness.py check strategy.py`
4. Run quick sanity test:
   `python edit_harness.py eval strategy.py`
5. Submit final simplification:
   `python edit_harness.py submit "Replaced verbose helper with simplified logic"`
"""


class HarnessEnvironment:
    """Local execution environment executing commands in a subshell."""

    def __init__(self, cwd: Path | str, timeout: int = 60):
        self.cwd = Path(cwd).resolve()
        self.timeout = timeout
        self.submission_text: Optional[str] = None
        self.is_finished: bool = False

    def execute(self, action: Dict[str, Any], cwd: str = "") -> Dict[str, Any]:
        command = action.get("command", "").strip()
        exec_cwd = cwd or str(self.cwd)

        # Intercept direct submit command
        if "submit" in command and ("edit_harness.py submit" in command or command.startswith("submit")):
            match = re.search(r'submit\s+["\']?(.*?)["\']?$', command)
            desc = match.group(1) if match else "Ablation applied"
            self.submission_text = desc
            self.is_finished = True
            return {
                "output": f"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n{desc}",
                "returncode": 0,
                "exception_info": "",
            }

        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=exec_cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=os.environ.copy()
            )
            stdout = (res.stdout + ("\n" + res.stderr if res.stderr else "")).strip()

            if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in stdout:
                self.is_finished = True
                lines = stdout.split("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", 1)[1].strip()
                self.submission_text = lines or "Ablation submitted"

            return {"output": stdout, "returncode": res.returncode, "exception_info": ""}
        except subprocess.TimeoutExpired:
            return {"output": f"Command timed out after {self.timeout}s", "returncode": -1, "exception_info": "Timeout"}
        except Exception as e:
            return {"output": f"Execution error: {e}", "returncode": -1, "exception_info": str(e)}

    def get_template_vars(self, **kwargs) -> Dict[str, Any]:
        return {"cwd": str(self.cwd)}

    def serialize(self) -> Dict[str, Any]:
        return {"cwd": str(self.cwd), "timeout": self.timeout}


class MiniSweAgentHarness:
    """
    Mini-SWE-Agent runner executing interactive ablation sessions.
    Uses installed `minisweagent` classes if available, otherwise native fallback.
    """

    def __init__(
        self,
        workspace_dir: Path | str,
        model_name: str = DEFAULT_MODEL,
        fallback_model: str = FALLBACK_MODEL,
        client: Any = None,
        step_limit: int = 15,
        temperature: float = 0.6,
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.step_limit = step_limit
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.client = client

        self.messages: List[Dict[str, Any]] = []
        self.env = HarnessEnvironment(cwd=self.workspace_dir)
        self.model = GemmaGenAIModel(
            model_name=self.model_name,
            fallback_model=self.fallback_model,
            client=self.client,
        )

    def run(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent interactive editing loop."""
        from jinja2 import Template

        sys_msg = Template(SYSTEM_TEMPLATE).render()
        inst_msg = Template(INSTANCE_TEMPLATE).render(**task_context)

        self.messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": inst_msg},
        ]

        steps_taken = 0
        for step_idx in range(self.step_limit):
            steps_taken += 1
            try:
                model_msg = self.model.query(self.messages, temperature=self.temperature)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Model query failed: {e}",
                    "steps": steps_taken,
                }

            self.messages.append(model_msg)
            actions = model_msg.get("extra", {}).get("actions", [])

            if not actions:
                obs_msg = {
                    "role": "user",
                    "content": (
                        "Format Error: No command block found. Please provide your next command "
                        "in triple backticks: ```mswea_bash_command\n<command>\n```"
                    )
                }
                self.messages.append(obs_msg)
                continue

            outputs = [self.env.execute(action) for action in actions]
            obs_msgs = self.model.format_observation_messages(model_msg, outputs)
            self.messages.extend(obs_msgs)

            if self.env.is_finished:
                return {
                    "success": True,
                    "submission": self.env.submission_text or "Ablation submitted",
                    "steps": steps_taken,
                }

        return {
            "success": False,
            "submission": "Reached step limit without submission.",
            "steps": steps_taken,
        }


# ===========================================================================
# 5. Surgical Batch Mode (Mode = 'batch')
# ===========================================================================

def generate_surgical_batch_variants(
    client: Any,
    current_strategy: str,
    results_history: str,
    baseline_metrics: dict,
    complexity: dict,
    program_text: str = "",
    n_variants: int = 1,
    model_name: str = DEFAULT_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    temperature: float = 0.7,
) -> List[Tuple[str, str]]:
    """
    Fast single-turn surgical ablation generator.

    Prompts Gemma to output ONLY the targeted line or block replacement rather than
    regenerating the entire 500+ line strategy file.
    """
    from google.genai import types

    lines = current_strategy.splitlines()
    total_lines = len(lines)

    numbered_code = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))

    prompt = f"""\
{program_text}

---

## Current strategy.py ({total_lines} lines, numbered):
```python
{numbered_code}
```

## Current Complexity (Combined AST + Cyclomatic Score):
- Combined Score: {complexity.get('combined_score', 0):.1f} (LOWER IS BETTER)
- AST Nodes: {complexity.get('ast_nodes', 0)}
- Cyclomatic: {complexity.get('cyclomatic_total', 0)}
- Lines: {complexity.get('lines', 0)}

## Target Baseline Metrics to Preserve:
```json
{json.dumps(baseline_metrics, indent=2)}
```

## Recent Results History:
```
{results_history}
```

---

### Instructions for Surgical Ablation:
Propose {n_variants} distinct structural simplifications of `strategy.py`.
CRITICAL: DO NOT output the full file! Output ONLY surgical replacements.

For EACH variant, choose ONE of the following formats:

Format A (Replace a range of lines):
**Variant <N>:** <one-line description of what was simplified>
```replace_block
START_LINE: <start line number>
END_LINE: <end line number>
```python
<new code block to replace lines START_LINE through END_LINE>
```

Format B (Replace a single line):
**Variant <N>:** <one-line description of what was simplified>
```replace_line
LINE_NUM: <line number>
```python
<new single line content>
```

Format C (Search & Replace exact text):
**Variant <N>:** <one-line description of what was simplified>
```str_replace
<<<<<<< SEARCH
<exact existing code block>
=======
<simplified replacement code block>
>>>>>>> REPLACE
```

Focus on eliminating redundant calculations, inlining unnecessary helper functions,
flattening nested branches, and simplifying agent decision rules.
"""

    try:
        response, used_model = generate_content_with_failover(
            client=client,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=16384,
            ),
            primary_model=model_name,
            fallback_model=fallback_model,
        )
        text = extract_response_text(response)
    except Exception as e:
        print(f"[edit_harness] Batch generation API error: {e}", file=sys.stderr)
        return []

    return parse_surgical_edits(text, current_strategy, expected=n_variants)


def extract_response_text(response: Any) -> str:
    """Extract clean assistant text, handling thinking blocks and multi-part responses."""
    if not response:
        return ""
    if hasattr(response, "text") and response.text:
        return response.text
    if hasattr(response, "candidates") and response.candidates:
        content = getattr(response.candidates[0], "content", None)
        parts = getattr(content, "parts", []) if content else []
        for p in reversed(parts):
            if not getattr(p, "thought", False) and getattr(p, "text", None):
                return p.text
        for p in parts:
            if getattr(p, "text", None):
                return p.text
    return ""


def parse_surgical_edits(
    response_text: str,
    original_code: str,
    expected: int = 1
) -> List[Tuple[str, str]]:
    """Parse surgical edit blocks from model response and apply them to original_code."""
    variants = []

    pattern = r'\*\*Variant\s+\d+[:\s]*(.*?)\*\*(.*?)(?=(?:\*\*Variant|\Z))'
    matches = re.findall(pattern, response_text, re.DOTALL)
    variant_chunks: List[Tuple[str, str]] = []

    if matches:
        for header, body in matches:
            desc = header.strip().strip("*").strip("-").strip()
            if not desc:
                first_lines = [l.strip().strip("*").strip("-").strip() for l in body.strip().splitlines()]
                first_lines = [l for l in first_lines if l and not l.startswith("```")]
                desc = first_lines[0] if first_lines else "Surgical simplification"
            variant_chunks.append((desc, body))
    else:
        variant_chunks.append(("Surgical simplification", response_text))

    for desc, chunk in variant_chunks[:expected]:
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write(original_code)
            tmp_path = Path(tmp.name)

        try:
            editor = CodeEditor(tmp_path)
            applied = False

            # Pattern 1: replace_block (flexible regarding backticks)
            block_match = re.search(
                r'```replace_block\s*\n(?:[^\n]*\n)?START_LINE:\s*(\d+)\s*\nEND_LINE:\s*(\d+)\s*\n(?:\`\`\`(?:python)?\s*\n)?(.*?)\`\`\`',
                chunk, re.DOTALL
            )
            if block_match:
                s_line = int(block_match.group(1))
                e_line = int(block_match.group(2))
                new_content = block_match.group(3)
                res = editor.replace_block(s_line, e_line, new_content)
                if not res.startswith("Error"):
                    applied = True

            # Pattern 2: replace_line (flexible regarding backticks)
            if not applied:
                line_match = re.search(
                    r'```replace_line\s*\n(?:[^\n]*\n)?LINE_NUM:\s*(\d+)\s*\n(?:\`\`\`(?:python)?\s*\n)?(.*?)\`\`\`',
                    chunk, re.DOTALL
                )
                if line_match:
                    l_num = int(line_match.group(1))
                    new_line = line_match.group(2).strip("\r\n")
                    res = editor.replace_line(l_num, new_line)
                    if not res.startswith("Error"):
                        applied = True

            # Pattern 3: str_replace
            if not applied:
                str_match = re.search(
                    r'<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE',
                    chunk, re.DOTALL
                )
                if str_match:
                    search_str = str_match.group(1)
                    replace_str = str_match.group(2)
                    res = editor.str_replace(search_str, replace_str)
                    if not res.startswith("Error"):
                        applied = True

            if applied:
                modified_code = tmp_path.read_text(encoding="utf-8")
                try:
                    ast.parse(modified_code)
                    if modified_code != original_code:
                        if "def create_model" in original_code and "def create_model" not in modified_code:
                            pass
                        else:
                            variants.append((modified_code, desc[:120]))
                except SyntaxError:
                    pass

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return variants


# ===========================================================================
# 6. Unified Interface for Autoresearch Integration
# ===========================================================================

def generate_ablation_variants_harness(
    client: Any = None,
    current_strategy: str = "",
    results_history: str = "",
    baseline_metrics: Optional[dict] = None,
    complexity: Optional[dict] = None,
    program_text: str = "",
    n_variants: int = 1,
    mode: str = "batch",
    model_name: str = DEFAULT_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    temperature: float = 0.7,
    step_limit: int = 12,
    **kwargs,
) -> List[Tuple[str, str]]:
    """
    Drop-in replacement for `generate_ablation_variants()` in `autoresearch.py`.

    Modes:
      - 'batch': Fast, single-turn surgical ablation (recommended for speed & efficiency).
      - 'agent': Multi-turn interactive mini-swe-agent loop with environment bash tools.
    """
    baseline_metrics = baseline_metrics or {}
    complexity = complexity or compute_complexity_info(current_strategy)

    if mode == "agent":
        variants = []
        for var_idx in range(n_variants):
            with tempfile.TemporaryDirectory(prefix=f"ablation_var_{var_idx}_") as tmpdir:
                tmppath = Path(tmpdir)
                sandbox_strategy = tmppath / "strategy.py"
                sandbox_strategy.write_text(current_strategy, encoding="utf-8")

                # Copy edit_harness.py into sandbox so agent can invoke subcommands
                shutil.copy2(__file__, tmppath / "edit_harness.py")

                task_context = {
                    "current_complexity": complexity,
                    "baseline_metrics": baseline_metrics,
                    "results_history": results_history,
                    "variant_index": var_idx + 1,
                }

                harness = MiniSweAgentHarness(
                    workspace_dir=tmppath,
                    model_name=model_name,
                    fallback_model=fallback_model,
                    client=client,
                    step_limit=step_limit,
                    temperature=temperature,
                )

                res = harness.run(task_context)
                if res.get("success"):
                    modified_code = sandbox_strategy.read_text(encoding="utf-8")
                    try:
                        ast.parse(modified_code)
                        if modified_code != current_strategy and "def create_model" in modified_code:
                            desc = res.get("submission", f"Variant {var_idx + 1}")
                            variants.append((modified_code, desc[:120]))
                    except SyntaxError:
                        pass
        return variants

    else:
        # Default: batch surgical mode
        return generate_surgical_batch_variants(
            client=client,
            current_strategy=current_strategy,
            results_history=results_history,
            baseline_metrics=baseline_metrics,
            complexity=complexity,
            program_text=program_text,
            n_variants=n_variants,
            model_name=model_name,
            fallback_model=fallback_model,
            temperature=temperature,
        )


# ===========================================================================
# 7. Command Line Interface (CLI Tools)
# ===========================================================================

def parse_str_replace_block(text: str) -> Tuple[Optional[str], Optional[str]]:
    pattern = r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1), match.group(2)
    return None, None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sugarscape Mini-SWE-Agent Edit Harness (Line & Block File Editor)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Editing subcommands")

    # view
    p_view = subparsers.add_parser("view", help="View lines with line numbers")
    p_view.add_argument("file", help="File to view")
    p_view.add_argument("start", type=int, nargs="?", default=1, help="Start line")
    p_view.add_argument("end", type=int, nargs="?", default=None, help="End line")

    # replace_line
    p_rep_line = subparsers.add_parser("replace_line", help="Replace a single line")
    p_rep_line.add_argument("file", help="Target file")
    p_rep_line.add_argument("line_num", type=int, help="1-indexed line number")
    p_rep_line.add_argument("content", help="New line content")

    # replace_block
    p_rep_block = subparsers.add_parser("replace_block", help="Replace a range of lines")
    p_rep_block.add_argument("file", help="Target file")
    p_rep_block.add_argument("start", type=int, help="Start line (1-indexed)")
    p_rep_block.add_argument("end", type=int, help="End line (inclusive)")
    p_rep_block.add_argument("content", nargs="?", default=None, help="Replacement content (or stdin if omitted)")

    # str_replace
    p_str = subparsers.add_parser("str_replace", help="Search and replace code block")
    p_str.add_argument("file", help="Target file")
    p_str.add_argument("search", nargs="?", default=None, help="Search string (or stdin block if omitted)")
    p_str.add_argument("replace", nargs="?", default=None, help="Replacement string")

    # insert
    p_ins = subparsers.add_parser("insert", help="Insert lines after line number")
    p_ins.add_argument("file", help="Target file")
    p_ins.add_argument("after_line", type=int, help="Line number to insert after (0 for top)")
    p_ins.add_argument("content", nargs="?", default=None, help="Content to insert")

    # delete
    p_del = subparsers.add_parser("delete", help="Delete a line range")
    p_del.add_argument("file", help="Target file")
    p_del.add_argument("start", type=int, help="Start line")
    p_del.add_argument("end", type=int, help="End line")

    # check
    p_chk = subparsers.add_parser("check", help="Check syntax and report complexity")
    p_chk.add_argument("file", help="Target file")

    # diff
    p_diff = subparsers.add_parser("diff", help="Show unified diff against start of session")
    p_diff.add_argument("file", help="Target file")

    # eval
    p_eval = subparsers.add_parser("eval", help="Run quick simulation sanity check")
    p_eval.add_argument("file", nargs="?", default="strategy.py", help="Strategy file to evaluate")
    p_eval.add_argument("--steps", type=int, default=50, help="Steps to run (default 50)")

    # undo
    p_undo = subparsers.add_parser("undo", help="Undo last modification")
    p_undo.add_argument("file", help="Target file")

    # submit
    p_sub = subparsers.add_parser("submit", help="Finalize ablation and submit")
    p_sub.add_argument("description", nargs="+", help="One-line summary of what was simplified")

    # run (standalone harness execution)
    p_run = subparsers.add_parser("run", help="Run ablation round via harness")
    p_run.add_argument("--model", default=DEFAULT_MODEL, help="Model name (default: gemma-4-31b-it)")
    p_run.add_argument("--fallback-model", default=FALLBACK_MODEL, help="Fallback model when quota exhausted (default: gemma-4-26b-a4b-it)")
    p_run.add_argument("--mode", default="batch", choices=["batch", "agent"], help="Harness mode")
    p_run.add_argument("--steps", type=int, default=12, help="Max steps for agent")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "view":
        editor = CodeEditor(args.file)
        print(editor.view(args.start, args.end))

    elif args.command == "replace_line":
        editor = CodeEditor(args.file)
        print(editor.replace_line(args.line_num, args.content))

    elif args.command == "replace_block":
        editor = CodeEditor(args.file)
        content = args.content
        if content is None:
            content = sys.stdin.read()
        print(editor.replace_block(args.start, args.end, content))

    elif args.command == "str_replace":
        editor = CodeEditor(args.file)
        search_str = args.search
        replace_str = args.replace

        if search_str is None and replace_str is None:
            stdin_data = sys.stdin.read()
            s, r = parse_str_replace_block(stdin_data)
            if s is not None and r is not None:
                search_str, replace_str = s, r
            else:
                print("Error: Input does not match <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE format.")
                sys.exit(1)

        print(editor.str_replace(search_str, replace_str or ""))

    elif args.command == "insert":
        editor = CodeEditor(args.file)
        content = args.content or sys.stdin.read()
        print(editor.insert_lines(args.after_line, content))

    elif args.command == "delete":
        editor = CodeEditor(args.file)
        print(editor.delete_lines(args.start, args.end))

    elif args.command == "check":
        editor = CodeEditor(args.file)
        ok, msg = editor.check_syntax()
        if not ok:
            print(f"❌ {msg}")
            sys.exit(1)
        print(f"✅ {msg}")
        code = Path(args.file).read_text(encoding="utf-8")
        info = compute_complexity_info(code)
        gz_str = ""
        if "gzip_compression_ratio" in info:
            gz_str = f" | Gzip: {info['gzip_compression_ratio']:.1%} ({info['gzip_compressed_bytes']}B)"
        print(
            f"   Complexity: Combined Score = {info['combined_score']:.1f} | "
            f"AST Nodes: {info['ast_nodes']} | Cyclomatic: {info['cyclomatic_total']}"
            f"{gz_str} | Lines: {info['lines']}"
        )

    elif args.command == "diff":
        editor = CodeEditor(args.file)
        print(editor.get_diff())

    elif args.command == "eval":
        res = quick_evaluate(args.file, steps=args.steps)
        if res.get("success"):
            print(f"✅ Simulation sanity check passed ({res.get('steps_run')} steps).")
            if res.get("final_gini") is not None:
                print(f"   Final Gini: {res['final_gini']:.4f}")
        else:
            print(f"❌ Simulation failed: {res.get('error')}")
            sys.exit(1)

    elif args.command == "undo":
        editor = CodeEditor(args.file)
        print(editor.undo())

    elif args.command == "submit":
        desc = " ".join(args.description)
        print("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
        print(desc)

    elif args.command == "run":
        if not STRATEGY_FILE.exists():
            print(f"Error: {STRATEGY_FILE} not found.")
            sys.exit(1)

        print(f"Starting ablation round (mode={args.mode}) with model: {args.model}")
        current_code = STRATEGY_FILE.read_text(encoding="utf-8")
        complexity = compute_complexity_info(current_code)

        results_hist = "(No previous history)"
        if RESULTS_FILE.exists():
            results_hist = "\n".join(RESULTS_FILE.read_text(encoding="utf-8").strip().splitlines()[-20:])

        baseline_metrics = {}
        if BASELINE_METRICS_FILE.exists():
            baseline_metrics = json.loads(BASELINE_METRICS_FILE.read_text(encoding="utf-8")).get("mean_metrics", {})

        program_txt = PROGRAM_FILE.read_text(encoding="utf-8") if PROGRAM_FILE.exists() else ""

        variants = generate_ablation_variants_harness(
            client=None,
            current_strategy=current_code,
            results_history=results_hist,
            baseline_metrics=baseline_metrics,
            complexity=complexity,
            program_text=program_txt,
            n_variants=1,
            mode=args.mode,
            model_name=args.model,
            fallback_model=args.fallback_model,
            step_limit=args.steps,
        )

        if variants:
            code, desc = variants[0]
            new_comp = compute_complexity_info(code)
            delta = complexity["combined_score"] - new_comp["combined_score"]
            print(f"\n🎉 Successfully generated variant: {desc}")
            print(f"   Score: {complexity['combined_score']:.1f} -> {new_comp['combined_score']:.1f} ({delta:+.1f})")
            gz_delta = ""
            if "gzip_compressed_bytes" in complexity and "gzip_compressed_bytes" in new_comp:
                gz_delta = f" | Gzip: {complexity['gzip_compressed_bytes']}B -> {new_comp['gzip_compressed_bytes']}B"
            print(f"   Lines: {complexity['lines']} -> {new_comp['lines']}{gz_delta}")
        else:
            print("\n❌ No passing variant generated this round.")


if __name__ == "__main__":
    main()
