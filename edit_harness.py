#!/usr/bin/env python3
"""
edit_harness.py — Mini-SWE-Agent Style Surgical Edit Harness for Gemma

This module wraps Gemma model calls in a minimalist agentic harness inspired by
mini-swe-agent (https://github.com/swe-agent/mini-swe-agent).

Instead of requiring Gemma to rewrite the entire strategy file (~450+ lines)
for every ablation attempt, this harness enables surgical edits:
  - Single-line replacements (e.g. modifying a parameter or condition)
  - Multi-line block replacements (e.g. simplifying an entire method)
  - Line insertions and deletions
  - Exact search-and-replace blocks (Aider / SWE-agent style)
  - Interactive multi-turn agent loop with syntax error detection (ast.parse)
  - Single-turn batch block applicator for high-throughput variant generation

Can be used as:
  1. A drop-in replacement for autoresearch.py:
       from edit_harness import generate_ablation_variants_harness
  2. An interactive agent loop:
       agent = MiniSWEAgent(model, editor)
       result = agent.run(task_prompt)
  3. A standalone CLI script:
       python edit_harness.py --file strategy.py --task "Simplify trade logic"
  4. A dependency-free self-test:
       python edit_harness.py --self-test
"""

import ast
import difflib
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# In-Memory Code Editor
# ---------------------------------------------------------------------------

class CodeEditor:
    """
    In-memory code editor providing line-based and block-based surgical edits,
    AST syntax validation, unified diff tracking, and undo capability.
    """

    def __init__(self, content: str, filename: str = "strategy.py"):
        self.filename = filename
        self.original_content = content
        self.lines: List[str] = content.splitlines()
        self.undo_stack: List[List[str]] = []
        self.redo_stack: List[List[str]] = []

    def get_content(self) -> str:
        """Return the current file content with standard line endings."""
        return "\n".join(self.lines) + ("\n" if self.lines else "")

    def view(
        self,
        start: int = 1,
        end: Optional[int] = None,
        context: int = 0,
        highlight_range: Optional[Tuple[int, int]] = None,
    ) -> str:
        """
        Render lines with 1-based line numbers.
        Example output:
           45 |     def step(self):
           46 |         self.age += 1
        """
        total = len(self.lines)
        if total == 0:
            return "(file is empty)"

        start_idx = max(1, start - context)
        end_idx = min(total, (end if end is not None else total) + context)

        if start_idx > total:
            return f"(start line {start} is beyond end of file ({total} lines))"

        output_lines = []
        width = len(str(end_idx))
        for i in range(start_idx, end_idx + 1):
            line_content = self.lines[i - 1]
            marker = " "
            if highlight_range and highlight_range[0] <= i <= highlight_range[1]:
                marker = ">"
            output_lines.append(f"{marker}{i:{width}d} | {line_content}")

        return "\n".join(output_lines)

    def _save_undo(self):
        """Push current state onto undo stack and clear redo stack."""
        self.undo_stack.append(list(self.lines))
        self.redo_stack.clear()
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self) -> Tuple[bool, str]:
        """Revert the most recent edit."""
        if not self.undo_stack:
            return False, "Undo stack is empty."
        self.redo_stack.append(list(self.lines))
        self.lines = self.undo_stack.pop()
        valid, msg = self.validate_syntax()
        return True, f"Reverted last edit ({len(self.lines)} lines total). {msg}"

    def redo(self) -> Tuple[bool, str]:
        """Redo the most recently reverted edit."""
        if not self.redo_stack:
            return False, "Redo stack is empty."
        self.undo_stack.append(list(self.lines))
        self.lines = self.redo_stack.pop()
        valid, msg = self.validate_syntax()
        return True, f"Redid edit ({len(self.lines)} lines total). {msg}"

    def replace_lines(
        self,
        start_line: int,
        end_line: int,
        new_content: str,
    ) -> Tuple[bool, str]:
        """
        Replace lines in the 1-based inclusive range [start_line, end_line]
        with new_content.
        
        Surgical cases:
          - Single line edit: start_line == end_line
          - Multi-line block: start_line < end_line
          - Deletion: new_content is empty string
          - Insertion before start_line: start_line == end_line + 1
        """
        total = len(self.lines)

        # Allow insertion at end of file
        if start_line == total + 1 and end_line == total:
            new_lines = new_content.splitlines() if new_content else []
            self._save_undo()
            self.lines.extend(new_lines)
            return self._post_edit_report(start_line, start_line + len(new_lines) - 1)

        # Normal bounds checking
        if start_line < 1:
            return False, f"Invalid start line {start_line}: must be >= 1"
        if end_line < start_line - 1:
            return False, f"Invalid range {start_line}..{end_line}: end line must be >= start_line - 1"
        if start_line > total and total > 0:
            return False, f"Start line {start_line} exceeds file length ({total} lines)"
        if end_line > total:
            return False, f"End line {end_line} exceeds file length ({total} lines)"

        new_lines = new_content.splitlines() if new_content else []

        self._save_undo()

        # Perform replacement
        # 1-indexed to 0-indexed: start_line - 1 to end_line
        self.lines[start_line - 1 : end_line] = new_lines

        return self._post_edit_report(start_line, start_line + len(new_lines) - 1)

    def insert_lines(
        self,
        line_number: int,
        new_content: str,
        after: bool = True,
    ) -> Tuple[bool, str]:
        """
        Insert new_content before or after line_number (1-indexed).
        """
        total = len(self.lines)
        if line_number < 0 or line_number > total:
            return False, f"Line number {line_number} out of bounds (1..{total})"

        target_idx = line_number if after else max(0, line_number - 1)
        new_lines = new_content.splitlines() if new_content else []

        self._save_undo()
        self.lines[target_idx:target_idx] = new_lines

        start_mod = target_idx + 1
        end_mod = target_idx + len(new_lines)
        return self._post_edit_report(start_mod, end_mod)

    def delete_lines(self, start_line: int, end_line: int) -> Tuple[bool, str]:
        """Delete lines from start_line to end_line inclusive (1-indexed)."""
        return self.replace_lines(start_line, end_line, "")

    def str_replace(
        self,
        search_block: str,
        replace_block: str,
    ) -> Tuple[bool, str]:
        """
        Search for an exact code block and replace it.
        Ensures the search block is unique to prevent accidental edits elsewhere.
        """
        current_text = self.get_content()
        count = current_text.count(search_block)

        if count == 0:
            return False, (
                "Search block not found. Make sure the search block matches "
                "the exact characters, leading indentation, and newlines."
            )
        if count > 1:
            return False, (
                f"Search block found {count} times in the file. "
                "Include more surrounding context lines to make it unique."
            )

        # Locate line range for reporting
        char_idx = current_text.find(search_block)
        start_line = current_text[:char_idx].count("\n") + 1
        search_lines_count = search_block.count("\n") + 1
        end_line = start_line + search_lines_count - 1

        self._save_undo()
        new_text = current_text.replace(search_block, replace_block, 1)
        self.lines = new_text.splitlines()

        replace_lines_count = replace_block.count("\n") + (1 if replace_block else 0)
        new_end_line = start_line + replace_lines_count - 1

        return self._post_edit_report(start_line, max(start_line, new_end_line))

    def validate_syntax(self) -> Tuple[bool, str]:
        """
        Validate Python syntax using ast.parse.
        Returns (is_valid, diagnostic_message).
        """
        code = self.get_content()
        try:
            tree = ast.parse(code, filename=self.filename)
            node_count = sum(1 for _ in ast.walk(tree))
            return True, f"Syntax OK ({len(self.lines)} lines, {node_count} AST nodes)"
        except SyntaxError as e:
            line_str = f"line {e.lineno}" if e.lineno else "unknown line"
            col_str = f", col {e.offset}" if e.offset else ""
            error_line = (e.text or "").strip()
            return False, (
                f"SyntaxError at {line_str}{col_str}: {e.msg}\n"
                f"  --> {error_line}"
            )
        except Exception as e:
            return False, f"Code parsing error: {e}"

    def diff(self) -> str:
        """Generate a unified diff between original and current content."""
        orig_lines = [l + "\n" for l in self.original_content.splitlines()]
        curr_lines = [l + "\n" for l in self.lines]
        diff = difflib.unified_diff(
            orig_lines,
            curr_lines,
            fromfile=f"a/{self.filename}",
            tofile=f"b/{self.filename}",
        )
        return "".join(diff)

    def _post_edit_report(self, start_line: int, end_line: int) -> Tuple[bool, str]:
        """Generate observation report after an edit, including syntax check and view snippet."""
        valid, syntax_msg = self.validate_syntax()
        view_snippet = self.view(
            start=max(1, start_line - 2),
            end=min(len(self.lines), max(start_line, end_line) + 2),
            highlight_range=(start_line, max(start_line, end_line)),
        )

        status_symbol = "✓" if valid else "❌"
        report = (
            f"{status_symbol} {syntax_msg}\n"
            f"Modified lines {start_line}..{end_line}:\n"
            f"{view_snippet}"
        )
        return valid, report


# ---------------------------------------------------------------------------
# Command Action Definitions & Parser
# ---------------------------------------------------------------------------

@dataclass
class EditAction:
    command: str  # REPLACE, INSERT, DELETE, SEARCH_REPLACE, VIEW, UNDO, DIFF, SUBMIT
    start_line: int = 0
    end_line: int = 0
    line_number: int = 0
    content: str = ""
    search_block: str = ""
    replace_block: str = ""
    description: str = ""
    thought: str = ""
    raw: str = ""


class ActionParser:
    """
    Robust parser extracting surgical editing commands and thoughts from
    model responses. Supports both SWE-agent commands and markdown blocks.
    """

    @staticmethod
    def parse(response_text: str) -> EditAction:
        text = response_text.strip()
        thought = ""

        # Extract thought if present
        thought_match = re.search(r"(?:THOUGHT|REASONING|PLAN)[:\s]+(.*?)(?=\n[A-Z_]+[:\s]|$)", text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # 1. SUBMIT command
        submit_match = re.search(r"(?:SUBMIT|COMPLETE|DONE)(?:\s+(.*))?", text, re.IGNORECASE)
        if submit_match and not any(k in text.upper() for k in ["REPLACE ", "INSERT ", "<<<<<<< SEARCH"]):
            desc = (submit_match.group(1) or "").strip()
            return EditAction(command="SUBMIT", description=desc or thought or "Ablation completed", thought=thought, raw=text)

        # 2. SEARCH_REPLACE (Aider / SWE-agent style)
        if "<<<<<<< SEARCH" in text and "=======" in text and ">>>>>>>" in text:
            sr_match = re.search(r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>", text, re.DOTALL)
            if sr_match:
                search_part = sr_match.group(1)
                replace_part = sr_match.group(2)
                return EditAction(
                    command="SEARCH_REPLACE",
                    search_block=search_part,
                    replace_block=replace_part,
                    thought=thought,
                    raw=text,
                )

        # 3. REPLACE <start> <end>
        replace_match = re.search(
            r"REPLACE\s+(\d+)(?:\s*[:,\-\s]\s*|\s+)(\d+)\s*[:\s]*\n(?:```(?:python)?\s*\n)?(.*?)(?:```|END_REPLACE|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if replace_match:
            start = int(replace_match.group(1))
            end = int(replace_match.group(2))
            code = replace_match.group(3).rstrip()
            code = re.sub(r"\n?END_REPLACE\s*$", "", code, flags=re.IGNORECASE)
            return EditAction(
                command="REPLACE",
                start_line=start,
                end_line=end,
                content=code,
                thought=thought,
                raw=text,
            )

        # 4. Single-line REPLACE <line>
        single_replace_match = re.search(
            r"REPLACE\s+(\d+)\s*[:\s]*\n(?:```(?:python)?\s*\n)?(.*?)(?:```|END_REPLACE|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if single_replace_match:
            line_no = int(single_replace_match.group(1))
            code = single_replace_match.group(2).rstrip()
            code = re.sub(r"\n?END_REPLACE\s*$", "", code, flags=re.IGNORECASE)
            return EditAction(
                command="REPLACE",
                start_line=line_no,
                end_line=line_no,
                content=code,
                thought=thought,
                raw=text,
            )

        # 5. INSERT <line>
        insert_match = re.search(
            r"INSERT\s+(?:AFTER\s+)?(\d+)\s*[:\s]*\n(?:```(?:python)?\s*\n)?(.*?)(?:```|END_INSERT|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if insert_match:
            line_no = int(insert_match.group(1))
            code = insert_match.group(2).rstrip()
            code = re.sub(r"\n?END_INSERT\s*$", "", code, flags=re.IGNORECASE)
            return EditAction(
                command="INSERT",
                line_number=line_no,
                content=code,
                thought=thought,
                raw=text,
            )

        # 6. DELETE <start> <end>
        delete_match = re.search(
            r"DELETE\s+(\d+)(?:\s*[:,\-\s]\s*|\s+)(\d+)",
            text,
            re.IGNORECASE,
        )
        if delete_match:
            start = int(delete_match.group(1))
            end = int(delete_match.group(2))
            return EditAction(
                command="DELETE",
                start_line=start,
                end_line=end,
                thought=thought,
                raw=text,
            )

        # 7. Single-line DELETE <line>
        single_delete_match = re.search(r"DELETE\s+(\d+)", text, re.IGNORECASE)
        if single_delete_match:
            line_no = int(single_delete_match.group(1))
            return EditAction(
                command="DELETE",
                start_line=line_no,
                end_line=line_no,
                thought=thought,
                raw=text,
            )

        # 8. VIEW <start> <end>
        view_match = re.search(r"VIEW\s+(\d+)(?:\s*[:,\-\s]\s*|\s+)(\d+)", text, re.IGNORECASE)
        if view_match:
            return EditAction(
                command="VIEW",
                start_line=int(view_match.group(1)),
                end_line=int(view_match.group(2)),
                thought=thought,
                raw=text,
            )

        # 9. UNDO
        if re.search(r"\bUNDO\b", text, re.IGNORECASE):
            return EditAction(command="UNDO", thought=thought, raw=text)

        # 10. DIFF
        if re.search(r"\bDIFF\b", text, re.IGNORECASE):
            return EditAction(command="DIFF", thought=thought, raw=text)

        # Fallback: check if the response is a SUBMIT with descriptive text
        if "SUBMIT" in text.upper():
            return EditAction(command="SUBMIT", description=thought or text[:200], thought=thought, raw=text)

        return EditAction(command="UNKNOWN", raw=text, thought=thought)

    @staticmethod
    def parse_all_blocks(response_text: str) -> List[EditAction]:
        """
        Extract multiple sequential block edits from a single response.
        Useful for batch / single-turn patch execution.
        """
        actions: List[EditAction] = []

        # Find all SEARCH/REPLACE blocks
        for m in re.finditer(r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>", response_text, re.DOTALL):
            actions.append(
                EditAction(
                    command="SEARCH_REPLACE",
                    search_block=m.group(1),
                    replace_block=m.group(2),
                )
            )

        # Find all REPLACE <start> <end> blocks
        for m in re.finditer(
            r"REPLACE\s+(\d+)(?:\s*[:,\-\s]\s*|\s+)(\d+)\s*[:\s]*\n(?:```(?:python)?\s*\n)?(.*?)(?:```|END_REPLACE|$)",
            response_text,
            re.DOTALL | re.IGNORECASE,
        ):
            code = m.group(3).rstrip()
            code = re.sub(r"\n?END_REPLACE\s*$", "", code, flags=re.IGNORECASE)
            actions.append(
                EditAction(
                    command="REPLACE",
                    start_line=int(m.group(1)),
                    end_line=int(m.group(2)),
                    content=code,
                )
            )

        # Find all DELETE <start> <end> blocks
        for m in re.finditer(r"DELETE\s+(\d+)(?:\s*[:,\-\s]\s*|\s+)(\d+)", response_text, re.IGNORECASE):
            actions.append(
                EditAction(
                    command="DELETE",
                    start_line=int(m.group(1)),
                    end_line=int(m.group(2)),
                )
            )

        return actions


# ---------------------------------------------------------------------------
# Gemma Model Wrapper
# ---------------------------------------------------------------------------

class GemmaModelClient:
    """
    Clean wrapper for invoking Gemma models (via google.genai or mock fallback).
    Handles retry logic, token parameters, and system prompts.
    """

    def __init__(
        self,
        model_name: str = "gemma-4-26b-a4b-it",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        temperature: float = 0.5,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self._client = client
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except ImportError:
            raise RuntimeError(
                "google-genai is not installed in this environment. "
                "Install with `pip install google-genai` or pass a client instance."
            )

    def generate(self, prompt: str, system_instruction: str = "", max_output_tokens: int = 4096) -> str:
        """Call Gemma with the specified prompt and return the response text."""
        client = self._get_client()
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction or None,
        )

        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                if response and response.text:
                    return response.text.strip()
                return ""
            except Exception as e:
                err_str = str(e)
                if ("500" in err_str or "503" in err_str or "ServerError" in type(e).__name__) and attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise e
        return ""


# ---------------------------------------------------------------------------
# MiniSWEAgent: Core Agent Loop
# ---------------------------------------------------------------------------

class MiniSWEAgent:
    """
    Lightweight, interactive software engineering agent inspired by mini-swe-agent.
    
    The agent receives the code with line numbers, reasons about simplifications,
    issues surgical edit commands (REPLACE, DELETE, INSERT, SEARCH_REPLACE),
    receives observations (diffs + AST syntax diagnostics), and self-corrects
    if syntax errors occur until submitting the completed ablation.
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert software engineer and research agent performing structural ablation.
    Your goal is to simplify Python code by making surgical single-line or block edits.
    
    DO NOT output the entire file. Instead, use the commands below to modify only
    the specific lines you want to simplify.
    
    AVAILABLE COMMANDS:
    1. Replace lines:
       REPLACE <start_line> <end_line>
       ```python
       <replacement code>
       ```
       (To replace a single line, use REPLACE <line> <line>)
    
    2. Delete lines:
       DELETE <start_line> <end_line>
    
    3. Insert lines after a line:
       INSERT <line_number>
       ```python
       <code to insert>
       ```
    
    4. Exact search & replace:
       <<<<<<< SEARCH
       <exact code to find>
       =======
       <replacement code>
       >>>>>>>
    
    5. View lines:
       VIEW <start_line> <end_line>
    
    6. Undo previous edit:
       UNDO
    
    7. Submit when done:
       SUBMIT <brief description of what was simplified>
    
    FORMAT YOUR RESPONSE:
    THOUGHT: <Brief 1-2 sentence explanation of the specific simplification you are making>
    <COMMAND>
    
    CRITICAL RULES:
    - Maintain valid Python syntax and exact indentation.
    - If a syntax error is reported, fix it immediately in the next turn.
    - Each turn should execute ONE command. When your simplification is complete, call SUBMIT.
    """)

    def __init__(
        self,
        model: GemmaModelClient,
        editor: CodeEditor,
        max_steps: int = 6,
        verbose: bool = True,
    ):
        self.model = model
        self.editor = editor
        self.max_steps = max_steps
        self.verbose = verbose
        self.history: List[Dict[str, str]] = []

    def _log(self, msg: str):
        if self.verbose:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [EditHarness] {msg}", flush=True)

    def run(self, task_instruction: str) -> Tuple[bool, str, str, str]:
        """
        Execute the agent loop.
        Returns: (success, final_code, description, diff)
        """
        self._log(f"Starting agent run (max_steps={self.max_steps})")
        
        # Initial user turn with code view
        numbered_code = self.editor.view(1, len(self.editor.lines))
        initial_prompt = textwrap.dedent(f"""\
        ## Task:
        {task_instruction}
        
        ## Target File ({self.editor.filename}):
        ```python
        {numbered_code}
        ```
        
        Inspect the code, plan your first surgical edit, and issue a command.
        """)

        conversation: List[Dict[str, str]] = [
            {"role": "user", "content": initial_prompt}
        ]

        description = "ablation variant"
        syntax_valid = True

        for step in range(1, self.max_steps + 1):
            self._log(f"Turn {step}/{self.max_steps}: Prompting Gemma...")

            full_prompt = self._format_conversation(conversation)

            try:
                response = self.model.generate(
                    prompt=full_prompt,
                    system_instruction=self.SYSTEM_PROMPT,
                )
            except Exception as e:
                self._log(f"Model generation error: {e}")
                return False, self.editor.get_content(), f"Model error: {e}", self.editor.diff()

            if not response:
                self._log("Received empty response from model.")
                break

            action = ActionParser.parse(response)
            self._log(f"Action parsed: {action.command} (thought: {action.thought[:60]}...)")

            # Handle SUBMIT
            if action.command == "SUBMIT":
                description = action.description or action.thought or description
                valid, msg = self.editor.validate_syntax()
                if not valid:
                    self._log(f"Cannot submit with syntax error: {msg}")
                    conversation.append({"role": "assistant", "content": response})
                    conversation.append({
                        "role": "user",
                        "content": f"❌ Cannot SUBMIT: The code has a syntax error:\n{msg}\nPlease fix it before submitting.",
                    })
                    continue
                self._log(f"✓ SUBMIT received: {description}")
                return True, self.editor.get_content(), description, self.editor.diff()

            # Execute command on editor
            obs_valid, observation = self._execute_action(action)
            syntax_valid = obs_valid

            conversation.append({"role": "assistant", "content": response})
            conversation.append({
                "role": "user",
                "content": f"OBSERVATION:\n{observation}\n\nWhat is your next action? (Issue another edit or call SUBMIT if finished)",
            })

        # Max steps reached: ensure syntax validity
        final_valid, final_msg = self.editor.validate_syntax()
        if not final_valid:
            self._log(f"Agent finished with syntax error, rolling back to last valid state...")
            while self.editor.undo_stack and not final_valid:
                self.editor.undo()
                final_valid, _ = self.editor.validate_syntax()

        return final_valid, self.editor.get_content(), description, self.editor.diff()

    def _execute_action(self, action: EditAction) -> Tuple[bool, str]:
        """Execute a parsed action on the editor and return (is_valid, observation)."""
        if action.command == "REPLACE":
            return self.editor.replace_lines(action.start_line, action.end_line, action.content)

        elif action.command == "INSERT":
            return self.editor.insert_lines(action.line_number, action.content, after=True)

        elif action.command == "DELETE":
            return self.editor.delete_lines(action.start_line, action.end_line)

        elif action.command == "SEARCH_REPLACE":
            return self.editor.str_replace(action.search_block, action.replace_block)

        elif action.command == "VIEW":
            snippet = self.editor.view(action.start_line, action.end_line)
            return True, f"Code lines {action.start_line}..{action.end_line}:\n{snippet}"

        elif action.command == "UNDO":
            return self.editor.undo()

        elif action.command == "DIFF":
            diff_text = self.editor.diff()
            return True, f"Current diff:\n{diff_text or '(no changes from original)'}"

        else:
            return False, (
                f"Unknown command. Available commands: "
                "REPLACE <start> <end>, DELETE <start> <end>, INSERT <line>, "
                "<<<<<<< SEARCH ... ======= ... >>>>>>>, VIEW <start> <end>, UNDO, SUBMIT."
            )

    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format message history into a single structured prompt."""
        formatted = []
        for msg in conversation:
            role = msg["role"].upper()
            content = msg["content"]
            formatted.append(f"=== {role} ===\n{content}\n")
        return "\n".join(formatted)


# ---------------------------------------------------------------------------
# Single-Turn Batch Patch Applicator
# ---------------------------------------------------------------------------

class SingleTurnPatchApplicator:
    """
    Fast, single-round patch applicator. Gemma is prompted to output one or
    more surgical edit blocks in a single response, which are then applied
    and validated without requiring multiple interactive turns.
    """

    BATCH_PROMPT_TEMPLATE = textwrap.dedent("""\
    ## Task
    {task}
    
    ## Baseline Context & Goals
    Reduce complexity score while keeping metrics valid.
    
    ## Current Strategy Code ({filename}):
    ```python
    {numbered_code}
    ```
    
    INSTRUCTIONS:
    Propose 1 to 4 surgical edits to simplify this code.
    DO NOT output the entire file! Only output the specific lines being changed.
    
    Output your edits using ONE of these formats:
    
    FORMAT A (Line-based replace):
    REPLACE <start_line> <end_line>:
    ```python
    <new code>
    ```
    
    FORMAT B (Delete lines):
    DELETE <start_line> <end_line>
    
    FORMAT C (Search & Replace):
    <<<<<<< SEARCH
    <exact old lines to find>
    =======
    <new replacement lines>
    >>>>>>>
    
    CRITICAL:
    - Target only specific helper methods, conditions, or loops to simplify.
    - Leave create_model(), run_model(), and SugarScapeScenario definitions untouched.
    
    At the end of your response, write:
    SUBMIT: <one-line description of the simplification>
    """)

    @classmethod
    def apply_response(
        cls,
        code: str,
        response_text: str,
        filename: str = "strategy.py",
    ) -> Tuple[bool, str, str, str]:
        """
        Apply all edit blocks found in response_text to code.
        Returns: (success, modified_code, description, diff)
        """
        editor = CodeEditor(code, filename=filename)
        actions = ActionParser.parse_all_blocks(response_text)

        # Extract description
        desc_match = re.search(r"SUBMIT[:\s]+(.*)", response_text, re.IGNORECASE)
        description = desc_match.group(1).strip() if desc_match else "Surgical ablation"

        if not actions:
            single = ActionParser.parse(response_text)
            if single.command not in ("UNKNOWN", "SUBMIT"):
                actions = [single]

        if not actions:
            return False, code, "No valid edit blocks parsed", ""

        # Sort line-based actions in descending order of start_line
        line_actions = [a for a in actions if a.command in ("REPLACE", "DELETE", "INSERT")]
        sr_actions = [a for a in actions if a.command == "SEARCH_REPLACE"]

        line_actions.sort(key=lambda a: (a.start_line or a.line_number), reverse=True)

        # Apply SEARCH/REPLACE blocks first
        for action in sr_actions:
            success, msg = editor.str_replace(action.search_block, action.replace_block)
            if not success:
                return False, code, f"Search/Replace failed: {msg}", ""

        # Apply line actions in reverse line order
        for action in line_actions:
            if action.command == "REPLACE":
                success, msg = editor.replace_lines(action.start_line, action.end_line, action.content)
            elif action.command == "DELETE":
                success, msg = editor.delete_lines(action.start_line, action.end_line)
            elif action.command == "INSERT":
                success, msg = editor.insert_lines(action.line_number, action.content)
            else:
                success = True

            if not success:
                return False, code, f"Edit failed at lines {action.start_line}..{action.end_line}: {msg}", ""

        valid, syntax_msg = editor.validate_syntax()
        if not valid:
            return False, code, f"Edits resulted in {syntax_msg}", ""

        return True, editor.get_content(), description, editor.diff()


# ---------------------------------------------------------------------------
# Autoresearch Integration Wrapper
# ---------------------------------------------------------------------------

def generate_ablation_variants_harness(
    client: Any,
    current_strategy: str,
    results_history: str,
    baseline_metrics: dict,
    complexity: dict,
    program_text: str = "",
    n_variants: int = 1,
    mode: str = "batch",  # "batch" (single-round blocks) or "agent" (interactive multi-turn)
    model_name: str = "gemma-4-26b-a4b-it",
    temperature: float = 0.7,
    max_agent_steps: int = 5,
) -> List[Tuple[str, str]]:
    """
    Drop-in replacement for autoresearch.py's generate_ablation_variants.
    
    Instead of asking Gemma to reproduce all of strategy.py (~450 lines),
    it instructs Gemma to perform surgical edits, resulting in faster responses,
    fewer token limits hit, and lower likelihood of syntax errors.
    
    Returns:
        List of (modified_code, description) tuples.
    """
    model_wrapper = GemmaModelClient(
        model_name=model_name,
        client=client,
        temperature=temperature,
    )

    baseline_json = json.dumps(baseline_metrics, indent=2) if isinstance(baseline_metrics, dict) else str(baseline_metrics)
    
    task_parts = []
    if program_text:
        task_parts.append(program_text.strip())
        task_parts.append("\n---\n")

    task_parts.append(textwrap.dedent(f"""\
    ## Current Complexity:
    - Score: {complexity.get('combined_score', 0):.1f}
    - AST Nodes: {complexity.get('ast_nodes', 0)}
    - Cyclomatic Complexity: {complexity.get('cyclomatic_total', 0)}
    - Lines: {complexity.get('lines', 0)}

    ## Target Baseline Metrics to Preserve:
    ```json
    {baseline_json}
    ```

    ## Results History (learn from past ablations):
    ```
    {results_history[-1000:] if results_history else '(none)'}
    ```

    ## CRITICAL API & CODE CONTRACT:
    - DO NOT rewrite the entire file or whole classes!
    - DO NOT change the signature or return format of create_model() or run_model().
    - SugarScapeScenario MUST accept rng and kwargs (sc = SugarScapeScenario(rng=seed, **scenario_kwargs)).
    - Focus on surgical ablations: inline helper methods, simplify trade calculations, prune dead branches.
    """))

    task = "\n".join(task_parts)

    variants: List[Tuple[str, str]] = []

    for v_idx in range(n_variants):
        if mode == "agent":
            editor = CodeEditor(current_strategy, filename="strategy.py")
            agent = MiniSWEAgent(model_wrapper, editor, max_steps=max_agent_steps, verbose=True)
            success, code, desc, diff = agent.run(task)
            if success and "def create_model" in code and "def run_model" in code:
                variants.append((code, desc))
        else:
            # Batch mode: single prompt with numbered lines
            editor = CodeEditor(current_strategy, filename="strategy.py")
            numbered_code = editor.view(1, len(editor.lines))
            prompt = SingleTurnPatchApplicator.BATCH_PROMPT_TEMPLATE.format(
                task=task,
                filename="strategy.py",
                numbered_code=numbered_code,
            )
            try:
                response = model_wrapper.generate(prompt=prompt)
                success, code, desc, diff = SingleTurnPatchApplicator.apply_response(
                    current_strategy, response, filename="strategy.py"
                )
                if success and "def create_model" in code and "def run_model" in code:
                    variants.append((code, desc))
                else:
                    print(f"[EditHarness] Batch apply unsuccessful: {desc}", file=sys.stderr)
            except Exception as e:
                print(f"[EditHarness] Generation error: {e}", file=sys.stderr)

    return variants


# ---------------------------------------------------------------------------
# Self-Test Suite (Dependency-Free)
# ---------------------------------------------------------------------------

def run_self_test():
    """
    Run self-contained unit tests covering CodeEditor, ActionParser,
    AST syntax validation, undo/redo, and patch application without
    requiring external libraries or API keys.
    """
    print("=" * 60)
    print("Running EditHarness Built-in Self-Tests")
    print("=" * 60)

    sample_code = textwrap.dedent("""\
    class SugarAgent:
        def __init__(self, pos, sugar):
            self.pos = pos
            self.sugar = sugar
            self.alive = True

        def step(self):
            # Consume sugar
            self.sugar -= 1
            if self.sugar <= 0:
                self.alive = False

        def trade(self, other):
            if self.sugar > other.sugar:
                self.sugar -= 1
                other.sugar += 1
    """)

    # Test 1: CodeEditor Initialization & Line Numbered View
    editor = CodeEditor(sample_code, filename="test.py")
    assert len(editor.lines) == 16, f"Expected 16 lines, got {len(editor.lines)}"
    view_out = editor.view(1, 5)
    assert "1 | class SugarAgent:" in view_out
    assert "5 |         self.alive = True" in view_out
    print("✓ Test 1: Initialization & Line Numbered View passed")

    # Test 2: Single-Line Replacement
    success, rep = editor.replace_lines(9, 9, "        self.sugar -= 2  # Accelerated metabolism")
    assert success, f"Single line replace failed: {rep}"
    assert "self.sugar -= 2" in editor.lines[8]
    valid, _ = editor.validate_syntax()
    assert valid, "Syntax should be valid after single-line edit"
    print("✓ Test 2: Single-Line Replacement passed")

    # Test 3: Multi-Line Block Replacement
    new_step = textwrap.dedent("""\
        def step(self):
            self.sugar -= 1
            self.alive = (self.sugar > 0)
    """).strip()
    success, rep = editor.replace_lines(7, 11, new_step)
    assert success, f"Block replace failed: {rep}"
    assert "self.alive = (self.sugar > 0)" in editor.get_content()
    valid, _ = editor.validate_syntax()
    assert valid, "Syntax should be valid after block replace"
    print("✓ Test 3: Multi-Line Block Replacement passed")

    # Test 4: Deletion
    # In the updated 14-line file, trade method is at lines 10-14
    initial_len = len(editor.lines)
    success, rep = editor.delete_lines(10, 14)
    assert success, f"Delete failed: {rep}"
    assert len(editor.lines) < initial_len
    assert "def trade" not in editor.get_content()
    valid, _ = editor.validate_syntax()
    assert valid, "Syntax should be valid after deletion"
    print("✓ Test 4: Deletion passed")

    # Test 5: Undo Stack
    undo_success, undo_msg = editor.undo()
    assert undo_success
    assert "def trade" in editor.get_content(), "Undo should restore trade method"
    print("✓ Test 5: Undo functionality passed")

    # Test 6: Syntax Error Detection
    success, rep = editor.replace_lines(1, 1, "class SugarAgent(")
    assert not success, "Syntax validation should flag unmatched parenthesis"
    assert "SyntaxError" in rep
    editor.undo()
    valid, _ = editor.validate_syntax()
    assert valid, "Undo should restore valid syntax"
    print("✓ Test 6: AST Syntax Error Detection passed")

    # Test 7: Exact Search & Replace (Aider / SWE-agent style)
    success, rep = editor.str_replace("self.alive = True", "self.alive = True\n        self.age = 0")
    assert success, f"str_replace failed: {rep}"
    assert "self.age = 0" in editor.get_content()
    print("✓ Test 7: Exact Search & Replace passed")

    # Test 8: ActionParser Extraction
    llm_sample = textwrap.dedent("""\
    THOUGHT: I want to simplify trade by removing the sugar check.
    REPLACE 14 16:
    ```python
            self.sugar -= 1
            other.sugar += 1
    ```
    """)
    action = ActionParser.parse(llm_sample)
    assert action.command == "REPLACE"
    assert action.start_line == 14
    assert action.end_line == 16
    assert "other.sugar += 1" in action.content
    print("✓ Test 8: ActionParser passed")

    # Test 9: SingleTurnPatchApplicator (Batch block mode)
    batch_sample = textwrap.dedent("""\
    I propose simplifying the agent:
    
    REPLACE 9 9:
    ```python
            self.sugar -= 5
    ```
    
    SUBMIT: Ablated metabolism rate
    """)
    applied, mod_code, desc, diff = SingleTurnPatchApplicator.apply_response(sample_code, batch_sample)
    assert applied, f"Batch apply failed: {desc}"
    assert "self.sugar -= 5" in mod_code
    assert desc == "Ablated metabolism rate"
    assert len(diff) > 0
    print("✓ Test 9: SingleTurnPatchApplicator passed")

    print("=" * 60)
    print("All 9 Self-Tests Passed Successfully!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Mini-SWE-Agent style surgical edit harness for Gemma"
    )
    parser.add_argument("--file", "-f", default="strategy.py", help="File to edit")
    parser.add_argument("--task", "-t", default="", help="Task or ablation instruction")
    parser.add_argument("--mode", choices=["agent", "batch"], default="batch", help="Editing mode")
    parser.add_argument("--model", default="gemma-4-26b-a4b-it", help="Gemma model name")
    parser.add_argument("--max-steps", type=int, default=6, help="Max turns in agent mode")
    parser.add_argument("--dry-run", action="store_true", help="Print diff without modifying file")
    parser.add_argument("--self-test", action="store_true", help="Run internal unit tests")

    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    target_path = Path(args.file)
    if not target_path.exists():
        print(f"Error: Target file '{args.file}' does not exist.", file=sys.stderr)
        return 1

    content = target_path.read_text(encoding="utf-8")
    task = args.task or "Simplify this code while preserving its external behavior."

    model = GemmaModelClient(model_name=args.model)

    if args.mode == "agent":
        editor = CodeEditor(content, filename=target_path.name)
        agent = MiniSWEAgent(model, editor, max_steps=args.max_steps, verbose=True)
        success, new_code, desc, diff = agent.run(task)
    else:
        editor = CodeEditor(content, filename=target_path.name)
        prompt = SingleTurnPatchApplicator.BATCH_PROMPT_TEMPLATE.format(
            task=task,
            filename=target_path.name,
            numbered_code=editor.view(1, len(editor.lines)),
        )
        print(f"Prompting {args.model} (batch block mode)...")
        response = model.generate(prompt=prompt)
        success, new_code, desc, diff = SingleTurnPatchApplicator.apply_response(
            content, response, filename=target_path.name
        )

    if not success:
        print(f"Edit failed: {desc}", file=sys.stderr)
        return 1

    print(f"\n✓ Success: {desc}")
    print("\n--- Diff ---")
    print(diff)

    if not args.dry_run:
        target_path.write_text(new_code, encoding="utf-8")
        print(f"\nSaved changes to {args.file}")
    else:
        print("\nDry-run mode: file not written.")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
