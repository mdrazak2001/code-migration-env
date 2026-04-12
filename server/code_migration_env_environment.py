# code_migration_env\server\code_migration_env_environment.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Code Migration Env Environment Implementation.

This environment evaluates production-style code migration tasks. It supports
iterative refinement by returning actionable feedback and partial credit across
multiple attempts.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import CodeMigrationAction, CodeMigrationObservation
except ImportError:
    from models import CodeMigrationAction, CodeMigrationObservation


class CodeMigrationEnvironment(Environment):
    SCORE_EPSILON = 0.01
    TARGET_SUCCESS_IDIOM_SCORE = 0.8

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, scenario_dir: str = None):
        repo_root = Path(__file__).resolve().parent.parent
        default_scenario = repo_root / "scenarios" / "easy"

        if scenario_dir is None:
            scenario_dir = os.environ.get("SCENARIO_DIR", str(default_scenario))

        self.scenario_dir = Path(scenario_dir)
        self.task_meta = json.loads((self.scenario_dir / "meta.json").read_text())
        self.attempts = 0
        self.max_attempts = 3
        self.history: List[str] = []
        self.source_code = (self.scenario_dir / "source.py").read_text()

    def reset(self, seed=None, episode_id=None, **kwargs):
        if not episode_id:
            episode_id = "easy"

        repo_root = Path(__file__).resolve().parent.parent
        base_dir = Path(os.environ.get("SCENARIOS_BASE", str(repo_root / "scenarios")))

        self.scenario_dir = base_dir / episode_id
        self.task_meta = json.loads((self.scenario_dir / "meta.json").read_text())
        self.source_code = (self.scenario_dir / "source.py").read_text()
        self.attempts = 0
        self.history = []

        return CodeMigrationObservation(
            task_id=self.task_meta["id"],
            difficulty=self.task_meta["difficulty"],
            source_code=self.source_code,
            source_language=self.task_meta["source_lang"],
            target_language=self.task_meta["target_lang"],
            requirements=self.task_meta["requirements"],
            test_description=self.task_meta["test_desc"],
            history=self.history,
            info=self._build_task_info(),
        )

    def _validate_syntax(self, code: str, lang: str) -> Tuple[bool, str]:
        if lang == "python":
            try:
                import ast

                ast.parse(code)
                return True, "Syntax OK"
            except SyntaxError as exc:
                return False, f"SyntaxError: {exc}"

        if lang == "javascript":
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as handle:
                    handle.write(code)
                    tmp = handle.name
                result = subprocess.run(
                    ["node", "--check", tmp],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                os.unlink(tmp)
                return result.returncode == 0, result.stderr or "Syntax OK"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return True, "Syntax check skipped (node unavailable)"

        return True, "Syntax check skipped"

    def step(self, action: CodeMigrationAction, timeout_s=None, **kwargs) -> CodeMigrationObservation:
        self.attempts += 1
        code = action.translated_code

        syntax_ok, syntax_msg = self._validate_syntax(code, self.task_meta["target_lang"])
        idiom_score = self._grade_idioms(code, self.task_meta["required_idioms"])
        missing_idioms = self._get_missing_idioms(code, self.task_meta["required_idioms"])

        if not syntax_ok:
            reward = self._clamp_open_score(self.SCORE_EPSILON + (0.05 * idiom_score))
            done = self.attempts >= self.max_attempts
            feedback = self._build_feedback(
                syntax_ok=False,
                syntax_msg=syntax_msg,
                test_ok=False,
                test_msg="Tests were skipped because the submission did not parse.",
                idiom_score=idiom_score,
                missing_idioms=missing_idioms,
            )
            obs = self._get_observation(feedback)
            obs.reward = reward
            obs.done = done
            return obs

        reward = 0.3
        test_ok, test_msg = self._run_tests(code, self.task_meta["target_lang"])
        test_progress = self._extract_test_progress(test_msg, test_ok)
        reward += 0.2 * test_progress

        reward += idiom_score * 0.4
        reward -= 0.02 * max(0, self.attempts - 1)
        reward = self._clamp_open_score(reward)

        done = (
            (test_ok and idiom_score >= self.TARGET_SUCCESS_IDIOM_SCORE)
            or self.attempts >= self.max_attempts
        )

        feedback = self._build_feedback(
            syntax_ok=True,
            syntax_msg=syntax_msg,
            test_ok=test_ok,
            test_msg=test_msg,
            idiom_score=idiom_score,
            missing_idioms=missing_idioms,
        )
        obs = self._get_observation(feedback)
        obs.reward = reward
        obs.done = done

        info = {
            "attempt": self.attempts,
            "final_reward": reward,
            "tests_passed": test_ok,
            "test_progress": test_progress,
            "idiom_score": idiom_score,
            "missing_idioms": missing_idioms,
            "attempts_remaining": max(0, self.max_attempts - self.attempts),
            "submitted_code_preview": code[:200] + "..." if len(code) > 200 else code,
        }
        print("info: ", info)

        return obs

    def _clamp_open_score(self, score: float) -> float:
        return max(self.SCORE_EPSILON, min(1.0 - self.SCORE_EPSILON, score))

    def _build_task_info(self) -> Dict[str, Any]:
        return {
            "business_context": self.task_meta.get("business_context", ""),
            "stakeholder_request": self.task_meta.get("stakeholder_request", ""),
            "acceptance_checks": self.task_meta.get("acceptance_checks", []),
            "pitfalls": self.task_meta.get("pitfalls", []),
            "runtime_budget": self.task_meta.get("runtime_budget", ""),
            "max_attempts": self.max_attempts,
            "attempts_used": self.attempts,
            "attempts_remaining": max(0, self.max_attempts - self.attempts),
        }

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_meta["id"],
            "difficulty": self.task_meta["difficulty"],
            "source_language": self.task_meta["source_lang"],
            "target_language": self.task_meta["target_lang"],
            "attempts_used": self.attempts,
            "attempts_remaining": max(0, self.max_attempts - self.attempts),
            "history": self.history[-5:],
            "required_idioms": self.task_meta["required_idioms"],
        }

    def _run_tests(self, code: str, lang: str) -> Tuple[bool, str]:
        tests = json.loads((self.scenario_dir / "tests.json").read_text())
        timeout_s = float(self.task_meta.get("test_timeout_s", 2.0))

        if lang == "python":
            try:
                wrapper = f"CODE_SOURCE = {code!r}\n{tests.get('runner', '')}"
                result = subprocess.run(
                    [sys.executable, "-c", wrapper],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                return result.returncode == 0, result.stdout or result.stderr
            except subprocess.TimeoutExpired:
                return False, "Test timeout"
            except Exception as exc:
                return False, f"Test runner error: {exc}"

        if lang == "javascript":
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as handle:
                    handle.write(code + "\n\n" + tests.get("runner", ""))
                    tmp = handle.name

                result = subprocess.run(
                    ["node", tmp],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                os.unlink(tmp)
                return result.returncode == 0, result.stdout or result.stderr
            except subprocess.TimeoutExpired:
                return False, "Test timeout"
            except FileNotFoundError:
                return False, "Node not available"
            except Exception as exc:
                return False, f"Test runner error: {exc}"

        return True, "Tests skipped"

    def _grade_idioms(self, code: str, idioms: List[str]) -> float:
        matches = sum(1 for pattern in idioms if pattern in code)
        return matches / len(idioms) if idioms else 1.0

    def _get_missing_idioms(self, code: str, idioms: List[str]) -> List[str]:
        return [pattern for pattern in idioms if pattern not in code]

    def _extract_test_progress(self, test_msg: str, test_ok: bool) -> float:
        if test_ok:
            return 1.0

        match = re.search(r"SCORE:([0-9]*\.?[0-9]+)", str(test_msg))
        if match:
            try:
                score = float(match.group(1))
            except ValueError:
                return 0.0
            return max(0.0, min(1.0, score))

        return 0.0

    def _normalize_feedback(self, text: str) -> str:
        compact = " ".join(str(text).split())
        return compact[:240]

    def _build_feedback(
        self,
        syntax_ok: bool,
        syntax_msg: str,
        test_ok: bool,
        test_msg: str,
        idiom_score: float,
        missing_idioms: List[str],
    ) -> str:
        messages: List[str] = []

        if not syntax_ok:
            messages.append(f"Syntax issue: {self._normalize_feedback(syntax_msg)}")
        elif not test_ok:
            messages.append(f"Functional check failed: {self._normalize_feedback(test_msg)}")
        else:
            messages.append("Functional tests passed.")

        if missing_idioms:
            messages.append("Still missing target idioms: " + ", ".join(missing_idioms[:4]))
        else:
            messages.append("Target-language idioms look strong.")

        messages.append(f"Current idiom score: {idiom_score:.2f}")

        if self.attempts < self.max_attempts and (not test_ok or missing_idioms):
            messages.append("Revise the submission using this feedback and try again.")

        return " ".join(messages)

    def _get_observation(self, feedback: str) -> CodeMigrationObservation:
        self.history.append(f"Attempt {self.attempts}: {feedback}")
        return CodeMigrationObservation(
            task_id=self.task_meta["id"],
            difficulty=self.task_meta["difficulty"],
            source_code=self.source_code,
            source_language=self.task_meta["source_lang"],
            target_language=self.task_meta["target_lang"],
            requirements=self.task_meta["requirements"],
            test_description=self.task_meta["test_desc"],
            history=self.history[-5:],
            info=self._build_task_info(),
        )
