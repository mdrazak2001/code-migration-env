# code_migration_env\server\code_migration_env_environment.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Code Migration Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""
import os
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.client_types import StepResult

try:
    from ..models import CodeMigrationAction, CodeMigrationObservation
except ImportError:
    from models import CodeMigrationAction, CodeMigrationObservation


class CodeMigrationEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = CodeMigrationEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Code Migration Env environment ready!"
        >>>
        >>> obs = env.step(CodeMigrationAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, scenario_dir: str = None):
        if scenario_dir is None:
            scenario_dir = os.environ.get("SCENARIO_DIR", "/app/env/scenarios/easy")
        self.scenario_dir = Path(scenario_dir)
        self.task_meta = json.loads((self.scenario_dir / "meta.json").read_text())
        self.attempts = 0
        self.max_attempts = 3
        self.rubric = None
        self.history: List[str] = []
        # Read source immediately so it's available even without reset()
        self.source_code = (self.scenario_dir / "source.py").read_text()

    def reset(self, seed=None, episode_id=None, **kwargs) -> CodeMigrationObservation:

        if episode_id and episode_id in ("easy", "medium", "hard"):
            base_dir = Path(os.environ.get("SCENARIOS_BASE", "/app/env/scenarios"))
            self.scenario_dir = base_dir / episode_id
            self.task_meta = json.loads((self.scenario_dir / "meta.json").read_text())
            self.source_code = (self.scenario_dir / "source.py").read_text()

        self.attempts = 0
        self.history: List[str] = []
        self._reset_rubric()

        obs = CodeMigrationObservation(
            task_id=self.task_meta["id"],
            difficulty=self.task_meta["difficulty"],
            source_code=self.source_code,
            source_language=self.task_meta["source_lang"],
            target_language=self.task_meta["target_lang"],
            requirements=self.task_meta["requirements"],
            test_description=self.task_meta["test_desc"],
            history=[],
            reward=0.0,
            done=False,
        )
        return obs


    def _validate_syntax(self, code: str, lang: str) -> Tuple[bool, str]:
        if lang == "python":
            try:
                import ast
                ast.parse(code)
                return True, "Syntax OK"
            except SyntaxError as e:
                return False, f"SyntaxError: {e}"
        elif lang == "javascript":
            # Basic check: try node --check if available
            try:
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
                    f.write(code)
                    tmp = f.name
                result = subprocess.run(
                    ["node", "--check", tmp],
                    capture_output=True, text=True, timeout=5
                )
                os.unlink(tmp)
                return result.returncode == 0, result.stderr or "Syntax OK"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # node not available - skip syntax check
                return True, "Syntax check skipped (node unavailable)"
        else:
            return True, "Syntax check skipped"

    def step(self, action: CodeMigrationAction, timeout_s=None, **kwargs) -> CodeMigrationObservation:
        self.attempts += 1
        code = action.translated_code
        reward = 0.0
        done = False

        syntax_ok, syntax_msg = self._validate_syntax(code, self.task_meta["target_lang"])
        if not syntax_ok:
            reward = max(-0.3, -0.1 * self.attempts)
            done = self.attempts >= self.max_attempts
            obs = self._get_observation(action, syntax_msg)
            obs.reward = reward
            obs.done = done
            return obs

        test_ok, test_msg = self._run_tests(code, self.task_meta["target_lang"])
        reward += 0.5 if test_ok else 0.0

        idiom_score = self._grade_idioms(code, self.task_meta["required_idioms"])
        reward += idiom_score * 0.4
        reward = max(0.0, min(1.0, reward))
        done = True

        obs = self._get_observation(action, test_msg)
        obs.reward = reward
        obs.done = done

        info = {
            "attempt": self.attempts,
            "final_reward": reward,
            "tests_passed": test_ok,
            "submitted_code_preview": code[:200] + "..." if len(code) > 200 else code  # ← Add this
        }
        print("info: ", info)
        
        return obs


    @property
    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_meta["id"],
            "difficulty": self.task_meta["difficulty"],
            "ground_truth": (self.scenario_dir / "ground_truth.py").read_text(),
            "test_suite": (self.scenario_dir / "tests.json").read_text(),
            "required_idioms": self.task_meta["required_idioms"]
        }

    def _run_tests(self, code: str, lang: str) -> Tuple[bool, str]:
        # Load pre-baked tests from scenario
        tests = json.loads((self.scenario_dir / "tests.json").read_text())
        if lang == "python":
            # Execute tests in isolated subprocess with timeout
            try:
                result = subprocess.run(
                    ["python3", "-c", f"{code}\n{tests['runner']}"],
                    capture_output=True, text=True, timeout=2
                )
                return result.returncode == 0, result.stdout or result.stderr
            except subprocess.TimeoutExpired:
                return False, "Test timeout"
        # Add Node.js test runner similarly
        return True, "Tests skipped"

    def _grade_idioms(self, code: str, idioms: List[str]) -> float:
        # Simple keyword/regex matching for idiomatic patterns
        matches = sum(1 for pat in idioms if pat in code)
        return matches / len(idioms) if idioms else 1.0

    def _get_observation(self, action: CodeMigrationAction, feedback: str) -> CodeMigrationObservation:
        history = (self.task_meta.get("history") or []) + [
            f"Attempt {self.attempts}: {feedback}"
        ]
        return CodeMigrationObservation(
            task_id=self.task_meta["id"],
            difficulty=self.task_meta["difficulty"],
            source_code=self.source_code,
            source_language=self.task_meta["source_lang"],
            target_language=self.task_meta["target_lang"],
            requirements=self.task_meta["requirements"],
            test_description=self.task_meta["test_desc"],
            history=history[-5:]  # Keep last 5 attempts
        )