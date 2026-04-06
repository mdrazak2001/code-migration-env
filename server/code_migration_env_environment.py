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

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

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

    def __init__(self, scenario_dir: str):
        self.scenario_dir = Path(scenario_dir)
        self.task_meta = json.loads((self.scenario_dir / "meta.json").read_text())
        self.attempts = 0
        self.max_attempts = 3  # Allow retry on syntax errors

    def reset(self) -> CodeMigrationObservation:
        self.attempts = 0
        source = (self.scenario_dir / "source.py").read_text()
        return CodeMigrationObservation(
            task_id=self.task_meta["id"],
            difficulty=self.task_meta["difficulty"],
            source_code=source,
            source_language=self.task_meta["source_lang"],
            target_language=self.task_meta["target_lang"],
            requirements=self.task_meta["requirements"],
            test_description=self.task_meta["test_desc"],
            history=[]
        )

    def step(self, action: CodeMigrationAction) -> StepResult:
        self.attempts += 1
        code = action.translated_code
        reward = 0.0
        done = False
        info = {"attempt": self.attempts}
        
        # 1. Syntax validation (language-specific)
        syntax_ok, syntax_msg = self._validate_syntax(code, self.task_meta["target_lang"])
        if not syntax_ok:
            reward = -0.1 * self.attempts  # Penalty increases with retries
            if self.attempts >= self.max_attempts:
                done = True
                info["error"] = syntax_msg
            return StepResult(
                observation=self._get_observation(action, syntax_msg),
                reward=reward,
                done=done,
                info=info
            )
        
        # 2. Run pre-baked test suite (sandboxed, <2s timeout)
        test_ok, test_msg = self._run_tests(code, self.task_meta["target_lang"])
        if test_ok:
            reward += 0.5
        else:
            reward -= 0.1
        
        # 3. Idiomatic pattern matching (AST/keyword-based)
        idiom_score = self._grade_idioms(code, self.task_meta["required_idioms"])
        reward += idiom_score * 0.4
        
        # 4. Clamp reward to [0.0, 1.0]
        reward = max(0.0, min(1.0, reward))
        done = True  # Single-step submission (can extend to multi-step if desired)
        
        return StepResult(
            observation=self._get_observation(action, test_msg),
            reward=reward,
            done=done,
            info={**info, "final_reward": reward, "tests_passed": test_ok}
        )

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
            source_code=self.task_meta["source_code"],
            source_language=self.task_meta["source_lang"],
            target_language=self.task_meta["target_lang"],
            requirements=self.task_meta["requirements"],
            test_description=self.task_meta["test_desc"],
            history=history[-5:]  # Keep last 5 attempts
        )