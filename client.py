# code_migration_env\client.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Code Migration Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import CodeMigrationAction, CodeMigrationObservation
except ImportError:
    from models import CodeMigrationAction, CodeMigrationObservation


class CodeMigrationEnv(
    EnvClient[CodeMigrationAction, CodeMigrationObservation, State]
):
    """
    Client for the Code Migration Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CodeMigrationEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(CodeMigrationAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CodeMigrationEnv.from_docker_image("code_migration_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CodeMigrationAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CodeMigrationAction) -> Dict:
        """
        Convert CodeMigrationAction to JSON payload for step message.

        Args:
            action: CodeMigrationAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "translated_code": action.translated_code,
            "explanation": action.explanation,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CodeMigrationObservation]:
        """
        Parse server response into StepResult[CodeMigrationObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CodeMigrationObservation
        """
        obs_data = payload.get("observation", {})
        observation = CodeMigrationObservation(
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", "medium"),
            source_code=obs_data.get("source_code", ""),
            source_language=obs_data.get("source_language", ""),
            target_language=obs_data.get("target_language", ""),
            requirements=obs_data.get("requirements", ""),
            test_description=obs_data.get("test_description", ""),
            history=obs_data.get("history", []),
            info=obs_data.get("info", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
