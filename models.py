# code_migration_env\models.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Code Migration Env Environment.

The code_migration_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Dict, Any


class CodeMigrationAction(Action):
    translated_code: str = Field(description="The migrated code in target language")
    explanation: str = Field(max_length=2000, description="Brief rationale for key changes")
    
    @field_validator('translated_code')
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("translated_code cannot be empty")
        return v


class CodeMigrationObservation(Observation):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    source_code: str
    source_language: str
    target_language: str
    requirements: str
    test_description: str
    history: Optional[List[str]] = None
    reward: float = 0.0   # add if not in base class
    done: bool = False    # add if not in base class
    info: Dict[str, Any] = Field(default_factory=dict)
