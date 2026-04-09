
---
title: Code Migration Environment
emoji: 🔄
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "latest"
pinned: false
app_port: 8000
tags:
  - openenv
---

# Code Migration Environment

An OpenEnv environment where AI agents migrate real-world code across 
languages and frameworks. Models must produce syntactically correct, 
idiomatic, functionally equivalent code.

## Motivation
Code migration is a high-frequency engineering task — modernizing legacy 
Python, porting backends across languages, adopting faster libraries. 
This environment provides deterministic, reproducible grading for agent 
evaluation on these tasks.

## Action Space
| Field | Type | Description |
|-------|------|-------------|
| translated_code | str | Migrated code in target language |
| explanation | str | Rationale for key changes (max 2000 chars) |

## Observation Space
| Field | Type | Description |
|-------|------|-------------|
| task_id | str | Task identifier |
| difficulty | str | easy / medium / hard |
| source_code | str | Code to migrate |
| source_language | str | Source language |
| target_language | str | Target language |
| requirements | str | Migration requirements |
| test_description | str | What tests validate |
| history | list[str] | Prior attempt feedback |

## Tasks
| Task | Difficulty | Description |
|------|-----------|-------------|
| python_modernize | Easy | Python 3.8 → 3.12 idioms (match/case, pathlib, type hints) |
| python_to_node | Medium | Python requests → Node.js async/await fetch |
| pandas_to_polars | Hard | pandas pipeline → Polars LazyFrame with window expressions |

## Reward Design
- +0.3: Syntax valid
- +0.5: Passes functional tests
- +0.0–0.4: Idiomatic pattern score (weighted keyword matching)
- −0.1×attempts: Penalty for syntax errors
- Clamped to [0.0, 1.0]

## Baseline Scores (mistralai/devstral-2-123b)
| Task | Score |
|------|-------|
| python_modernize | 1.000 |
| python_to_node | 0.620 |
| pandas_to_polars | 0.700 |

## Setup
```bash
docker build -t code-migration-env .
docker run -p 8000:8000 code-migration-env

# Run baseline
export API_BASE_URL=https://integrate.api.nvidia.com/v1
export MODEL_NAME=mistralai/devstral-2-123b-instruct-2512
export OPENAI_API_KEY=your_key
python inference.py
```