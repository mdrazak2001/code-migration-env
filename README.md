---
title: Code Migration Environment
emoji: "🔁"
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "latest"
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - code-migration
  - benchmarking
---

# Code Migration Environment

`code_migration_env` is an OpenEnv benchmark for a real engineering workflow:
migrating production code while preserving behavior, modernizing idioms, and
meeting operational constraints.

Instead of toy puzzles, the environment evaluates the kind of work platform and
application engineers actually do:

- modernizing old Python utilities before a runtime upgrade
- porting an API integration from Python to Node.js
- rewriting a memory-sensitive analytics pipeline from pandas to Polars

The benchmark is designed to reward more than surface-level syntax changes.
Agents must preserve function behavior, satisfy deterministic tests, and adopt
the target ecosystem's idioms.

## Why This Is Useful

Code migration is a frequent and expensive engineering task. Teams regularly:

- update shared utilities during Python version upgrades
- move service clients across languages during platform consolidation
- replace slow or memory-heavy data pipelines with faster libraries

This environment packages those workflows into reproducible tasks with
programmatic grading so model quality can be compared reliably.

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `translated_code` | `str` | The migrated implementation to evaluate |
| `explanation` | `str` | Brief reasoning for the migration approach |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Unique task identifier |
| `difficulty` | `str` | `easy`, `medium`, or `hard` |
| `source_code` | `str` | The original code snippet to migrate |
| `source_language` | `str` | Language of the source code |
| `target_language` | `str` | Required target language |
| `requirements` | `str` | High-level migration requirements |
| `test_description` | `str` | Summary of what the grader validates |
| `history` | `list[str]` | Feedback from prior attempts |
| `info` | `dict[str, Any]` | Business context, acceptance checks, pitfalls, and runtime constraints |

## Tasks

| Task | Difficulty | Real-world framing |
|------|------------|--------------------|
| `python_modernize` | Easy | Modernize a shared Python helper used in deployment and maintenance scripts |
| `python_to_node` | Medium | Port a production API client from Python to a Node.js edge/service context |
| `pandas_to_polars_advanced` | Hard | Rewrite a memory-sensitive analytics transformation from pandas to Polars lazy execution |

Each task includes:

- a source implementation
- a concrete migration objective
- hidden deterministic checks
- idiom-sensitive grading
- scenario-specific context such as stakeholder constraints and pitfalls

## Reward Design

The reward function gives partial credit instead of only binary success.

- `+0.3` for valid target-language syntax
- `+0.5` for passing functional tests
- `+0.0` to `+0.4` for target-idiom alignment
- penalties for clearly bad outcomes such as invalid syntax

To satisfy OpenEnv validation and downstream evaluation rules, task rewards are
clamped to the open interval `(0, 1)`.

## Baseline Behavior

The included [`inference.py`](/C:/Users/moham/Projects/rl_env/code_migration_env/inference.py)
uses the OpenAI client and reads model settings from environment variables. A
typical baseline run with `mistralai/devstral-2-123b-instruct-2512` produces
scores in roughly this range:

| Task | Typical score |
|------|---------------|
| `python_modernize` | `~0.99` |
| `python_to_node` | `~0.62` |
| `pandas_to_polars_advanced` | `~0.70` |

These values can vary slightly with model behavior and proxy settings, but they
should stay within the valid OpenEnv score range.

## Setup

Build and run the environment locally:

```bash
docker build -t code-migration-env .
docker run -p 8000:8000 code-migration-env
```

Run the baseline agent:

```bash
export API_BASE_URL=https://your-litellm-proxy.example/v1
export MODEL_NAME=mistralai/devstral-2-123b-instruct-2512
export API_KEY=your_proxy_api_key
export HF_TOKEN=your_hf_token_if_needed
python inference.py
```

Validate the environment structure:

```bash
openenv validate
```

## Resource Expectations

The environment is designed to run comfortably on modest infrastructure:

- CPU-only execution
- no local LLM weights loaded into the container
- small deterministic test fixtures
- suitable for runners around `2 vCPU / 8 GB RAM`

Most runtime cost comes from remote LLM latency during baseline inference, not
from the FastAPI environment itself.

## Repository Structure

```text
.
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── scenarios/
│   ├── easy/
│   ├── medium/
│   └── hard/
└── server/
    ├── app.py
    └── code_migration_env_environment.py
```

## Notes For Reviewers

Two design choices are intentional:

- The benchmark stays deterministic and lightweight enough for automated
  evaluation.
- The task observations include operational context so agents are rewarded for
  migrations that feel production-aware, not just syntactically valid.
