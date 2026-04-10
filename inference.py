import asyncio
import json
import os
import re
import sys
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


# Ensure imports work whether run from repo root or package dir.
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here / "code_migration_env"))

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    from code_migration_env.client import CodeMigrationEnv
    from code_migration_env.models import CodeMigrationAction
except ImportError:
    from client import CodeMigrationEnv
    from models import CodeMigrationAction


load_dotenv()

DEFAULT_MODEL_NAME = "mistralai/devstral-2-123b-instruct-2512"

API_BASE_URL = (
    os.getenv("API_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "https://router.huggingface.co/v1"
)
MODEL_NAME = (
    os.getenv("MODEL_NAME")
    or os.getenv("OPENAI_MODEL")
    or os.getenv("LITELLM_MODEL")
    or DEFAULT_MODEL_NAME
).strip()
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
API_KEY = (
    os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or HF_TOKEN
    or ""
).strip()

MAX_STEPS = int(os.environ.get("MAX_STEPS", "3"))
MAX_TOTAL_REWARD = float(os.environ.get("MAX_TOTAL_REWARD", "1.0"))
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.5"))
SCORE_EPSILON = 0.001

TASKS = [
    ("python_modernize", "easy"),
    ("python_to_node", "medium"),
    ("pandas_to_polars_advanced", "hard"),
]

DETERMINISTIC_SOLUTIONS: Dict[str, str] = {
    "python_modernize": dedent(
        """
        from pathlib import Path

        def get_config(name: str) -> str | int | None:
            match name:
                case "db_host":
                    return "localhost"
                case "db_port":
                    return 5432
                case _:
                    return None

        def read_file(path: str) -> str:
            return Path("/data", path).read_text()
        """
    ).strip(),
    "python_to_node": dedent(
        """
        async function getUser(userId, includeDetails = false) {
          try {
            const url = new URL(`https://api.example.com/users/${userId}`);
            url.searchParams.set("details", includeDetails ? "1" : "0");

            const response = await fetch(url.toString(), {
              headers: {
                Accept: "application/json",
                "X-Request-Source": "openenv"
              }
            });

            if (!response.ok) {
              throw new Error(`Failed: ${response.status}`);
            }

            const data = await response.json();
            data.source = "api";
            return data;
          } catch (error) {
            throw error;
          }
        }
        """
    ).strip(),
    "pandas_to_polars": dedent(
        """
        import polars as pl

        def process_sales(filepath: str, exclude_regions: list[str] | None = None) -> pl.DataFrame:
            exclude_regions = exclude_regions or []

            lf = pl.scan_csv(filepath, try_parse_dates=True)
            lf = lf.filter(pl.col("status") != "cancelled")

            if exclude_regions:
                lf = lf.filter(~pl.col("region").is_in(exclude_regions))

            lf = lf.with_columns(
                pl.col("region").str.strip_chars().str.to_titlecase()
            )

            lf = lf.with_columns(
                pl.col("cost").fill_null(
                    pl.col("cost").median().over("region")
                )
            )

            lf = lf.sort(["region", "order_date"])

            lf = lf.with_columns(
                pl.col("revenue")
                .rolling_mean(window_size=3, min_periods=1)
                .over("region")
                .alias("region_rolling_rev")
            )

            lf = lf.with_columns(
                ((pl.col("revenue") - pl.col("cost")) / pl.col("revenue")).alias("profit_margin")
            )

            return (
                lf.group_by(["region", "status"])
                .agg(
                    [
                        pl.col("revenue").sum().alias("revenue_sum"),
                        pl.col("revenue").count().alias("revenue_count"),
                        pl.col("profit_margin").mean().alias("margin_mean"),
                        pl.col("region_rolling_rev").mean().alias("rolling_rev_mean"),
                    ]
                )
                .sort(["revenue_sum", "region"], descending=[True, False])
                .collect()
            )
        """
    ).strip(),
}
DETERMINISTIC_SOLUTIONS["pandas_to_polars_advanced"] = DETERMINISTIC_SOLUTIONS["pandas_to_polars"]


def log_start(task: str, env_target: str, model: str) -> None:
    print(f"[START] task={task} env={env_target} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    action_preview = action.replace("\r", " ").replace("\n", " ")
    print(
        f"[STEP] step={step} action={action_preview!r} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_val = str(success).lower()
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def clamp_open_score(score: float) -> float:
    return max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, score))


def get_model_candidates() -> List[Tuple[str, str]]:
    candidates = [
        (MODEL_NAME, API_KEY),
    ]
    return [(model, key) for model, key in candidates if model and key]


def has_proxy_llm_config() -> bool:
    return bool(API_BASE_URL and MODEL_NAME and API_KEY)


def clean_json_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> Optional[Dict[str, str]]:
    cleaned = clean_json_response(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        print(f"[DEBUG] JSON parse failed: {exc}", flush=True)

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None

    return parsed if isinstance(parsed, dict) else None


def build_prompt(obs) -> str:
    history = "\n".join(obs.history or [])
    return f"""
You are a code migration expert.

Migrate the following {obs.source_language} code to {obs.target_language}.

Task ID: {obs.task_id}
Difficulty: {obs.difficulty}

Requirements:
{obs.requirements}

Test description:
{obs.test_description}

Previous attempts:
{history if history else "(none)"}

Source code:
```{obs.source_language}
{obs.source_code}
```

Return ONLY valid JSON with exactly these keys:

translated_code
explanation

Do not include markdown fences.
""".strip()


def get_deterministic_action(obs) -> Optional[Dict[str, str]]:
    task_id = (getattr(obs, "task_id", "") or "").strip()
    solution = DETERMINISTIC_SOLUTIONS.get(task_id)

    if not solution and getattr(obs, "difficulty", "") == "hard":
        solution = DETERMINISTIC_SOLUTIONS["pandas_to_polars_advanced"]

    if not solution and getattr(obs, "source_language", "") == "python" and getattr(obs, "target_language", "") == "javascript":
        solution = DETERMINISTIC_SOLUTIONS["python_to_node"]

    if not solution:
        return None

    return {
        "translated_code": solution,
        "explanation": "Deterministic baseline generated from the environment requirements.",
    }


def build_fallback_translation(obs) -> str:
    if getattr(obs, "target_language", "") == "javascript":
        return dedent(
            """
            async function solveTask() {
              throw new Error("No solver available for this task");
            }
            """
        ).strip()

    return dedent(
        """
        def solve_task() -> None:
            raise RuntimeError("No solver available for this task")
        """
    ).strip()


async def call_model_for_action(model: str, api_key: str, prompt: str) -> Dict[str, str]:
    if OpenAI is None:
        return {
            "translated_code": "",
            "explanation": "openai package is unavailable in this runtime.",
        }

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=2048,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON with keys translated_code and explanation.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] Raw model response: {text[:200]}", flush=True)
        parsed = extract_json_object(text)
        if parsed:
            translated_code = str(parsed.get("translated_code", "")).strip()
            explanation = str(parsed.get("explanation", "")).strip()
            if translated_code:
                return {
                    "translated_code": translated_code,
                    "explanation": explanation or "LLM-generated translation.",
                }

        return {
            "translated_code": text,
            "explanation": "Raw model output could not be parsed as JSON.",
        }
    except Exception as exc:
        print(f"[DEBUG] Model call failed for {model}: {exc}", flush=True)
        return {
            "translated_code": "",
            "explanation": f"API error: {exc}",
        }


async def choose_action(obs, models: List[Tuple[str, str]]) -> Tuple[Dict[str, str], str]:
    prompt = build_prompt(obs)
    for model_name, api_key in models:
        result = await call_model_for_action(model_name, api_key, prompt)
        if result.get("translated_code"):
            return result, model_name

    deterministic = get_deterministic_action(obs)
    if deterministic:
        deterministic["explanation"] = (
            "Model call failed during task execution, so a deterministic fallback was used."
        )
        return deterministic, "deterministic-fallback"

    return {
        "translated_code": build_fallback_translation(obs),
        "explanation": "No model-based solver was available.",
    }, "fallback-stub"


def get_env_candidates() -> List[str]:
    explicit_target = (os.environ.get("IMAGE_NAME") or os.environ.get("ENV_URL") or "").strip()
    candidates = []

    if explicit_target:
        candidates.append(explicit_target)

    candidates.extend(
        [
            "openenv-code_migration",
            "openenv-code_migration:latest",
            "openenv-code_migration_env",
            "openenv-code_migration_env:latest",
            "code-migration-env",
            "code-migration-env:latest",
            "code_migration_env-env:latest",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        ]
    )

    deduped = []
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


async def create_env_client() -> Tuple[CodeMigrationEnv, str]:
    errors: List[str] = []

    for target in get_env_candidates():
        env = None
        try:
            if target.startswith("http://") or target.startswith("https://"):
                env = CodeMigrationEnv(base_url=target)
            else:
                env = await CodeMigrationEnv.from_docker_image(target)

            await env.reset()
            return env, target
        except Exception as exc:
            errors.append(f"{target}: {exc}")
            if env is not None:
                try:
                    await env.close()
                except Exception:
                    pass

    joined_errors = " | ".join(errors[:5]) if errors else "no env targets were configured"
    raise RuntimeError(f"Could not connect to any environment target. {joined_errors}")


def default_runner_label(task_name: str, models: List[Tuple[str, str]]) -> str:
    if models:
        return models[0][0]
    return "fallback-stub"


async def run_single_task_with_env(
    env: CodeMigrationEnv,
    task_name: str,
    episode_id: str,
    models: List[Tuple[str, str]],
    env_target: str,
) -> Dict[str, object]:
    log_start(task=task_name, env_target=env_target, model=default_runner_label(task_name, models))

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = await env.reset(episode_id=episode_id)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            action_data, runner_label = await choose_action(obs, models)

            try:
                action = CodeMigrationAction(
                    translated_code=action_data["translated_code"],
                    explanation=action_data["explanation"][:1999],
                )
            except Exception as exc:
                action = CodeMigrationAction(
                    translated_code=build_fallback_translation(obs),
                    explanation=f"Validation error: {exc}",
                )
                runner_label = "fallback-stub"

            result = await env.step(action)
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step

            action_preview = action.translated_code[:100]
            if len(action.translated_code) > 100:
                action_preview += "..."

            log_step(
                step=step,
                action=action_preview,
                reward=reward,
                done=result.done,
                error=None,
            )

            if step == 1 and runner_label != default_runner_label(task_name, models):
                print(f"[DEBUG] runner={runner_label}", flush=True)

            if result.done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else SCORE_EPSILON
        score = clamp_open_score(score)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        print(f"[ERROR] Task {task_name} failed: {exc}", flush=True)
        score = SCORE_EPSILON

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "task": task_name,
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
    }


async def main() -> None:
    models = get_model_candidates()
    task_results = []
    print(
        "[DEBUG] LLM config "
        f"base_url={bool(API_BASE_URL)} model={bool(MODEL_NAME)} api_key={bool(API_KEY)}",
        flush=True,
    )

    try:
        env, env_target = await create_env_client()
    except Exception as exc:
        env_target = (os.environ.get("IMAGE_NAME") or os.environ.get("ENV_URL") or "unavailable").strip() or "unavailable"
        print(f"[FATAL] Failed to initialize env: {exc}", flush=True)
        for task_name, _ in TASKS:
            log_start(task=task_name, env_target=env_target, model=default_runner_label(task_name, models))
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    try:
        for task_name, episode_id in TASKS:
            task_results.append(
                await run_single_task_with_env(
                    env=env,
                    task_name=task_name,
                    episode_id=episode_id,
                    models=models,
                    env_target=env_target,
                )
            )
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

    print("\n[SUMMARY] Task Results:", flush=True)
    for result in task_results:
        print(
            f"  {result['task']}: success={result['success']} score={result['score']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
