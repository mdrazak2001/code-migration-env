# code_migration_env\inference.py

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import re
import json
import asyncio
from typing import List, Optional, Dict, Tuple

from dotenv import load_dotenv
from openai import OpenAI

try:
    from code_migration_env.client import CodeMigrationEnv
    from code_migration_env.models import CodeMigrationAction
except ImportError:
    from client import CodeMigrationEnv
    from models import CodeMigrationAction

load_dotenv()

def get_env_var(name: str, required: bool = True, default: Optional[str] = None) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        print(f"ERROR: Mandatory environment variable '{name}' is not set.", flush=True)
        raise SystemExit(1)
    return value or ""


# Spec-required variable names
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

NVIDIA_BASE_URL = API_BASE_URL  # keep your existing logic working

TASK_NAME = os.environ.get("TASK_NAME", "python_modernize")
IMAGE_NAME = "code-migration-env"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "3"))
MAX_TOTAL_REWARD = float(os.environ.get("MAX_TOTAL_REWARD", "1.0"))
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.5"))

TASKS = [
    ("python_modernize", "easy"),
    ("python_to_node",   "medium"),
    ("pandas_to_polars", "hard"),
]

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    # ✅ Format: reward to 2 decimals, done as lowercase bool, error always present
    done_val = str(done).lower()  # true/false
    error_val = error if error else "null"  # always include field
    print(
        f"[STEP] step={step} action={action!r} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    # ✅ Format: success as lowercase bool, score to 2-3 decimals, rewards comma-separated with 2 decimals
    success_val = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

def get_nvidia_model_candidates() -> List[Tuple[str, str]]:
    """
    Returns (model_name, api_key) pairs in preference order.
    Only non-empty entries are included.
    """
    candidates = [
        # Spec-required primary (judges will set these)
        (MODEL_NAME, API_KEY),

        # fallbacks
        # (os.environ.get("NVIDIA_MODEL_DEVSTRAL", ""), os.environ.get("NVIDIA_KEY_DEVSTRAL", "")),
        # (os.environ.get("NVIDIA_MODEL_STEP_FLASH", ""), os.environ.get("NVIDIA_KEY_STEP_FLASH", "")),
        # (os.environ.get("NVIDIA_MODEL_KIMI_K2", ""), os.environ.get("NVIDIA_KEY_KIMI_K2", "")),
        # (os.environ.get("NVIDIA_MODEL_MISTRAL_LARGE", ""), os.environ.get("NVIDIA_KEY_MISTRAL_LARGE", "")),
        # (os.environ.get("NVIDIA_MODEL_DEEPSEEK_V3_1", ""), os.environ.get("NVIDIA_KEY_DEEPSEEK_V3_1", "")),
        # (os.environ.get("NVIDIA_MODEL_MAGISTRAL_SMALL", ""), os.environ.get("NVIDIA_KEY_MAGISTRAL_SMALL", "")),
        # (os.environ.get("NVIDIA_MODEL_GLM47", ""), os.environ.get("NVIDIA_KEY_GLM47", "")),
        # (os.environ.get("gemma_4b_model", ""), os.environ.get("gemma_4b_key", ""))
    ]
    return [(model, key) for model, key in candidates if model and key]

def clean_json_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()

def extract_json_object(text: str) -> Optional[Dict]:
    cleaned = clean_json_response(text)
    try:
        return json.loads(cleaned)
    except Exception as e:
        print(f"[DEBUG] JSON parse failed: {e}", flush=True)  # ADD

    # fallback: try to grab the first {...} block
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None

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

async def call_model_for_action(model: str, api_key: str, prompt: str) -> Dict[str, str]:
    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=2048,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON with keys translated_code and explanation."
                },
                {"role": "user", "content": prompt},
            ],
        )

        text = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] Raw model response: {text[:200]}", flush=True)
        parsed = extract_json_object(text)
        print(f"[DEBUG] Parsed result: {parsed}", flush=True)
        if parsed and isinstance(parsed, dict):
            translated_code = str(parsed.get("translated_code", "")).strip()
            explanation = str(parsed.get("explanation", "")).strip()
            if translated_code:
                return {
                    "translated_code": translated_code,
                    "explanation": explanation or "NVIDIA-generated translation.",
                }
        
        # fallback: wrap raw output
        return {
            "translated_code": text,
            "explanation": "Raw model output could not be parsed as JSON.",
        }
    except Exception as e:
        print(f"[DEBUG] NVIDIA call failed for {model}: {e}", flush=True)
        return {
            "translated_code": "",
            "explanation": f"API error: {e}"
        }

async def run_single_task_with_env(env, task_name: str, episode_id: str, models, current_model):
    log_start(task=task_name, env=IMAGE_NAME, model=current_model)


    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # Reset with episode_id to switch scenario — no new container needed
        result = await env.reset(episode_id=episode_id)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            prompt = build_prompt(obs)

            action_data = None
            for model_name, api_key in models:
                # print("model: ", model_name)
                # print("key: ", api_key)
                res = await call_model_for_action(model_name, api_key, prompt)
                if res["translated_code"]:
                    action_data = res
                    current_model = model_name
                    break

            if not action_data:
                action_data = {"translated_code": "error", "explanation": "All models failed"}

            try:
                action = CodeMigrationAction(
                    translated_code=action_data["translated_code"],
                    explanation=action_data["explanation"][:1999]
                )
            except Exception as e:
                action = CodeMigrationAction(
                    translated_code="error",
                    explanation=f"Validation error: {e}"
                )

            result = await env.step(action)
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action.translated_code[:100] + ("..." if len(action.translated_code) > 100 else ""),
                reward=reward,
                done=result.done,
                error=None
            )

            if result.done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] Task {task_name} failed: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "task": task_name,
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards
    }


async def main() -> None:
    models = get_nvidia_model_candidates()
    if not models:
        print("ERROR: No NVIDIA model/key pairs found.", flush=True)
        raise SystemExit(1)

    current_model = models[0][0]
    task_results = []


    # ONE env for all tasks
    env = await CodeMigrationEnv.from_docker_image(IMAGE_NAME)

    try:
        for task_name, episode_id in TASKS:
            result = await run_single_task_with_env(
                env, task_name, episode_id, models, current_model
            )
            task_results.append(result)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    print("\n[SUMMARY] Task Results:", flush=True)
    for r in task_results:
        print(f"  {r['task']}: success={r['success']} score={r['score']:.3f}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())

