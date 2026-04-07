# inference.py (ROOT DIRECTORY - MUST MATCH EXACT FORMAT)
import os
import asyncio
from openai import OpenAI
from typing import List
from dotenv import load_dotenv

# Load local .env file if it exists
load_dotenv()

# Required env vars
def get_env_var(name: str):
    value = os.environ.get(name)
    if not value:
        print(f"ERROR: Mandatory environment variable '{name}' is not set.", flush=True)
        import sys
        sys.exit(1)
    return value

API_BASE_URL = get_env_var("API_BASE_URL")
MODEL_NAME = get_env_var("MODEL_NAME")
HF_TOKEN = get_env_var("HF_TOKEN")
API_KEY = HF_TOKEN

TASK_NAME = os.environ.get("TASK_NAME", "python_modernize")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "huggingface/spaces/razak123/code-migration-env")
MAX_STEPS = 3
MAX_TOTAL_REWARD = 1.0  # Normalized max possible reward
SUCCESS_SCORE_THRESHOLD = 0.5

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    error_str = f" error={error}" if error else ""
    print(f"[STEP] step={step} action={action!r} reward={reward:.3f} done={done}{error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] success={success} steps={steps} score={score:.3f} rewards={rewards}", flush=True)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Import auto-generated client
    from client import CodeMigrationEnv
    from models import CodeMigrationAction
    
    env = await CodeMigrationEnv.from_docker_image(IMAGE_NAME)
    
    log_start(task=TASK_NAME, env=IMAGE_NAME, model=MODEL_NAME)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        result = await env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
            
            # Build prompt from observation
            obs = result.observation
            prompt = f"""You are a code migration expert. Migrate the following {obs.source_language} code to {obs.target_language}.

Requirements:
{obs.requirements}

Source code:
```{obs.source_language}
{obs.source_code}
```
Output ONLY valid JSON with keys: translated_code, explanation. Do not include markdown or extra text.
"""

            # Call LLM
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Respond ONLY with valid JSON matching the MigrationAction schema."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1024
                )
                response_text = (completion.choices[0].message.content or "").strip()
            except Exception as e:
                response_text = '{"translated_code": "", "explanation": "API error"}'
                print(f"[DEBUG] LLM call failed: {e}", flush=True)
            
            # Parse action
            try:
                action = CodeMigrationAction.model_validate_json(response_text)
            except Exception as e:
                action = CodeMigrationAction(translated_code="error", explanation=f"Parse error: {e}")
            
            # Step environment
            result = await env.step(action)
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            
            log_step(
                step=step,
                action=action.translated_code[:100] + ("..." if len(action.translated_code) > 100 else ""),
                reward=reward,
                done=result.done,
                error=getattr(result, 'info', {}).get("error")
            )
            
            if result.done:
                break
        
        # Compute final score
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
