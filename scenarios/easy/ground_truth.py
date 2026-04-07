# code_migration_env\scenarios\easy\ground_truth.py
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