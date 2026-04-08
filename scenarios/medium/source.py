import requests

def get_user(user_id: int, include_details: bool = False) -> dict:
    params = {"details": "1" if include_details else "0"}
    headers = {
        "Accept": "application/json",
        "X-Request-Source": "openenv",
    }

    response = requests.get(
        f"https://api.example.com/users/{user_id}",
        params=params,
        headers=headers,
        timeout=5,
    )

    if response.status_code == 200:
        data = response.json()
        data["source"] = "api"
        return data

    raise RuntimeError(f"Failed: {response.status_code}")