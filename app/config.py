import json
import os

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    api_key = config.get("api_key")
    db_url = os.getenv("DATABASE_URL", config.get("database_url"))

    if not api_key:
        raise ValueError("API key not found in config file")
    if not db_url:
        raise ValueError("Database URL not found in config file or environment variables")

    return {"api_key": api_key, "database_url": db_url}
