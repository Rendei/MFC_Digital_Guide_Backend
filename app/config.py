import json
import os

CONFIG_PATH = "config.json"


def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    api_key = config.get("api_key")
    if not api_key:
        raise ValueError("API key not found in config file")

    return config
