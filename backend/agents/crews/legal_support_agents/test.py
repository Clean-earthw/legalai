from pathlib import Path
import os

BASE_DIR = Path(__file__).parent

agents_config_path = os.path.join(BASE_DIR, "configs", "agents.yaml")
tasks_config_path = os.path.join(BASE_DIR, "configs", "tasks.yaml")

print("Your agent dir:",agents_config_path)