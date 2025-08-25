from pathlib import Path
import os 

BASE_DIR = Path(__file__).parent

agents_config_path = os.path.join(BASE_DIR,"configs","agents.yaml")
tasks_config_path = os.path.join(BASE_DIR,"configs","tasks.yaml")

print(f"FOUNDS THE AGENTS PATH: {agents_config_path}")
print(f"FOUNDS THE tasks PATH: {tasks_config_path}")