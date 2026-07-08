import os
import yaml

base_dir = "/home/minjae/research/contractionRL/source/contractionRL/contractionRL/tasks/direct"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".yaml"):
            path = os.path.join(root, file)
            parts = path.split(os.sep)
            try:
                agents_idx = parts.index("agents")
                env_name = parts[agents_idx - 1]
            except ValueError:
                continue

            with open(path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if line.strip().startswith("directory:"):
                    val = line.split(":", 1)[1].strip().strip('"\'')
                    print(f"[{env_name}] -> {val}")
