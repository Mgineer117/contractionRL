import os
import yaml

base_dir = "/home/minjae/research/contractionRL/source/contractionRL/contractionRL/tasks/direct"

mismatches = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".yaml"):
            path = os.path.join(root, file)
            # Find the environment name from the path.
            # Example: .../tasks/direct/classic/cartpole/agents/skrl_c3m_cfg.yaml
            # Environment name is typically the parent of 'agents'
            parts = path.split(os.sep)
            try:
                agents_idx = parts.index("agents")
                env_name = parts[agents_idx - 1]
            except ValueError:
                continue

            # Read file and look for 'directory: '
            with open(path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if line.strip().startswith("directory:"):
                    val = line.split(":", 1)[1].strip().strip('"\'')
                    if env_name not in val and not (env_name == 'cartpole' and 'contractionrl' in val):
                        print(f"{path}:{i+1} env={env_name}, dir={val}")
