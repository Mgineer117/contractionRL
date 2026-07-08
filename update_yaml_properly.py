import yaml
import glob
import os

files = glob.glob('source/contractionRL/contractionRL/tasks/direct/**/skrl_ppo_cfg.yaml', recursive=True)

for file_path in files:
    with open(file_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('clip_actions:'):
            if 'True' in line:
                lines[i] = line.replace('True', 'false')
            elif 'true' in line:
                lines[i] = line.replace('true', 'false')
        if line.strip().startswith('entropy_loss_scale:'):
            parts = line.split(':')
            lines[i] = f"{parts[0]}: 0.01"
            
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))
        
print("Updated all yamls.")
