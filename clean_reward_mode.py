import os
import glob
import re

base_path = "/home/minjae/research/contractionRL/source/contractionRL/contractionRL/tasks/direct/classic"
env_files = glob.glob(f"{base_path}/*/env.py")

for f in env_files:
    with open(f, 'r') as file:
        content = file.read()
    
    # Remove from __init__ signature
    content = re.sub(r'\s*reward_mode:\s*str\s*=\s*"default",\n?', '\n', content)
    # Remove from super().__init__ or self._build_cfg
    content = re.sub(r'reward_mode=reward_mode,\s*', '', content)
    
    with open(f, 'w') as file:
        file.write(content)

env_base_path = f"{base_path}/common/env_base.py"
with open(env_base_path, 'r') as file:
    content = file.read()

# Remove from __init__ signature
content = re.sub(r'\s*reward_mode:\s*str\s*=\s*"default",\n?', '\n', content)
# Remove docstring mention
content = re.sub(r'/reward_mode', '', content)
# Remove cfg["reward_mode"] = reward_mode
content = re.sub(r'\s*cfg\["reward_mode"\] = reward_mode\n?', '\n', content)
# Remove self.reward_mode = ...
content = re.sub(r'\s*self\.reward_mode = env_config\.get\("reward_mode", "default"\)\n?', '\n', content)

# Remove the block in get_rewards
# It looks like:
#         if self.reward_mode == "inverse":
#             tracking_reward = 1 / (1 + abs(tracking_reward))
#             control_reward = 1 / (1 + abs(control_reward))
content = re.sub(r'\s*if self\.reward_mode == "inverse":\n\s*tracking_reward = 1 / \(1 \+ abs\(tracking_reward\)\)\n\s*control_reward = 1 / \(1 \+ abs\(control_reward\)\)\n?', '\n', content)

# And the shaping inverse mode check
#             if self.reward_mode == "inverse":
#                 prev_tracking_reward = 1 / (1 + abs(prev_tracking_reward))
content = re.sub(r'\s*if self\.reward_mode == "inverse":\n\s*prev_tracking_reward = 1 / \(1 \+ abs\(prev_tracking_reward\)\)\n?', '\n', content)

with open(env_base_path, 'w') as file:
    file.write(content)

print("Done cleaning reward_mode")
