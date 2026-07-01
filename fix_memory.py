import os
import yaml

base_dir = "source/contractionRL/contractionRL/tasks/direct"
tasks = ["quadruped_path_tracking", "humanoid_path_tracking", "manipulator_path_tracking"]

for t in tasks:
    agents_dir = os.path.join(base_dir, t, "agents")
    if not os.path.exists(agents_dir): continue
    
    for fname in os.listdir(agents_dir):
        if fname.endswith(".yaml"):
            fpath = os.path.join(agents_dir, fname)
            with open(fpath, "r") as f:
                content = f.read()
            
            # Check if memory is missing
            if "\nmemory:" not in content and "\nmemory: " not in content and "\nmemory\n" not in content:
                # Add memory block before agent:
                memory_block = "\nmemory:\n  class: RandomMemory\n  memory_size: 4096\n\n"
                content = content.replace("\nagent:\n", memory_block + "agent:\n")
                
                with open(fpath, "w") as f:
                    f.write(content)
                print(f"Fixed memory in {fpath}")
