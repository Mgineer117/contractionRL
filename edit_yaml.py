import os, glob, re

count = 0
for path in glob.glob('/home/minjae/research/contractionRL/source/contractionRL/contractionRL/tasks/**/*.yaml', recursive=True):
    with open(path, 'r') as f:
        content = f.read()
    new_content = re.sub(r'write_interval:\s*[0-9]+', 'write_interval: "auto"', content)
    new_content = re.sub(r'checkpoint_interval:\s*[0-9]+', 'checkpoint_interval: "auto"', new_content)
    if new_content != content:
        with open(path, 'w') as f:
            f.write(new_content)
        print(f'Updated {path}')
        count += 1
print(f"Updated {count} files")
