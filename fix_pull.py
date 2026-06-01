import subprocess, os

def is_bad(path):
    return any(part.endswith(' ') for part in path.replace('\\', '/').split('/'))

result = subprocess.run(
    ['git', 'diff', '--name-status', 'HEAD', 'origin/main'],
    capture_output=True, text=True, encoding='utf-8', errors='replace'
)

changed = skipped = failed = 0

for line in result.stdout.strip().splitlines():
    if not line.strip():
        continue
    parts = line.split('\t', 1)
    status = parts[0].strip()[0]
    filepath = parts[1].strip() if len(parts) > 1 else ''

    if is_bad(filepath):
        print(f'SKIP  {filepath}')
        skipped += 1
        continue

    if status == 'D':
        if os.path.exists(filepath):
            os.remove(filepath)
        subprocess.run(['git', 'rm', '--cached', '-f', '--', filepath], capture_output=True)
        print(f'DEL   {filepath}')
        changed += 1
    else:
        r = subprocess.run(['git', 'show', f'origin/main:{filepath}'], capture_output=True)
        if r.returncode != 0:
            print(f'FAIL  {filepath}')
            failed += 1
            continue
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(r.stdout)
        subprocess.run(['git', 'add', '--', filepath], capture_output=True)
        print(f'OK    {filepath}')
        changed += 1

print(f'\ndone: {changed} applied, {skipped} skipped, {failed} failed')

merge_head = subprocess.run(
    ['git', 'rev-parse', 'origin/main'],
    capture_output=True, text=True
).stdout.strip()
with open('.git/MERGE_HEAD', 'wb') as f:
    f.write((merge_head + '\n').encode())
with open('.git/MERGE_MSG', 'w') as f:
    f.write("Merge remote-tracking branch 'origin/main'\n\nSkipped invalid Windows paths:\n  output/Before audit /\n")
print('MERGE_HEAD set — run: git commit')
