"""One-time milestone baseline audit, independent of git/untracked state."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[3]
    baseline=json.loads(Path(__file__).with_name('implementation_baseline.json').read_text())
    changed=[]
    for relative,expected in baseline.items():
        path=root/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            changed.append(relative)
    if changed:
        raise SystemExit('Pre-existing files differ from implementation-start baseline: '+', '.join(changed))
    print(f'All {len(baseline)} pre-existing files match the implementation-start SHA256 baseline.')


if __name__=='__main__':
    main()
