#!/usr/bin/env python3
"""Patch the installed `omnilearned` package to actually use MPS (Apple
Silicon GPU) instead of silently falling back to CPU.

Why this has to touch the installed site-packages file (not just a
monkeypatch from our launcher, like `_run_omnilearned.py` does for its two
bugs): `train.py`/`evaluate.py` hardcode `device = "cpu"` as a plain string
literal inside their `run()` functions whenever `torch.cuda.is_available()`
is False - there is no MPS branch at all. A string literal assigned to a
local variable inside someone else's function can't be redirected from
outside; the only way to get `device` to actually become `"mps"` there is to
change that source line.

Two things need changing, in both `train.py` and `evaluate.py`:
1. The device-selection block only ever chooses `cuda:<rank>` or `"cpu"`.
   Patched to add an `mps` branch.
2. That branch then wraps the model in `DistributedDataParallel`, but
   `torch.distributed`'s collectives (e.g. `allgather_`, used during DDP's
   startup param sync) are not implemented for the MPS device - confirmed by
   reproducing it directly (`DDP(nn.Linear(4,4).to("mps"))` under a
   single-process gloo group raises `NotImplementedError: ... 'c10d::allgather_'
   is not currently implemented for the MPS device`). Since this demo only
   ever runs with world_size=1, DDP is doing no useful distributed work
   anyway, so the MPS branch skips it and wraps the model in a tiny
   `.module`-exposing shim instead (several call sites elsewhere assume a
   DDP-wrapped model has a `.module` attribute, e.g. EMA and checkpoint
   saving in `utils.py`).

Idempotent: running this twice is a no-op the second time.
"""

import importlib.util
import sys
from pathlib import Path

DDP_BLOCK_OLD = '''    # Transfer model to GPU if available
    kwarg = {}
    if torch.cuda.is_available():
        device = local_rank
        model.to(local_rank)
        kwarg["device_ids"] = [device]
    else:
        model.cpu()
        device = "cpu"

    model = DDP(
        model,
        **kwarg,
    )'''

DDP_BLOCK_NEW = '''    # Transfer model to GPU if available
    kwarg = {}
    if torch.cuda.is_available():
        device = local_rank
        model.to(local_rank)
        kwarg["device_ids"] = [device]
        model = DDP(model, **kwarg)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # torch.distributed collectives (e.g. allgather_) aren't implemented
        # for MPS, and this only ever runs single-process anyway, so skip
        # DDP and keep a `.module` attribute for code that expects one.
        device = "mps"
        model.to(device)

        class _SingleProcessWrapper(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        model = _SingleProcessWrapper(model)
    else:
        model.cpu()
        device = "cpu"
        model = DDP(model, **kwarg)'''

PRINT_LINE_OLD = '        d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")'
PRINT_LINE_NEW = (
    '        d = torch.device("cuda:0" if torch.cuda.is_available() '
    'else ("mps" if getattr(torch.backends, "mps", None) is not None '
    'and torch.backends.mps.is_available() else "cpu"))'
)


def patch_file(path: Path) -> bool:
    text = path.read_text()
    changed = False

    if DDP_BLOCK_NEW in text:
        pass  # already patched
    elif DDP_BLOCK_OLD in text:
        text = text.replace(DDP_BLOCK_OLD, DDP_BLOCK_NEW)
        changed = True
    else:
        raise RuntimeError(
            f"{path}: expected DDP device-selection block not found - "
            "omnilearned's source has likely changed, re-check before patching"
        )

    if PRINT_LINE_NEW in text:
        pass
    elif PRINT_LINE_OLD in text:
        text = text.replace(PRINT_LINE_OLD, PRINT_LINE_NEW)
        changed = True
    else:
        raise RuntimeError(f"{path}: expected device-print line not found")

    if changed:
        path.write_text(text)
    return changed


def main():
    spec = importlib.util.find_spec("omnilearned")
    if spec is None or spec.origin is None:
        print("omnilearned is not installed", file=sys.stderr)
        sys.exit(1)
    pkg_dir = Path(spec.origin).parent

    any_changed = False
    for name in ("train.py", "evaluate.py"):
        changed = patch_file(pkg_dir / name)
        print(f"{name}: {'patched' if changed else 'already up to date'}")
        any_changed = any_changed or changed

    if not any_changed:
        print("Nothing to do, omnilearned already supports MPS.")


if __name__ == "__main__":
    main()
