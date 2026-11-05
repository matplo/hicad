"""Load the single shared physics implementation without depending on cwd."""
import importlib.util
from pathlib import Path
import sys

_path = Path(__file__).resolve().parents[2] / "boltzmann" / "clumps.py"
_name = "hicad_shared_clumps"
if _name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _module
    _spec.loader.exec_module(_module)
core = sys.modules[_name]
