import json
from pathlib import Path
import sys

DISPLAY = Path(__file__).resolve().parents[2]
REPO = DISPLAY.parents[1]
if str(DISPLAY) not in sys.path:
    sys.path.insert(0, str(DISPLAY))


def request(include_controls=True):
    """Exercise all supported controls in tests; the shipped UI defaults to shared only."""
    spec=json.loads((DISPLAY / "clump_explorer/examples/source_only.json").read_text())
    if include_controls:
        spec['clump_config']['variants']=['shared_boost','azimuth_decohered','event_mixed']
    return spec
