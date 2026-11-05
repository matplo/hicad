"""Build a self-contained offline HTML browser with embedded, finite result data."""
import json
from pathlib import Path


def build_browser(results,path):
    template=(Path(__file__).parent/'browser.html').read_text()
    encoded=json.dumps(results,separators=(',',':'),allow_nan=False).replace('<',r'\u003c')
    Path(path).write_text(template.replace('__RESULTS_JSON__',encoded))
