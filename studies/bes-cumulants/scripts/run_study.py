#!/usr/bin/env python3
"""Run/restart the offline BES study and generate its standalone results browser."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bes_cumulants.study import analyze,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pilot.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/pilot-100k')
    parser.add_argument('--events',type=int,help='Override selected events; use a separate output directory')
    parser.add_argument('--strength-only',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.events is not None:config['selected_events']=args.events
    if args.strength_only:config.pop('shape',None);config.pop('clump_scan',None);config.pop('joint',None)
    result=analyze(config,args.output)
    from bes_cumulants.validation import validate_results
    result['validation']=validate_results(result,args.output)
    write_json(args.output/'validation.json',result['validation'])
    write_json(args.output/'results.json',result)
    from bes_cumulants.browser import build_browser
    build_browser(result,args.output/'index.html')
    from bes_cumulants.plots import export_figures
    export_figures(result,args.output/'figures')
    print(f"Results: {args.output/'index.html'}",flush=True)


if __name__=='__main__':main()
