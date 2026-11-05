#!/usr/bin/env python3
"""New additive clump explorer, leaving the legacy display on port 8765 intact."""
import argparse
import errno
from pathlib import Path
import webbrowser
from clump_explorer.api import make_server
from clump_explorer.background import BackgroundProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8766, type=int, help="Listening port (default: 8766); use 0 to choose a free port")
    parser.add_argument("--data-dir", action="append", help="Allowed HDF5 file or directory; repeatable")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--benchmark", type=int, metavar="SAMPLES", help="Measure uncached local generation/serialization and exit")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]; use 0 to choose a free port")
    boltzmann_dir = Path(__file__).resolve().parents[1] / "boltzmann"
    defaults = [boltzmann_dir / "canonical-output", boltzmann_dir / "output"]
    provider = BackgroundProvider(args.data_dir if args.data_dir is not None else defaults)
    if args.benchmark is not None:
        if not 1 <= args.benchmark <= 1000:
            parser.error("--benchmark samples must be in [1,1000]")
        import json
        from clump_explorer.benchmark import benchmark
        print(json.dumps(benchmark(provider,args.benchmark),indent=2))
        return
    try:
        server = make_server(args.host, args.port, provider)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.error(f"{args.host}:{args.port} is already in use. "
                         "Choose another --port, or use --port 0 to choose a free port automatically.")
        raise
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Clump explorer: {url} ({len(provider.files)} HDF5 datasets)", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
