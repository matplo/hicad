"""Bounded local HTTP API. Physics and composition live outside the handler."""
from collections import OrderedDict
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit, parse_qs

from .background import BackgroundProvider
from .contracts import ScenarioSpec, capabilities, MODES, COMPOSITIONS, BINNING
from .generator_adapter import core
from .scenario import generate_scenario, scenario_id
from . import diagnostics, exports

ROOT = Path(__file__).resolve().parent
STATIC = {"/": (ROOT.parent / "clump_event_display.html", "text/html"),
          "/clump_event_display.html": (ROOT.parent / "clump_event_display.html", "text/html")}
for name in ("explorer.js", "controls.js", "renderer.js", "diagnostics.js", "explorer.css", "renderer3d.js", "request_state.js", "downloads.js"):
    STATIC["/static/" + name] = (ROOT / "static" / name, "text/css" if name.endswith("css") else "text/javascript")
STATIC["/examples/source_only.json"] = (ROOT / "examples/source_only.json", "application/json")
STATIC["/examples/variable_parameters.json"] = (ROOT / "examples/variable_parameters.json", "application/json")


def memory_size(value, seen=None):
    """Account retained Python containers, including derived caches, once per object."""
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(memory_size(k, seen)+memory_size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(memory_size(v, seen) for v in value)
    return size


class Service:
    def __init__(self, provider=None, max_bundles=32, max_bytes=128*1024*1024):
        self.provider = provider if provider is not None else BackgroundProvider()
        self.cache = OrderedDict()
        self.sizes = {}
        self.lock = threading.RLock()
        self.work = threading.BoundedSemaphore(2)
        self.pairs = threading.BoundedSemaphore(1)
        self.max_bundles, self.max_bytes = max_bundles, max_bytes

    def _budget(self, sid, bundle):
        with self.lock:
            # Include separately cached observations and diagnostics in memory budget.
            self.sizes[sid] = memory_size([bundle.data, bundle.observation_cache, bundle.diagnostic_cache])
            self.cache[sid] = bundle
            self.cache.move_to_end(sid)
            while len(self.cache) > self.max_bundles or sum(self.sizes.values()) > self.max_bytes:
                old, _ = self.cache.popitem(last=False)
                self.sizes.pop(old, None)

    def get(self, sid):
        with self.lock:
            if sid not in self.cache:
                raise core.ParameterError("scenario_id", "Scenario expired or unknown; POST its saved recipe to regenerate", "expired_scenario", 410)
            bundle = self.cache[sid]
            self.cache.move_to_end(sid)
        self.provider.check_reference(bundle.data["resolved_request"]["background"])
        return bundle

    def scenario(self, request):
        started = time.perf_counter()
        spec = ScenarioSpec.parse(request)
        self.provider.check_reference(spec.request["background"])
        if not self.work.acquire(blocking=False):
            raise core.ParameterError("service", "Generation busy; retry latest draft", "busy", 503)
        try:
            sid = scenario_id(spec)
            with self.lock:
                bundle = self.cache.get(sid)
            cached = bundle is not None
            if bundle is None:
                bundle = generate_scenario(spec, self.provider)
            with bundle.lock:
                result = bundle.response(spec)
                result["timing_ms"]["response_preparation"] = (time.perf_counter()-started)*1000
                result["cache_hit"] = cached
                self._budget(sid, bundle)
            return result
        finally:
            self.work.release()

    def diagnostic(self, request):
        keys = {"scenario_id", "observation_mode", "variant", "composition", "binning_id"}
        core.fields(request, keys, keys, "diagnostics")
        core.choice(request["observation_mode"], MODES, "observation_mode")
        core.choice(request["composition"], COMPOSITIONS, "composition")
        core.choice(request["binning_id"], [BINNING], "binning_id")
        if not isinstance(request["scenario_id"], str):
            core.fail("scenario_id", "Expected scenario ID string")
        bundle = self.get(request["scenario_id"])
        core.choice(request["variant"], list(bundle.data["variants"]), "variant")
        if not self.pairs.acquire(blocking=False):
            raise core.ParameterError("diagnostics", "Exact pair worker busy; retry latest view", "busy", 503)
        start = time.perf_counter()
        try:
            value = diagnostics.for_bundle(bundle, request["variant"], request["composition"], request["observation_mode"])
            with bundle.lock:
                self._budget(request["scenario_id"], bundle)
            return {**request, "diagnostics": value, "timing_ms": (time.perf_counter()-start)*1000}
        finally:
            self.pairs.release()


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HICADClump/1"

        def log_message(self, fmt, *args):
            pass

        def send(self, value, status=200, mime="application/json", filename=None):
            body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-HICAD-Core", core.CORE_VERSION)
            self.send_header("X-HICAD-RNG", core.RNG_POLICY)
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            if status == 503:
                self.send_header("Retry-After", "1")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def error(self, exc, request_id=None):
            self.send({"error": {"code": exc.code, "field": exc.field, "message": str(exc)}, "request_id": request_id}, exc.status)

        def do_GET(self):
            try:
                parsed = urlsplit(self.path)
                path = parsed.path
                if path in STATIC:
                    file, mime = STATIC[path]
                    self.send(file.read_bytes(), mime=mime)
                elif path == "/api/v1/capabilities":
                    self.send(capabilities())
                elif path == "/api/v1/backgrounds":
                    service.provider.refresh()
                    self.send({"backgrounds": service.provider.catalog(), "warnings": service.provider.warnings})
                elif path.startswith("/api/v1/scenarios/"):
                    parts = path.strip("/").split("/")
                    if len(parts) != 5 or parts[4] not in ("recipe", "snapshot"):
                        raise core.ParameterError("path", "Unknown endpoint", "not_found", 404)
                    bundle = service.get(parts[3])
                    query = parse_qs(parsed.query)
                    if any(k not in ("observation_mode", "variant") for k in query):
                        core.fail("query", "Unknown export parameter")
                    spec = deepcopy(bundle.data["resolved_request"])
                    if "observation_mode" in query:
                        spec["observation_mode"] = core.choice(query["observation_mode"][0], MODES, "observation_mode")
                    variant = query.get("variant", [next(iter(bundle.data["variants"]))])[0]
                    core.choice(variant, list(bundle.data["variants"]), "variant")
                    with bundle.lock:
                        if parts[4] == "recipe":
                            value = exports.recipe(bundle)
                            value["observation_mode"] = spec["observation_mode"]
                        else:
                            value = exports.snapshot(bundle, ScenarioSpec.parse(spec))
                            value["selected_variant"] = variant
                        service._budget(parts[3], bundle)
                    self.send(value, filename=f"clump-{parts[3]}-{parts[4]}.json")
                else:
                    raise core.ParameterError("path", "Unknown endpoint", "not_found", 404)
            except core.ParameterError as exc:
                self.error(exc)

        def do_POST(self):
            request_id = None
            try:
                origin = self.headers.get("Origin")
                if origin and origin != "http://" + self.headers.get("Host", ""):
                    raise core.ParameterError("Origin", "Same-origin requests required", "invalid_origin", 400)
                if self.headers.get_content_type() != "application/json":
                    raise core.ParameterError("Content-Type", "Declare application/json", "malformed_json", 400)
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    raise core.ParameterError("Content-Length", "Required integer body length", "malformed_json", 400)
                if length < 0 or length > 65536:
                    core.fail("body", "Maximum JSON request body is 65536 bytes")
                self.connection.settimeout(10)
                try:
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise ValueError("Incomplete JSON body")
                    def invalid_constant(value):
                        raise ValueError("Nonfinite JSON number: " + value)
                    request = json.loads(body, parse_constant=invalid_constant)
                except (ValueError, UnicodeDecodeError, TimeoutError) as exc:
                    raise core.ParameterError("body", str(exc), "malformed_json", 400)
                if isinstance(request, dict):
                    request_id = request.get("request_id")
                    if not isinstance(request_id, int) or isinstance(request_id, bool):
                        request_id = None
                routes = {"/api/v1/scenario": service.scenario, "/api/v1/background/select": service.provider.select,
                          "/api/v1/diagnostics": service.diagnostic}
                if self.path not in routes:
                    raise core.ParameterError("path", "Unknown endpoint", "not_found", 404)
                self.send(routes[self.path](request))
            except core.ParameterError as exc:
                self.error(exc, request_id)
            except (OSError, OverflowError) as exc:
                self.error(core.ParameterError("background", f"Data read failed: {exc}", "data_error", 409), request_id)
    return Handler


def make_server(host="127.0.0.1", port=8766, provider=None):
    service = Service(provider)
    server = ThreadingHTTPServer((host, port), handler_for(service))
    server.daemon_threads = True
    server.service = service
    return server
