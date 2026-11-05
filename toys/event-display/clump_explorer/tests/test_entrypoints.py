"""Smoke tests for coexistence and launching outside the checkout."""
import http.client
from http.server import ThreadingHTTPServer
import importlib.util
import re
import selectors
import subprocess
import sys
import tempfile
import threading
import unittest
from support import DISPLAY
from clump_explorer.api import make_server


class EntrypointTests(unittest.TestCase):
    def test_occupied_port_has_actionable_error_without_traceback(self):
        occupied=make_server(port=0)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result=subprocess.run([sys.executable,str(DISPLAY/'serve_clump_events.py'),
                                       '--port',str(occupied.server_port),'--data-dir',tmp],
                                      capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,2)
            self.assertIn('already in use',result.stderr)
            self.assertIn('--port 0',result.stderr)
            self.assertNotIn('Traceback',result.stderr)
        finally:
            occupied.server_close()

    def test_new_cli_from_another_directory_without_background_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc=subprocess.Popen([sys.executable,'-u',str(DISPLAY/'serve_clump_events.py'),
                                   '--port','0','--data-dir',tmp],cwd=tmp,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(proc.stdout,selectors.EVENT_READ)
                    self.assertTrue(selector.select(5),'No startup message')
                    line=proc.stdout.readline()
                match=re.search(r'http://127\.0\.0\.1:(\d+)/',line)
                self.assertIsNotNone(match,line)
                conn=http.client.HTTPConnection('127.0.0.1',int(match[1]),timeout=3)
                conn.request('GET','/');response=conn.getresponse()
                self.assertEqual(response.status,200)
                self.assertIn(b'Clump momentum lab',response.read());conn.close()
                self.assertIsNone(proc.poll())
            finally:
                proc.terminate();proc.wait(timeout=5);proc.stdout.close()

    def test_legacy_and_new_handlers_coexist(self):
        spec=importlib.util.spec_from_file_location('hicad_legacy_display_test',DISPLAY/'serve_h5_events.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        catalog=module.H5DatasetCatalog([])
        legacy=ThreadingHTTPServer(('127.0.0.1',0),module.make_handler(catalog,str(DISPLAY/'h5_event_display.html')))
        modern=make_server(port=0)
        threads=[threading.Thread(target=s.serve_forever,daemon=True) for s in [legacy,modern]]
        for t in threads:t.start()
        try:
            for server,expected in [(legacy,b'WebGPU'),(modern,b'Clump momentum lab')]:
                conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)
                conn.request('GET','/');response=conn.getresponse();self.assertEqual(response.status,200)
                self.assertIn(expected,response.read());conn.close()
        finally:
            for server in [legacy,modern]:server.shutdown();server.server_close()
            for thread in threads:thread.join()
