from copy import deepcopy
import http.client
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from support import request
from clump_explorer.api import make_server,Service
from clump_explorer.generator_adapter import core


class ServiceTests(unittest.TestCase):
    def test_cache_echo_id_identity_eviction_and_busy(self):
        service=Service(max_bundles=1);r=request()
        a=service.scenario(r);r['request_id']=37
        b=service.scenario(r)
        self.assertTrue(b['cache_hit']);self.assertEqual(b['request_id'],37)
        self.assertEqual(a['scenario_id'],b['scenario_id'])
        r['observation_mode']='packed_float16';self.assertEqual(service.scenario(r)['scenario_id'],a['scenario_id'])
        r['realization']['clump_revision']=1;service.scenario(r)
        with self.assertRaises(core.ParameterError) as exc:service.get(a['scenario_id'])
        self.assertEqual(exc.exception.status,410)
        service.work.acquire();service.work.acquire()
        try:
            with self.assertRaises(core.ParameterError) as exc:service.scenario(r)
            self.assertEqual(exc.exception.status,503)
        finally:service.work.release();service.work.release()

    def test_clients_and_batch_parity(self):
        from clump_explorer.scenario import generate_scenario
        service=Service();r=request();a=service.scenario(r)
        with ThreadPoolExecutor(max_workers=2) as pool:
            values=list(pool.map(service.scenario,[r,deepcopy(r)]))
        for v in values:self.assertEqual(v['variants'],a['variants'])
        self.assertEqual(generate_scenario(r).data['variants'],a['variants'])


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=make_server(port=0)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def send(self,path,body=None,mime='application/json',raw=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=10)
        conn.request('GET' if body is None and raw is None else 'POST',path,body=raw if raw is not None else json.dumps(body) if body is not None else None,headers={'Content-Type':mime})
        response=conn.getresponse();data=response.read();status=response.status;headers=dict(response.getheaders());conn.close()
        return status,json.loads(data) if 'application/json' in headers['Content-Type'] else data,headers

    def test_routes_scenario_diagnostics_exports_and_static(self):
        status,cap,_=self.send('/api/v1/capabilities');self.assertEqual(status,200);self.assertEqual(cap['limits']['particles'],20000)
        default=self.send('/examples/source_only.json')[1]
        self.assertEqual(default['clump_config']['variants'],['shared_boost'])
        default_result=self.send('/api/v1/scenario',default)[1]
        self.assertEqual(set(default_result['variants']),{'shared_boost'})
        self.assertEqual(self.send('/api/v1/backgrounds')[1]['backgrounds'],[])
        status,b,_=self.send('/api/v1/scenario',request());self.assertEqual(status,200)
        config=dict(scenario_id=b['scenario_id'],observation_mode='native',variant='shared_boost',composition='combined',binning_id=cap['binning_id'])
        status,d,_=self.send('/api/v1/diagnostics',config);self.assertEqual(status,200)
        for k,v in config.items():self.assertEqual(d[k],v)
        mixed_config={**config,'variant':'event_mixed'}
        status,mixed,_=self.send('/api/v1/diagnostics',mixed_config)
        self.assertEqual(status,200);self.assertEqual(mixed['diagnostics']['truth_pair_count'],0)
        self.assertFalse(mixed['diagnostics']['truth_daughter_pairs']['available'])
        status,r,h=self.send(f"/api/v1/scenarios/{b['scenario_id']}/recipe")
        self.assertEqual(status,200);self.assertIn('attachment',h['Content-Disposition'])
        self.assertEqual(self.send('/api/v1/scenario',r)[1]['variants'],b['variants'])
        self.assertIn('truth',self.send(f"/api/v1/scenarios/{b['scenario_id']}/snapshot")[1]['variants']['shared_boost'])
        for path in ['/','/static/explorer.js','/static/request_state.js','/static/downloads.js','/examples/source_only.json']:
            self.assertEqual(self.send(path)[0],200)

    def test_malformed_unknown_and_validation_errors(self):
        for raw,mime in [('bad','application/json'),('{}','text/plain'),('{"request_id":NaN}','application/json')]:
            status,body,_=self.send('/api/v1/scenario',raw=raw,mime=mime);self.assertEqual(status,400);self.assertIn('error',body)
        r=request();r['clump_config']['boost_speed']['value']=1;r['request_id']=17
        status,body,_=self.send('/api/v1/scenario',r)
        self.assertEqual(status,422);self.assertEqual(body['request_id'],17)
        self.assertEqual(body['error']['field'],'clump_config.boost_speed.value')
        self.assertEqual(self.send('/api/v1/scenario',raw='x'*65537)[0],422)
        self.assertEqual(self.send('/static/../api.py')[0],404)
        self.assertEqual(self.send('/api/v1/scenarios/missing/snapshot')[0],410)
        self.assertEqual(self.send('/api/v1/scenario',[])[0],422)
