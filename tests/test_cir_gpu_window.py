import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cir_gpu_window as window
from generate_fashioniq_pair_captions import require_idle


class GPUWindowTests(unittest.TestCase):
    def setUp(self):
        base=Path(tempfile.gettempdir()) if os.name=='posix' else Path(__file__).resolve().parents[1]/'.autodl/test_tmp'
        base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='fashioncir-window-test-',dir=base)
        self.root=Path(self.temp.name).resolve()
        assert self.root.parent==base.resolve() and self.root.name.startswith('fashioncir-window-test-')
        self.addCleanup(self.temp.cleanup)
        (self.root/'switch_to_online_20261016').mkdir()
        (self.root/'queue.pid').write_text('42')
        (self.root/'switch_to_online_20261016/online_start_snapshot.json').write_text(json.dumps({'code_root':'/code/shoes'}))
        self.report={'status':'qwen_train','config':{'generation_provider':'online-qwen37'}}
        self.put_report()
        self.queue=dict(pid=42,state='S',start_ticks='123',cwd='/code/shoes',command_sha256='queue-hash',
                        command=['python','scripts/run_cirr_shoes_benchmark.py','--dataset','shoes','--generation-provider','online-qwen37'])
        self.owner=dict(pid=99,state='R',start_ticks='456',cwd='/code/fashioniq',command_sha256='owner-hash',
                        command=['python','scripts/run_fashioniq_during_online.py'])
        self.proof=self.root/'gpu_hold.json'

    def put_report(self):(self.root/'experiment_report.json').write_text(json.dumps(self.report))

    def info(self,pid):return dict(self.queue if pid==42 else self.owner) if pid in (42,99) else None

    def put_proof(self):
        self.proof.write_text(json.dumps(dict(status='held',results=str(self.root),queue=dict(self.queue),owner=dict(self.owner))))

    def test_only_online_generation_can_acquire_window(self):
        with patch.object(window,'process_info',side_effect=self.info):
            self.assertEqual(window.check_online_queue(self.root)['pid'],42)
            for status in ('structured_off','complete','stopped_on_error'):
                self.report['status']=status;self.put_report()
                with self.assertRaises(RuntimeError):window.check_online_queue(self.root)
            self.report={'status':'qwen_train','config':{'generation_provider':'local'}};self.put_report()
            with self.assertRaises(RuntimeError):window.check_online_queue(self.root)

    def test_queue_command_and_cwd_must_match_recorded_instance(self):
        self.queue['cwd']='/unrelated'
        with patch.object(window,'process_info',side_effect=self.info):
            with self.assertRaises(RuntimeError):window.check_online_queue(self.root)

    def test_proof_requires_stopped_queue_and_live_matching_supervisor(self):
        self.put_proof()
        with patch.object(window,'process_info',side_effect=self.info):
            with self.assertRaises(RuntimeError):window.verify_hold(self.proof,self.root)
            self.queue['state']='T';self.put_proof()
            self.assertEqual(window.verify_hold(self.proof,self.root)['status'],'held')
            self.owner['state']='Z'
            with self.assertRaises(RuntimeError):window.verify_hold(self.proof,self.root)

    def test_pid_reuse_does_not_pass_identity_check(self):
        self.queue['state']='T';self.put_proof();self.queue['start_ticks']='999'
        with patch.object(window,'process_info',side_effect=self.info):
            with self.assertRaises(RuntimeError):window.verify_hold(self.proof,self.root)

    def test_default_guard_still_refuses_a_live_queue(self):
        with patch('generate_fashioniq_pair_captions.os.kill'):
            with self.assertRaises(RuntimeError):require_idle(self.root,8,self.root/'labels.json')

    def test_verified_hold_does_not_exempt_gpu_occupancy_or_low_disk(self):
        self.queue['state']='T';self.put_proof()
        with patch.object(window,'process_info',side_effect=self.info),patch('generate_fashioniq_pair_captions.subprocess.check_output',return_value='1234'):
            with self.assertRaises(RuntimeError):require_idle(self.root,8,self.root/'labels.json',self.proof)
        with patch.object(window,'process_info',side_effect=self.info),patch('generate_fashioniq_pair_captions.subprocess.check_output',return_value=''),patch('generate_fashioniq_pair_captions.shutil.disk_usage',return_value=SimpleNamespace(free=2**30)):
            with self.assertRaises(RuntimeError):require_idle(self.root,8,self.root/'labels.json',self.proof)

    @unittest.skipUnless(os.name=='posix','Linux process stop/continue coordination')
    def test_exception_releases_exact_dispatcher_and_preserves_proof(self):
        signals=[]
        def send(pid,sig):
            signals.append((pid,sig));self.queue['state']='T' if sig==signal.SIGSTOP else 'S'
        with patch.object(window,'process_info',side_effect=self.info),patch.object(window.os,'getpid',return_value=99),patch.object(window.os,'kill',side_effect=send):
            with self.assertRaisesRegex(ValueError,'model failure'):
                with window.hold_online_dispatcher(self.root,self.proof):raise ValueError('model failure')
        self.assertEqual(signals,[(42,signal.SIGSTOP),(42,signal.SIGCONT)])
        self.assertEqual(json.loads(self.proof.read_text())['status'],'released')

    @unittest.skipUnless(os.name=='posix','Linux process identity')
    def test_real_process_stamp_is_available_without_signaling(self):
        info=window.process_info(os.getpid())
        self.assertEqual(info['pid'],os.getpid());self.assertTrue(info['start_ticks'].isdigit())


if __name__=='__main__':unittest.main()
