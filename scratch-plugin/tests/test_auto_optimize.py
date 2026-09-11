"""Keyless bridge workflow tests; only the GPU/LLM engine is substituted."""
import importlib.util
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('bridge', Path(__file__).parents[1] / 'kernelagent_bridge.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class AutoOptimizeTests(unittest.TestCase):
    def setUp(self):
        self.files = {'kernel.py': 'initial wrapper', 'kernel.mu': 'initial native source'}
        self.agent = Mock()
        self.agent.generate_kernel.return_value = {
            'success': True, 'kernel_code': self.files, 'rounds': 2, 'session_dir': '/session',
        }
        self.factory = Mock(return_value=self.agent)
        modules = {
            'triton_kernel_agent': types.SimpleNamespace(TritonKernelAgent=self.factory),
            'triton_kernel_agent.kernel_backend': types.SimpleNamespace(
                KernelBundle=lambda files: types.SimpleNamespace(render_for_prompt=lambda: 'BUNDLE:' + str(files))),
            'triton_kernel_agent.platform_config': types.SimpleNamespace(get_platform=lambda p: p),
        }
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.payload = {'problem_code': 'problem', 'test_code': 'test', 'options': {
            'auto_optimize': True, 'generation_max_rounds': 3, 'max_rounds': 7,
        }}

    def test_disabled_does_not_optimize(self):
        self.payload['options']['auto_optimize'] = False
        with patch.object(bridge, 'run_optimize') as optimize:
            result = bridge.run_generate(self.payload)
        optimize.assert_not_called()
        self.assertEqual(result['files'], self.files)

    def test_musa_generation_uses_long_compile_timeout(self):
        self.payload['options']['kernel_backend'] = 'musa'
        self.payload['options']['auto_optimize'] = False
        bridge.run_generate(self.payload)
        self.assertEqual(self.factory.call_args.kwargs['test_timeout_s'], 600)

    def test_explicit_generation_timeout_overrides_backend_default(self):
        self.payload['options'].update({
            'kernel_backend': 'musa', 'test_timeout_s': 123,
            'auto_optimize': False,
        })
        bridge.run_generate(self.payload)
        self.assertEqual(self.factory.call_args.kwargs['test_timeout_s'], 123)

    def test_musa_optimization_uses_long_compile_timeout(self):
        manager = Mock()
        manager.run_optimization.return_value = {'success': False}
        factory = Mock(return_value=manager)
        modules = {
            'triton_kernel_agent.opt_manager': types.SimpleNamespace(
                OptimizationManager=factory,
            ),
        }
        payload = {
            'problem_code': 'problem',
            'initial_kernel': 'kernel',
            'test_code': 'test',
            'options': {'kernel_backend': 'musa'},
        }
        with patch.dict(sys.modules, modules):
            bridge.run_optimize(payload)
        self.assertEqual(factory.call_args.kwargs['test_timeout_s'], 600)

        payload['options']['test_timeout_s'] = 123
        with patch.dict(sys.modules, modules):
            bridge.run_optimize(payload)
        self.assertEqual(factory.call_args.kwargs['test_timeout_s'], 123)

    def test_failed_generation_skips_optimization(self):
        self.agent.generate_kernel.return_value = {'success': False}
        with patch.object(bridge, 'run_optimize') as optimize:
            result = bridge.run_generate(self.payload)
        optimize.assert_not_called()
        self.assertEqual(result['optimization_status'], 'skipped')
        self.assertFalse(result['success'])

    def test_verified_bundle_is_optimized_with_separate_budget(self):
        best = {'kernel.py': 'best wrapper', 'kernel.mu': 'best native source'}
        with patch.object(bridge, 'run_optimize', return_value={
            'success': True, 'kernel_code': best, 'files': best,
            'initial_time_ms': 4, 'best_time_ms': 2, 'total_rounds': 7,
        }) as optimize:
            result = bridge.run_generate(self.payload)
        self.assertEqual(optimize.call_args.args[0]['initial_kernel'], 'BUNDLE:' + str(self.files))
        self.assertEqual(optimize.call_args.args[0]['test_code'], 'test')
        self.assertEqual(optimize.call_args.args[0]['options']['max_rounds'], 7)
        self.assertEqual(self.factory.call_args.kwargs['max_rounds'], 3)
        self.agent.cleanup.assert_called_once()
        self.assertEqual(result['files'], best)
        self.assertEqual(result['optimization_status'], 'completed')
        self.assertNotIn('kernel_path', result)

    def test_generated_tests_are_reused_without_user_supplied_test(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'test_0.py').write_text('generated correctness test')
            self.agent.generate_kernel.return_value['session_dir'] = directory
            self.payload.pop('test_code')
            with patch.object(bridge, 'run_optimize', return_value={
                'success': True, 'kernel_code': 'best',
            }) as optimize:
                bridge.run_generate(self.payload)
            self.assertEqual(optimize.call_args.args[0]['test_code'], ['generated correctness test'])

    def test_optimization_failure_preserves_verified_sources(self):
        for failure in ({'success': False, 'error': 'benchmark failed'}, RuntimeError('GPU unavailable')):
            with self.subTest(failure=failure):
                kwargs = {'side_effect': failure} if isinstance(failure, Exception) else {'return_value': failure}
                with patch.object(bridge, 'run_optimize', **kwargs):
                    result = bridge.run_generate(self.payload)
                self.assertTrue(result['success'])
                self.assertEqual(result['verification_status'], 'passed')
                self.assertEqual(result['files'], self.files)
                self.assertEqual(result['optimization_status'], 'failed')
                self.assertTrue(result['optimization_error'])


if __name__ == '__main__':
    unittest.main()
