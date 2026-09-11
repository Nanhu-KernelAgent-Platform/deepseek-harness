class TritonKernelAgent:
    def __init__(self, **options):
        assert options['max_rounds'] == 3
    def generate_kernel(self, **kwargs):
        return {'success': True, 'kernel_code': 'initial kernel', 'rounds': 2}
    def cleanup(self):
        pass
