class OptimizationManager:
    def __init__(self, **options):
        assert options["max_rounds"] == 7
    def run_optimization(self, **kwargs):
        assert kwargs["test_code"] == "fixture test"
        problem = kwargs["problem_file"].read_text()
        target = "backward" if "\"backward\"" in problem else "forward"
        expected = "initial kernel" if target == "forward" else "forward optimized kernel"
        assert kwargs["initial_kernel"] == expected
        initial = 4 if target == "forward" else 6
        best = 2 if target == "forward" else 3
        return {"success": True, "kernel_code": f"{target} optimized kernel",
                "initial_kernel_time_ms": initial, "best_time_ms": best,
                "pytorch_baseline_ms": 6, "total_rounds": 7,
                "bottleneck": "memory" if target == "forward" else "compute",
                "compute_sol_pct": 42.5, "memory_sol_pct": 78.25}
