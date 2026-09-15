#!/usr/bin/env python3
"""kernelagent-injector bridge -- batch-1 (MVP).

Deploys a KernelAgent best_bundle into the managed runtime store:
  - resolves the source (example name / run dir / best_bundle dir / single file
    / serialized "FILE: ..." payload, with a baseline_bundle fallback),
  - copies the bundle files (sha256-verified, never modified),
  - builds the musa/cuda extension in isolation,
  - generates torch_integration.py (torch.ops.kernelagent::<op> registration),
  - records manifest.json with provenance + build status.

Deterministic control plane: no LLM, pure standard library.

CLI:
    kernelagent_injector_bridge.py --mode deploy --input payload.json [--output out.json]

payload (JSON): {"source": ..., "op_name": ..., "build": true, "store": ...}
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Bundle file conventions (see triton_kernel_agent/kernel_backend.py BACKENDS).
MUSA_FILES = ("kernel.py", "binding.cpp", "kernel.mu", "setup.py")
TRITON_FILES = ("kernel.py",)
BUILD_TIMEOUT_S = 600


class BridgeError(Exception):
    """User-facing error; message is shown in the harness report."""


# --------------------------------------------------------------------------- #
# Trace collector (R4)
# --------------------------------------------------------------------------- #

class TraceCollector:
    """Collects trace events during the inject flow.
    Events are emitted to stderr in real time and accumulated for L3 history.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def add(self, step: str, detail: str) -> None:
        self.events.append({"step": step, "detail": detail})
        print(f"[ka-injector] {step} {detail}", file=sys.stderr, flush=True)

    def to_list(self) -> list[dict]:
        return list(self.events)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def detect_backend(bundle_dir: Path) -> str:
    names = {p.name for p in bundle_dir.iterdir() if p.is_file()}
    if all(f in names for f in MUSA_FILES):
        return "musa"
    if "kernel.py" in names:
        return "triton"
    raise BridgeError(
        f"best_bundle contents unrecognised or incomplete: {sorted(names)}; "
        f"musa requires {list(MUSA_FILES)}, triton requires {list(TRITON_FILES)}"
    )


def _is_cuda_backend(bundle_dir: Path, meta: dict) -> bool:
    """Detect a CUDA-native extension bundle by meta.backend or setup.py CUDAExtension."""
    if meta.get("backend") == "cuda":
        return True
    setup_py = bundle_dir / "setup.py"
    if setup_py.is_file():
        src = setup_py.read_text(encoding="utf-8")
        if re.search(r"CUDAExtension", src):
            return True
    return False


def derive_schema(kernel_py: Path, op: str, meta: "dict | None" = None) -> str:
    """Derive the torch.library schema string for the kernel's forward entry.

    fix: metadata.schema wins when present, otherwise we parse
    kernel_function's AST (now supporting arbitrary argument lists, not just a
    single Tensor argument).
    """
    if meta and meta.get("schema"):
        return meta["schema"]
    try:
        tree = ast.parse(kernel_py.read_text(encoding="utf-8"))
    except SyntaxError as e:
        raise BridgeError(f"cannot parse kernel.py for schema: {e}")
    fwd = _find_func_def(tree, "kernel_function")
    if fwd is None:
        raise BridgeError(f"kernel_function not found in {kernel_py}")
    params = []
    for a in fwd.args.args:
        if a.arg in ("self", "cls"):
            continue
        params.append(f"{_annotation_to_schema(a.annotation)} {a.arg}")
    ret = _annotation_to_schema(fwd.returns)
    if not params:
        raise BridgeError(f"kernel_function has no tensor params in {kernel_py}")
    return f"{op}({', '.join(params)}) -> {ret}"


def _derive_backward_schema(kernel_py: Path, op: str, meta: "dict | None" = None) -> str:
    """Derive the torch.library schema string for a separate-style backward kernel.

    Mirrors derive_schema but targets ``kernel_backward``. The generated op name is
    ``<op>_bwd`` (e.g. ``relu_bwd``). Falls back to ``meta["backward_schema"]`` when
    the bundle ships its own (no cross-file change to the generator needed).
    """
    if meta and meta.get("backward_schema"):
        return meta["backward_schema"]
    try:
        tree = ast.parse(kernel_py.read_text(encoding="utf-8"))
    except SyntaxError as e:
        raise BridgeError(f"cannot parse kernel.py for backward schema: {e}")
    bwd = _find_func_def(tree, "kernel_backward")
    if bwd is None:
        raise BridgeError(f"kernel_backward not found in {kernel_py} for separate style")
    params = []
    for a in bwd.args.args:
        if a.arg in ("self", "cls"):
            continue
        params.append(f"{_annotation_to_schema(a.annotation)} {a.arg}")
    ret = _annotation_to_schema(bwd.returns)
    if not params:
        raise BridgeError(f"kernel_backward has no tensor params in {kernel_py}")
    return f"{op}_bwd({', '.join(params)}) -> {ret}"


def parse_ext_name(setup_py: Path) -> str | None:
    text = setup_py.read_text(encoding="utf-8")
    m = re.search(
        r"(?:MUSAExtension|CUDAExtension)\s*\(\s*name\s*=\s*[\"']([^\"']+)[\"']",
        text,
    )
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Op naming (v7: 3-level priority; target_op channel reserved for later batches)
# --------------------------------------------------------------------------- #

# Patterns to map a reference implementation (problem.py Model.forward) to the
# target operator semantic name.
PROBLEM_OP_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:torch|F)\.relu\b", "relu"),
    (r"\b(?:torch|F)\.sigmoid\b", "sigmoid"),
    (r"\b(?:torch|F)\.silu\b", "silu"),
    (r"\bnn\.ReLU\b", "relu"),
    (r"\bnn\.Sigmoid\b", "sigmoid"),
    (r"\bnn\.SiLU\b", "silu"),
    # Extend with the examples catalogue over time (rmsnorm / matvec / ...).
]

OP_ALIAS: dict[str, set[str]] = {
    "relu": {"relu", "relu_", "rectified_linear"},
    "sigmoid": {"sigmoid", "sigmoid_", "logistic"},
    "silu": {"silu", "swish"},
}


def derive_op_from_problem(problem_py: Path, trace: TraceCollector | None = None) -> str:
    """Resolve the target op from a reference problem.py Model.forward.

    Multiple distinct operators -> ambiguous error; none -> ask for op_name.
    """
    tc = trace or TraceCollector()
    text = problem_py.read_text(encoding="utf-8")
    hits = {name for pat, name in PROBLEM_OP_PATTERNS if re.search(pat, text)}
    if len(hits) == 1:
        op = hits.pop()
        tc.add("opname.derived",
               f"problem.py '{problem_py}' matched pattern -> '{op}' (priority 3)")
        return op
    if len(hits) > 1:
        tc.add("opname.derived",
               f"problem.py matches multiple operators {sorted(hits)} -> ambiguous")
        raise BridgeError(
            f"problem.py matches multiple operators {sorted(hits)}; pass op_name explicitly"
        )
    tc.add("opname.derived",
           f"problem.py '{problem_py}' did not match any operator pattern")
    raise BridgeError("problem.py did not resolve a target operator; pass op_name explicitly")


def check_op_target_consistency(op: str, target: str | None) -> None:
    """Block mismatched names (e.g. sigmoid kernel replacing relu).

    batch-1 does not expose target_op through the TS tool yet, so target is
    usually None and this check is a no-op until later batches wire it up.
    """
    if not target or op == target:
        return
    if target in OP_ALIAS.get(op, set()):
        return
    if op.split("_")[0] == target.split("_")[0]:  # relu_opt vs relu
        return
    raise BridgeError(
        f"op name '{op}' does not align with target training operator '{target}'; "
        f"name the op after what it replaces (target=relu -> op=relu); "
        f"if intentional, pass op_name explicitly and confirm"
    )


# --------------------------------------------------------------------------- #
# Solution-A metadata synthesis & read helpers (bottleneck-1 fix)
#
# Metadata is the contract between a KernelAgent-generated bundle and the
# injector. When a bundle ships its own metadata.json we use it verbatim;
# otherwise we synthesise an equivalent metadata dict from the existing bundle
# files (setup.py / kernel.py / binding.cpp). This keeps the injector decoupled
# from any KernelAgent code change.
# --------------------------------------------------------------------------- #

_AUTOGRAD_BASE_ATTR = "Function"  # name of torch.autograd.Function base class

# Modules whose import lines must not be treated as a "compiled extension" when
# cross-checking the kernel's extension import against metadata.
_CROSSCHECK_IGNORE_MODULES = frozenset({
    "torch", "torch.nn", "torch.nn.functional", "os", "sys", "math", "numpy", "np",
})


def _find_func_def(tree: ast.AST, name: str) -> "ast.FunctionDef | None":
    """Return the FunctionDef named `name`, or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _find_autograd_subclass(tree: ast.AST) -> "str | None":
    """Return the name of a torch.autograd.Function subclass, or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if (isinstance(base, ast.Attribute) and base.attr == _AUTOGRAD_BASE_ATTR) or \
                   (isinstance(base, ast.Name) and base.id == _AUTOGRAD_BASE_ATTR):
                    return node.name
    return None


def _annotation_to_schema(ann: "ast.AST | None") -> str:
    """Map a Python annotation to a torch.library schema type string.

    Supports Tensor / scalar base types plus Optional[X] / Tuple[X, Y] /
    List[X]. Used by both derive_schema (multi-arg) and metadata synthesis.
    """
    if ann is None:
        return "Any"
    if isinstance(ann, ast.Attribute):
        if ann.attr == "Tensor":
            return "Tensor"
        # typing.Optional[...] value recursion; other attrs fall back to name
        return _annotation_to_schema(ann.value) if ann.attr == "Optional" else ann.attr
    if isinstance(ann, ast.Name):
        return {"int": "int", "float": "float", "bool": "bool", "Tensor": "Tensor"}.get(ann.id, "Any")
    if isinstance(ann, ast.Subscript):  # Optional[X] / Tuple[X, Y] / List[X] / Tensor?
        if isinstance(ann.value, (ast.Attribute, ast.Name)):
            base = ann.value.attr if isinstance(ann.value, ast.Attribute) else ann.value.id
            if base == "Optional":
                return _annotation_to_schema(ann.slice) + "?"
            if base == "List":
                return _annotation_to_schema(ann.slice)
            if base == "Tuple":
                elems = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
                return "(" + ", ".join(_annotation_to_schema(e) for e in elems) + ")"
        return _annotation_to_schema(ann.value)
    return "Any"


def _read_op_from_side_info(run_dir: Path) -> "str | None":
    """Best-effort op-name discovery from KernelAgent side-info files.

    Priority: demo_summary.json (semantic_type) > problem.txt heuristic.
    Returns a lowercase op name or None.
    """
    cand = run_dir / "demo_summary.json"
    if cand.is_file():
        try:
            st = json.loads(cand.read_text(encoding="utf-8")).get("experience", {}).get("semantic_type")
            if st:
                return str(st)
        except Exception:
            pass
    prob = run_dir / "problem.txt"
    if prob.is_file():
        try:
            txt = prob.read_text(encoding="utf-8")
            m = re.search(r"(?:op|operator|relu|sigmoid|silu)\b[^\n]*", txt, re.I)
            hit = re.search(r"\b(relu|sigmoid|silu)\b", m.group(0), re.I) if m else None
            if hit:
                return hit.group(0).lower()
        except Exception:
            pass
    return None


def _synthesize_metadata_from_existing(bundle_dir: Path, run_dir: "Path | None") -> "dict | None":
    """Synthesise a metadata dict from the bundle's real files.

    Reads setup.py (extension name + backend), kernel.py (AST: forward entry and
    backward style), and binding.cpp (C++ symbols + device/dtype guards).
    Returns None when the bundle lacks the required kernel.py / setup.py.
    """
    kernel_py = bundle_dir / "kernel.py"
    setup_py = bundle_dir / "setup.py"
    binding_cpp = bundle_dir / "binding.cpp"
    if not (kernel_py.is_file() and setup_py.is_file()):
        return None

    meta: dict = {"version": "2"}

    setup_src = setup_py.read_text(encoding="utf-8")
    m_name = re.search(r'name\s*=\s*["\']([^"\']+)["\']', setup_src)
    if m_name:
        meta["extension_module"] = m_name.group(1)
    m_ext = re.search(r"(MUSAExtension|CUDAExtension|CppExtension|HIPExtension)", setup_src)
    if m_ext:
        meta["backend"] = {
            "MUSAExtension": "musa", "CUDAExtension": "cuda",
            "CppExtension": "cpu", "HIPExtension": "hip",
        }[m_ext.group(1)]

    tree = ast.parse(kernel_py.read_text(encoding="utf-8"))
    if _find_func_def(tree, "kernel_function") is None:
        return None
    meta["kernel_module"] = "kernel"
    meta["forward_entry"] = "kernel_function"

    autograd_cls = _find_autograd_subclass(tree)
    if autograd_cls:
        meta["backward_mode"] = "kernel"
        meta["backward_style"] = "autograd_function"
        meta["backward_entry"] = autograd_cls
        # Compatibility: an explicit kernel_backward function implies the
        # "separate" style (forward/backward registered as two distinct ops).
        if _find_func_def(tree, "kernel_backward") is not None:
            meta["backward_style"] = "separate"
            meta["backward_entry"] = "kernel_backward"
    else:
        meta["backward_mode"] = "eager"

    if binding_cpp.is_file():
        csrc = binding_cpp.read_text(encoding="utf-8")
        defs = re.findall(r'm\.def\(\s*["\']([^"\']+)["\']', csrc)
        if defs:
            meta["forward_symbol"] = next((d for d in defs if "forward" in d), defs[0])
            meta["backward_symbol"] = next((d for d in defs if "backward" in d), defs[-1])
        if "is_musa()" in csrc:
            meta["device"] = "musa"
        elif "is_cuda()" in csrc:
            meta["device"] = "cuda"
        if "kFloat" in csrc:
            meta["dtype"] = ["float32"]
        elif "kHalf" in csrc:
            meta["dtype"] = ["float16"]

    op = _read_op_from_side_info(run_dir or bundle_dir.parent)
    if op:
        meta["op"] = op
    return meta


def _read_metadata(bundle_dir: Path, run_dir: "Path | None" = None) -> dict:
    """Unified metadata entry point.

    P1: bundle ships its own metadata.json -> use it verbatim.
    P2: otherwise synthesise from existing files.
    P3: empty shell so downstream code can fall back field-by-field.
    """
    bundled = bundle_dir / "metadata.json"
    if bundled.is_file():
        try:
            return json.loads(bundled.read_text(encoding="utf-8"))
        except Exception:
            pass
    synth = _synthesize_metadata_from_existing(bundle_dir, run_dir)
    if synth is not None:
        return synth
    return {"version": "2"}


def _cross_check_extension(meta: dict, kernel_py: Path, tc: "TraceCollector | None" = None) -> bool:
    """A8/R3: extension imported by kernel.py must match meta.extension_module.

    We scan *all* import statements (not just the first line) because kernel.py
    typically imports `torch` before the compiled extension. On mismatch we
    downgrade to eager mode (falls back to OP_FORMULAS) rather than silently
    mis-registering the kernel.
    """
    if not kernel_py.is_file() or "extension_module" not in meta:
        return True
    src = kernel_py.read_text(encoding="utf-8")
    # Collect every top-level imported module name ("import x" / "from x import").
    found = re.findall(r"^\s*(?:import\s+([A-Za-z_][\w]*)|from\s+([A-Za-z_][\w]*)\s+import)", src, re.M)
    imported_names = [a or b for a, b in found]
    ext = meta["extension_module"]
    if ext in imported_names:
        if tc:
            tc.add("meta.crosscheck", f"OK: kernel.py imports '{ext}' == metadata")
        return True
    # Extension not imported: only downgrade if a *different* extension is the
    # one actually imported (a real mismatch). Standard libs (torch/os/sys/...)
    # are ignored.
    ext_candidates = [n for n in imported_names if n not in _CROSSCHECK_IGNORE_MODULES]
    if ext_candidates:
        msg = (f"cross-check FAIL: kernel.py imports '{ext_candidates[0]}' but metadata "
               f"extension_module='{ext}'; downgrading to eager")
        if tc:
            tc.add("meta.crosscheck", msg)
        meta["backward_mode"] = "eager"
        meta["_crosscheck_warning"] = msg
        return False
    if tc:
        tc.add("meta.crosscheck", f"OK: no conflicting extension import (expected '{ext}')")
    return True


# --------------------------------------------------------------------------- #
# R6: Training script dynamic generation & injection
# --------------------------------------------------------------------------- #

# Per-op backward formula registry (deterministic, no LLM).
# torch_integration generator and training-scaffold generator share this.
OP_FORMULAS: dict[str, dict] = {
    "relu": {
        "save": ["x"],
        "backward": "return grad_out * (x > 0).to(grad_out.dtype)",
    },
    "sigmoid": {
        "save": ["x", "y"],
        "backward": "return grad_out * y * (1 - y)",
    },
    "silu": {
        "save": ["x", "y"],
        "backward": (
            "sig = torch.sigmoid(x)\n"
            "        return grad_out * (sig * (1 + x * (1 - sig)))"
        ),
    },
}


INJECTION_BLOCK_START = "# ===== kernelagent-runtime injection (auto-generated by kernelagent-injector) ====="
INJECTION_BLOCK_END = "# ===== end injection ====="


def _generate_class_name(op: str, script_text: str) -> str:
    """Generate a scaffold class name, avoiding collisions with user symbols."""
    base = f"_KA{op.capitalize()}"
    if base not in script_text:
        return base
    # collision: add random suffix
    import random, string
    suffix = "".join(random.choices(string.ascii_lowercase, k=4))
    return f"{base}_{suffix}"


def analyze_train_script(script: Path, op: str) -> dict:
    """AST analysis: alias map + call forms + insertion point + warnings.

    Returns:
        {
            "aliases": {"torch": "torch", "F": "F", ...},
            "forms": [("F", "relu"), ("torch", "relu"), ...],
            "insert_after_line": N,
            "nn_module_used": bool,
            "inplace_warn": bool,
            "tensor_method_used": bool,  # x.op() / self.op() call form (R6 D-R6-1)
            "stateful_op": bool,        # batch/instance/layer norm -> keep native (R7 D-R7-1)
        }
    """
    text = script.read_text(encoding="utf-8")
    tree = ast.parse(text)

    aliases: dict[str, str] = {}  # alias -> canonical module name
    forms: list[tuple[str, str]] = []
    nn_module_used = False
    inplace_warn = False
    tensor_method_used = False
    stateful_op = False
    last_import_line = 0
    # Stateful ops whose kernel is a pure function and cannot carry running
    # stats/buffers; we keep them native instead of patching (R7 D-R7-1).
    # Detection is scoped to the op being injected: a script that merely uses
    # BatchNorm while we are patching relu must NOT be skipped.
    _stateful_ops = {"batch_norm", "instance_norm", "layer_norm"}
    _stateful_class_to_op = {
        "BatchNorm1d": "batch_norm", "BatchNorm2d": "batch_norm", "BatchNorm3d": "batch_norm",
        "InstanceNorm1d": "instance_norm", "InstanceNorm2d": "instance_norm", "InstanceNorm3d": "instance_norm",
        "LayerNorm": "layer_norm",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    aliases[alias.asname or alias.name] = "torch"
                elif alias.name == "torch.nn.functional":
                    aliases[alias.asname or alias.name] = "F"
            last_import_line = max(last_import_line, node.end_lineno or node.lineno)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "torch":
                for alias in node.names:
                    if alias.name == "nn":
                        aliases[alias.asname or alias.name] = "nn"
            elif module == "torch.nn.functional":
                for alias in node.names:
                    if alias.name == "relu":
                        aliases[alias.asname or alias.name] = "F.relu"
            last_import_line = max(last_import_line, node.end_lineno or node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            # Detect nn.ReLU() instantiation
            if isinstance(func, ast.Attribute) and func.attr == "ReLU":
                nn_module_used = True
            # Detect functional call forms
            if isinstance(func, ast.Attribute) and func.attr == op:
                if isinstance(func.value, ast.Name) and func.value.id in aliases:
                    forms.append((func.value.id, op))
                else:
                    # x.op() / self.op() / other carrier: a Tensor-method form
                    # that needs the torch.Tensor.{op} monkeypatch (R6 D-R6-1).
                    tensor_method_used = True
            # Detect stateful ops (batch/instance/layer norm) -> keep native (R7 D-R7-1).
            # Only when the op we are injecting is itself a stateful op.
            if op in _stateful_ops:
                if isinstance(func, ast.Attribute) and func.attr == op:
                    stateful_op = True
                elif isinstance(func, ast.Attribute) and func.attr in _stateful_class_to_op:
                    if _stateful_class_to_op[func.attr] == op:
                        stateful_op = True
            # Detect inplace=True kwarg
            for kw in node.keywords:
                if kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    inplace_warn = True

    # Build canonical alias map for patching
    patch_aliases: dict[str, str] = {}
    for alias, canonical in aliases.items():
        if canonical == "torch":
            patch_aliases["torch"] = alias
        elif canonical == "F":
            patch_aliases["F"] = alias

    return {
        "aliases": patch_aliases,
        "forms": forms,
        "insert_after_line": last_import_line,
        "nn_module_used": nn_module_used,
        "inplace_warn": inplace_warn,
        "tensor_method_used": tensor_method_used,
        "stateful_op": stateful_op,
    }


def _param_names_from_schema(schema: "str | None", fallback: list[str]) -> list[str]:
    """Extract parameter names from a torch.library schema string.

    Used by the eager scaffold so that multi-argument operators are patched
    correctly (bottleneck-2 fix). Falls back to `fallback` when no schema.
    """
    if not schema:
        return fallback
    m = re.search(r"\(([^)]*)\)", schema)
    if not m or not m.group(1).strip():
        return fallback
    names: list[str] = []
    for part in m.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        names.append(part.split()[-1])  # "Tensor a" / "int k" -> "a" / "k"
    return names or fallback


def _get_backward_info(op: str, op_dir: "Path | None" = None, meta: "dict | None" = None) -> dict:
    """Resolve backward-mode info for `op`.

    Returns a dict with at least {"mode": ...}:
      - mode="kernel", style=autograd_function: backward is embedded in the
        registered forward entry (no extra scaffold needed).
      - mode="kernel", style=separate: a distinct backward op must be wired.
      - mode="eager": use the OP_FORMULAS formula as a fallback.

    Raises BridgeError when no backward info can be found (A6: explicit, not silent).
    """
    if meta is None and op_dir is not None:
        mp = op_dir / "_meta.json"
        if mp.is_file():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                meta = None

    if meta:
        mode = meta.get("backward_mode")
        if mode == "kernel":
            return {
                "mode": "kernel",
                "style": meta.get("backward_style", "autograd_function"),
                "forward_entry": meta.get("forward_entry", "kernel_function"),
                "backward_entry": meta.get("backward_entry"),
                "schema": meta.get("schema"),
                "extension_module": meta.get("extension_module"),
                "dtype": meta.get("dtype"),
                "device": meta.get("device"),
            }
        if mode == "eager" or "backward" in meta:
            return {"mode": "eager", **meta.get("backward", {})}

    if op in OP_FORMULAS:
        return {"mode": "eager", **OP_FORMULAS[op]}

    raise BridgeError(
        f"no backward info for op '{op}'; "
        f"provide metadata.json(backward_mode=kernel) or extend OP_FORMULAS"
    )


def _scaffold_kwfilter_source(op: str, allowed: set[str]) -> str:
    """Generate the ``_ka_kwfilter`` helper for the injected training scaffold.

    R1a: module-delegated calls (e.g. ``nn.ReLU.forward`` -> ``F.relu(input,
    inplace=False)``) pass kwargs the kernel op's schema does not declare. We drop
    those kwargs (decision: discard + trace warning) so the routed call does not
    raise TypeError, while still making the substitution observable on stderr.
    """
    return (
        f"_KA_ALLOWED = {sorted(allowed)!r}\n"
        f"\n"
        f"def _ka_kwfilter(kw):\n"
        f"    out = {{}}\n"
        f"    for _k, _v in kw.items():\n"
        f"        if _k in _KA_ALLOWED:\n"
        f"            out[_k] = _v\n"
        f"        else:\n"
        f"            sys.stderr.write(\n"
        f"                f\"[ka-inject] dropped unsupported kwarg '{{_k}}' for op '{op}'\\n\")\n"
        f"    return out\n"
    )


def _normalize_list(v):
    """Normalize a metadata dtype/device value into a list-literal string.

    Accepts None (-> "[]" no guard), a single string ("float32"), or a
    list (["float32","bf16"]); always returns a Python list-literal string so
    it can be embedded verbatim in generated code (R9 dtype/device guard).
    """
    if v is None or v == "":
        items: list = []
    elif isinstance(v, str):
        items = [v]
    elif isinstance(v, (list, tuple)):
        items = list(v)
    else:
        items = [str(v)]
    return repr(items)


def generate_injected_train_script(
    script: Path,
    op: str,
    trace: TraceCollector | None = None,
    meta: "dict | None" = None,
    schema: "str | None" = None,
) -> Path:
    """Generate <stem>_ka_injected.py by inserting a kernelagent-runtime scaffold.

    The scaffold patches torch.nn.functional and torch module attributes
    (alias-proof) so any F.relu(...) / torch.relu(...) calls in the original
    script automatically route to torch.ops.kernelagent.<op>.
    """
    tc = trace
    # No longer require op in OP_FORMULAS; resolve backward info instead.
    bwd = _get_backward_info(op, script.parent / "ops" / op, meta)
    mode = bwd.get("mode")
    if mode == "eager" and op not in OP_FORMULAS and "backward" not in bwd:
        raise BridgeError(f"op '{op}' has no backward info (not in OP_FORMULAS, no metadata)")

    text = script.read_text(encoding="utf-8")
    analysis = analyze_train_script(script, op)

    # Strip any existing injection block to prevent bloat on re-generation
    stripped = text
    while True:
        start = stripped.find(INJECTION_BLOCK_START)
        if start == -1:
            break
        end = stripped.find(INJECTION_BLOCK_END, start)
        if end == -1:
            break
        stripped = stripped[:start] + stripped[end + len(INJECTION_BLOCK_END):]
        # clean up extra blank lines
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)

    # Determine insertion point (after last top-level import)
    lines = stripped.splitlines(keepends=True)
    insert_line = analysis["insert_after_line"]
    if insert_line < 1 or insert_line > len(lines):
        insert_line = 0

    class_name = _generate_class_name(op, stripped)
    aliases = analysis["aliases"]

    # R1a: build the kwarg filter once. Module-delegated calls (e.g.
    # nn.ReLU.forward -> F.relu(input, inplace=False)) pass kwargs the kernel op
    # rejects; the filter drops them and traces a warning (discard + trace).
    _allowed = set(_param_names_from_schema(schema, ["x"])) | {"x", "input"}

    stateful_op = analysis.get("stateful_op", False)
    tensor_method_used = analysis.get("tensor_method_used", False)

    # R6 (D-R6-1): route x.{op}() Tensor-method calls to the kernel via a
    # torch.Tensor.{op} monkeypatch. Global side-effect, but covers every call
    # form (x.op(), self.op(), callbacks) without any AST rewriting.
    _r6_patch = ""
    if tensor_method_used:
        _r6_patch = (
            f"\n"
            f"# R6: route x.{op}() (Tensor method) to the kernel via monkeypatch.\n"
            f"# Global side-effect on torch.Tensor.{op}; bypasses some C fast paths.\n"
            f"torch.Tensor.{op} = staticmethod(_ka_{op}_wrap)\n"
            f'sys.stderr.write(f"[ka-inject] routed torch.Tensor.{op}() to kernelagent\\n")\n'
        )

    # R7-2: surface inplace=True stripping to stderr. The kwarg is already
    # dropped by _ka_kwfilter; this only makes the substitution observable.
    _inplace_warn = ""
    if analysis.get("inplace_warn"):
        _inplace_warn = (
            f'sys.stderr.write('
            f'"[ka-inject] WARNING: inplace=True on op \'{op}\' is dropped by the kernel path\\n")\n'
        )

    if stateful_op:
        # R7-1: stateful op (batch/instance/layer norm) -> keep native.
        # The kernel is a pure function and cannot carry running stats/buffers,
        # so we skip patching and warn instead of silently losing state.
        scaffold = (
            f"{INJECTION_BLOCK_START}\n"
            f"# op: {op} | SKIPPED: stateful op kept native (kernel path skipped)\n"
            f"import sys\n"
            f'sys.stderr.write('
            f'"[ka-inject] WARNING: stateful op \'{op}\' kept native; kernel path skipped\\n")\n'
            f"{INJECTION_BLOCK_END}\n"
        )
    elif mode == "kernel":
        style = bwd.get("style", "autograd_function")
        if style == "separate":
            # Mode 1b: separate style. The bundle ships the forward (kernel_function)
            # and backward (kernel_backward) as TWO distinct kernels. torch_integration.py
            # registers both ops and defines a bridge autograd.Function
            # (kernelagent_runtime._KA_<op>_Fn) that "wires" them into one
            # differentiable operator. We route the patched call through that bridge.
            _call_expr = f"kernelagent_runtime._KA_{op}_Fn.apply(x, *a, **_ka_kwfilter(kw))"
            _mode_comment = f"kernel(separate) | backward wired via kernelagent_runtime._KA_{op}_Fn"
            _rt_note = " AND exposes _KA_{op}_Fn bridge"
        else:
            # Mode 1: autograd_function style. Backward is embedded inside the
            # registered forward entry (an autograd.Function); patching the module
            # attributes is sufficient, and the call routes straight to the op.
            _call_expr = f"torch.ops.kernelagent.{op}(x, *a, **_ka_kwfilter(kw))"
            _mode_comment = "kernel(autograd_function) | backward embedded; replaces torch.{op}/F.{op}"
            _rt_note = ""
        patch_lines = []
        if "F" in aliases:
            patch_lines.append(f'_ka_F.{op} = lambda x, *a, **kw: _ka_{op}_wrap(x, *a, **kw)')
        if "torch" in aliases:
            patch_lines.append(f'_ka_torch.{op} = lambda x, *a, **kw: _ka_{op}_wrap(x, *a, **kw)')
        if not patch_lines:
            patch_lines = [
                f'_ka_F.{op} = lambda x, *a, **kw: _ka_{op}_wrap(x, *a, **kw)',
                f'_ka_torch.{op} = lambda x, *a, **kw: _ka_{op}_wrap(x, *a, **kw)',
            ]
        scaffold = (
            f"{INJECTION_BLOCK_START}\n"
            f"# op: {op} | {_mode_comment}\n"
            f"import sys\n"
            f"import atexit\n"
            f"import kernelagent_runtime  # registers torch.ops.kernelagent.*{_rt_note}\n"
            f"import torch as _ka_torch\n"
            f"import torch.nn.functional as _ka_F\n"
            f'_KA_CALLS = {{"{op}": 0}}\n'
            f"\n"
            + _scaffold_kwfilter_source(op, _allowed) +
            f"\n"
            f"def _ka_{op}_wrap(x, *a, **kw):\n"
            f"    _KA_CALLS['{op}'] += 1\n"
            f"    return {_call_expr}\n"
            + _r6_patch +
            f"\n"
            f"_KA_ORIG = {{'F.{op}': _ka_F.{op}, 'torch.{op}': _ka_torch.{op}}}\n"
            + "\n".join(patch_lines) + "\n"
            f"\n"
            f"def _ka_report():\n"
            f"    print(f\"[ka-inject] {op} kernel calls = {{_KA_CALLS['{op}']}}\")\n"
            f"atexit.register(_ka_report)\n"
            f"{INJECTION_BLOCK_END}\n"
        )
    else:
        # Mode 2: eager (old bundle op in OP_FORMULAS or explicit backward dict).
        formula = bwd if "backward" in bwd else OP_FORMULAS[op]
        # Bottleneck-2 fix: derive parameter names from schema for N-arg ops;
        # single-arg stays identical to the legacy scaffold (A5 regression).
        param_names = _param_names_from_schema(
            meta.get("schema") if meta else None, formula.get("save", ["x"])
        )
        multi = len(param_names) > 1
        save_stmt = "ctx.save_for_backward(*args)" if multi else f"ctx.save_for_backward({param_names[0]})"
        forward_arg = "*args" if multi else param_names[0]
        call_args = "*args" if multi else param_names[0]
        unpack = (", ".join(param_names) + " = ctx.saved_tensors") if multi else f"{param_names[0]}, = ctx.saved_tensors"

        patch_lines = []
        if "F" in aliases:
            patch_lines.append(f'_ka_F.{op} = lambda *args, **kw: {class_name}.apply(*args, **_ka_kwfilter(kw))')
        if "torch" in aliases:
            patch_lines.append(f'_ka_torch.{op} = lambda *args, **kw: {class_name}.apply(*args, **_ka_kwfilter(kw))')
        if not patch_lines:
            patch_lines = [
                f'_ka_F.{op} = lambda *args, **kw: {class_name}.apply(*args, **_ka_kwfilter(kw))',
                f'_ka_torch.{op} = lambda *args, **kw: {class_name}.apply(*args, **_ka_kwfilter(kw))',
            ]
        scaffold = (
            f"{INJECTION_BLOCK_START}\n"
            f"# op: {op} | mode: eager | params={param_names}\n"
            f"import sys\n"
            f"import atexit\n"
            f"\n"
            f"import kernelagent_runtime\n"
            f"import torch as _ka_torch\n"
            f"import torch.nn.functional as _ka_F\n"
            + _scaffold_kwfilter_source(op, _allowed) +
            f"\n"
            + _inplace_warn +
            f'_KA_CALLS = {{"{op}": 0}}\n'
            f"\n"
            f"\n"
            f"def _ka_{op}_wrap(*a, **kw):\n"
            f"    _KA_CALLS['{op}'] += 1\n"
            f"    return {class_name}.apply(*a, **_ka_kwfilter(kw))\n"
            + _r6_patch +
            f"\n"
            f"class {class_name}(torch.autograd.Function):\n"
            f"    @staticmethod\n"
            f"    def forward(ctx, {forward_arg}):\n"
            f"        {save_stmt}\n"
            f"        _KA_CALLS['{op}'] += 1\n"
            f"        return torch.ops.kernelagent.{op}({call_args})\n"
            f"\n"
            f"    @staticmethod\n"
            f"    def backward(ctx, grad_out):\n"
            f"        {unpack}\n"
            f"        {formula['backward']}\n"
            f"\n"
            f"\n"
            f'_KA_ORIG = {{"F.{op}": _ka_F.{op}, "torch.{op}": _ka_torch.{op}}}\n'
            + "\n".join(patch_lines) + "\n"
            f"\n"
            f"\n"
            f"def _ka_report():\n"
            f"    print(f\"[ka-inject] {op} kernel calls = {{_KA_CALLS['{op}']}}\")\n"
            f"\n"
            f"\n"
            f"atexit.register(_ka_report)\n"
            f"{INJECTION_BLOCK_END}\n"
        )

    # Insert scaffold
    new_lines = lines[:insert_line] + ["\n", scaffold, "\n"] + lines[insert_line:]
    injected_name = script.with_suffix("").name + "_ka_injected.py"
    injected_path = script.parent / injected_name
    injected_path.write_text("".join(new_lines), encoding="utf-8")

    if tc:
        tc.add("traingen.analyze",
               f"aliases={aliases} forms={analysis['forms']} insert_after_line={insert_line} "
               f"nn_module_used={analysis['nn_module_used']} inplace_warn={analysis['inplace_warn']} "
               f"tensor_method_used={analysis.get('tensor_method_used', False)} "
               f"stateful_op={analysis.get('stateful_op', False)}")
        tc.add("traingen.write",
               f"generated {injected_path} (scaffold {scaffold.count(chr(10))} lines, "
               f"class={class_name})")

    return injected_path


# --------------------------------------------------------------------------- #
# Source resolution (example / run dir / bundle dir / single file / payload)
# --------------------------------------------------------------------------- #

# Serialized multi-file payload, e.g. optimized_kernel_musa.py produced by
# generate/optimize: "FILE: kernel.py\n```python\n...\n```" repeated.
SERIALIZED_FILE_RE = re.compile(
    r"FILE:\s*(\S+)[^\n]*\n+```[a-zA-Z]*\n(.*?)```", re.DOTALL
)


def _latest_run(results_root: Path) -> Path | None:
    runs = sorted(d for d in results_root.iterdir()
                  if d.is_dir() and d.name.startswith("run_"))
    return runs[-1] if runs else None


def _read_status(run_dir: Path) -> str:
    ds = run_dir / "demo_summary.json"
    if not ds.is_file():
        return "UNKNOWN"
    try:
        summary = json.loads(ds.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "UNKNOWN"
    return str(summary.get("status", "UNKNOWN")).upper()


# A bundle directory the caller points at directly is NOT automatically an
# unoptimized baseline. Only sources that are literally named/declared as
# baselines keep kind="baseline"; everything else is kind="local" and its
# status is read from its own provenance metadata instead of being assumed.
BASELINE_DIR_NAMES = ("baseline_bundle",)

PROVENANCE_FILES = ("demo_summary.json", "result.json", "_meta.json", "_provenance.json")


def _load_json_file(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _status_of(data: Any) -> str:
    """Extract a status string from a provenance document."""
    if not isinstance(data, dict):
        return ""
    status = str(data.get("status", "")).upper()
    if status:
        return status
    if data.get("success") is True:
        return "SUCCESS"
    if data.get("success") is False:
        return "FAILED"
    return ""


def _read_local_status(d: Path) -> str:
    """Status for a caller-supplied bundle, read from real provenance metadata.

    Looks at the bundle directory and its parent; a provenance file may point
    (via "result_json"/"result_path") at the original generation result, which
    is then followed so the status comes from the real artifact.
    """
    for base in (d, d.parent):
        if base is None:
            continue
        for name in PROVENANCE_FILES:
            f = base / name
            if not f.is_file():
                continue
            data = _load_json_file(f)
            if data is None:
                continue
            if isinstance(data, dict):
                for key in ("result_json", "result_path"):
                    ptr = data.get(key)
                    if isinstance(ptr, str) and Path(ptr).is_file():
                        status = _status_of(_load_json_file(Path(ptr)))
                        if status:
                            return status
            status = _status_of(data)
            if status:
                return status
    return "UNKNOWN"


def classify_bundle(d: Path, run_layout: bool) -> tuple[str, str]:
    """Classify a bundle by provenance instead of by its parent directory name."""
    if run_layout:
        return "run", _read_status(d.parent)
    if d.name in BASELINE_DIR_NAMES or (d / "BASELINE").is_file():
        return "baseline", "BASELINE"
    return "local", _read_local_status(d)


def _is_bundle_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    names = {p.name for p in d.iterdir() if p.is_file()}
    return all(f in names for f in MUSA_FILES) or "kernel.py" in names


def materialize_bundle_from_payload(payload_py: Path, staging_root: Path) -> Path:
    """Materialize a 'FILE: xxx\\n```...```' serialized file into a bundle dir."""
    text = payload_py.read_text(encoding="utf-8")
    files = SERIALIZED_FILE_RE.findall(text)
    if not files:
        raise BridgeError(
            f"{payload_py} is not a serialized bundle (no FILE blocks found); "
            f"if it is a plain single-file kernel it will be handled as triton single-file"
        )
    bundle = staging_root / payload_py.stem / "best_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    for name, code in files:
        (bundle / name.strip()).write_text(code, encoding="utf-8")
    return bundle


def resolve_source(
    source: str, working_dir: Path, staging_root: Path,
    allow_baseline: bool = True, trace: TraceCollector | None = None
) -> dict:
    """Resolve any supported source form to {bundle_dir, backend, kind, status}."""
    tc = trace or TraceCollector()
    raw = source.strip()
    tc.add("source.input", f"input={raw!r}")
    p = Path(raw)

    # S1: example name (no path separator)
    if "/" not in raw and "\\" not in raw:
        example = raw
        ex_dir = working_dir / "examples" / example
        if not ex_dir.is_dir():
            raise BridgeError(f"example not found: {ex_dir}")
        results_root = ex_dir / "results"
        if results_root.is_dir():
            run_dir = _latest_run(results_root)
            if run_dir is not None:
                bundle_dir = run_dir / "best_bundle"
                if bundle_dir.is_dir():
                    tc.add("source.probe",
                           f"results/run_*: found -> latest {run_dir.name}")
                    tc.add("source.resolved",
                           f"bundle_dir={bundle_dir}, backend={detect_backend(bundle_dir)}, "
                           f"status={_read_status(run_dir)}, kind=run")
                    return {
                        "kind": "run",
                        "example": example,
                        "run_dir": str(run_dir),
                        "best_bundle_dir": str(bundle_dir),
                        "backend": detect_backend(bundle_dir),
                        "status": _read_status(run_dir),
                        "problem_py": _find_problem_py(ex_dir, run_dir),
                    }
            tc.add("source.probe", "results/run_*: none found")
        else:
            tc.add("source.probe", "results/: directory does not exist")
        # baseline fallback: hand-written, unoptimized but correct bundle
        baseline = ex_dir / "baseline_bundle"
        if _is_bundle_dir(baseline):
            if not allow_baseline:
                raise BridgeError(
                    f"example '{example}' has no run_* results and allow_baseline=false; "
                    f"run the example first to generate real optimized kernel"
                )
            tc.add("source.resolved",
                   f"baseline_bundle exists -> fallback (kind=baseline)")
            return {
                "kind": "baseline",
                "example": example,
                "run_dir": "",
                "best_bundle_dir": str(baseline),
                "backend": detect_backend(baseline),
                "status": "BASELINE",
                "problem_py": _find_problem_py(ex_dir, None),
            }
        raise BridgeError(
            f"example '{example}' has no run_* results and no baseline_bundle/; "
            f"nothing to deploy"
        )

    # S3: file path (.py): serialized payload or plain single-file kernel
    if p.is_file() and p.suffix == ".py":
        materialized = materialize_bundle_from_payload(p, staging_root)
        tc.add("source.resolved",
               f"serialized payload -> {materialized}, kind=serialized")
        return {
            "kind": "serialized",
            "example": "",
            "run_dir": "",
            "best_bundle_dir": str(materialized),
            "backend": detect_backend(materialized),
            "status": "UNKNOWN",
            "problem_py": _find_problem_py(p.parent, None),
        }

    # S2: directory path
    if p.is_absolute() or p.is_dir():
        d = p if p.is_dir() else p.parent
        # run_* directory
        if d.name.startswith("run_") and (d / "best_bundle").is_dir():
            bundle_dir = d / "best_bundle"
            tc.add("source.resolved",
                   f"run_dir={d}, kind=run")
            return {
                "kind": "run",
                "example": d.parent.parent.name if d.parent.parent.name.startswith("optimize_") else "",
                "run_dir": str(d),
                "best_bundle_dir": str(bundle_dir),
                "backend": detect_backend(bundle_dir),
                "status": _read_status(d),
                "problem_py": _find_problem_py(d.parent, d),
            }
        # best_bundle directory or a dir containing best_bundle/
        if d.name == "best_bundle" and _is_bundle_dir(d):
            kind, status = classify_bundle(d, d.parent.name.startswith("run_"))
            tc.add("source.resolved",
                   f"best_bundle_dir={d}, kind={kind}")
            return {
                "kind": kind,
                "example": "",
                "run_dir": str(d.parent),
                "best_bundle_dir": str(d),
                "backend": detect_backend(d),
                "status": status,
                "problem_py": _find_problem_py(d.parent.parent, d.parent),
            }
        if (d / "best_bundle").is_dir() and _is_bundle_dir(d / "best_bundle"):
            bundle_dir = d / "best_bundle"
            kind, status = classify_bundle(bundle_dir, d.name.startswith("run_"))
            tc.add("source.resolved",
                   f"dir_with_best_bundle={d}, kind={kind}")
            return {
                "kind": kind,
                "example": "",
                "run_dir": str(d),
                "best_bundle_dir": str(bundle_dir),
                "backend": detect_backend(bundle_dir),
                "status": status,
                "problem_py": _find_problem_py(d.parent, d),
            }
        # bare bundle dir (only literally-named baselines are treated as baselines)
        if _is_bundle_dir(d):
            kind, status = classify_bundle(d, False)
            tc.add("source.resolved",
                   f"bare_bundle_dir={d}, kind={kind}")
            return {
                "kind": kind,
                "example": "",
                "run_dir": "",
                "best_bundle_dir": str(d),
                "backend": detect_backend(d),
                "status": status,
                "problem_py": _find_problem_py(d.parent, None),
            }
        # Fuser artifacts dir (composed kernel) -- not supported in batch-1
        if any(f.name.startswith("composed") for f in d.iterdir() if f.is_file()):
            raise BridgeError(
                f"{d} looks like a Fuser artifacts_dir (composed multi-op module); "
                f"batch-1 only supports single-operator bundles"
            )

    raise BridgeError(
        f"could not resolve source '{raw}'; expected an example name, a run/bundle "
        f"directory, or a single .py kernel / serialized payload file"
    )


def _find_problem_py(example_dir: Path, run_dir: Path | None) -> str:
    candidates = []
    if run_dir is not None:
        candidates.append(run_dir / "problem.py")
    candidates.append(example_dir / "problem.py")
    for c in candidates:
        if c.is_file():
            return str(c)
    return ""


# --------------------------------------------------------------------------- #
# Store validation
# --------------------------------------------------------------------------- #

def validate_store(store: Path) -> None:
    if ".." in store.parts:
        raise BridgeError(f"store path must not contain '..': {store}")
    sp = store.resolve()
    if (sp / "torch").exists() or (sp / "torch_musa").exists():
        raise BridgeError(
            f"store looks like a torch repo root (contains torch/ or torch_musa/): {sp}"
        )
    if "site-packages" in sp.parts:
        raise BridgeError(f"store must not live inside site-packages: {sp}")
    if sp.exists():
        if sp.is_dir() and not any(sp.iterdir()):
            return  # empty dir, usable
        if not (sp / "manifest.json").is_file():
            raise BridgeError(
                f"directory exists but is not a valid runtime store (no manifest.json): {sp}"
            )
    else:
        sp.mkdir(parents=True, exist_ok=True)
        (sp / "ops").mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Auto-generated kernelagent_runtime package template (self-bootstrapping)
# --------------------------------------------------------------------------- #

_RUNTIME_PACKAGE_TEMPLATE = '''\
"""kernelagent_runtime -- auto-generated runtime entry.

``import kernelagent_runtime`` registers every deployed-and-built op in the
runtime store as ``torch.ops.kernelagent::<op>``. No LLM, no patching.

Store location: env ``KERNELAGENT_RUNTIME_STORE``, else the parent directory of
this package (i.e. the directory containing the ``kernelagent_runtime/`` package).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

__all__ = ["enable", "list_ops"]


def _store() -> Path:
    # Prefer the store that actually contains this package (deploy_dir),
    # even if KERNELAGENT_RUNTIME_STORE points at a different default.
    here = Path(__file__).resolve().parent.parent
    if (here / "manifest.json").is_file():
        return here
    env = os.environ.get("KERNELAGENT_RUNTIME_STORE")
    if env:
        return Path(env).resolve()
    return here


def list_ops() -> list[str]:
    mf = _store() / "manifest.json"
    if not mf.is_file():
        return []
    return sorted(json.loads(mf.read_text(encoding="utf-8")).get("ops", {}))


def enable(ops: list[str] | None = None) -> list[str]:
    """Load ``torch_integration.py`` of each requested op.

    By default, all ops whose build succeeded are loaded. Pass an explicit list
    to selectively activate a subset. Falls back to scanning ops/*/ if the
    manifest is missing so verify can run before the index is rewritten.
    """
    store = _store()
    mf = store / "manifest.json"
    targets: list[str] = []
    if ops:
        targets = list(ops)
    elif mf.is_file():
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        targets = [
            op for op, entry in manifest.get("ops", {}).items()
            if entry.get("build", {}).get("success", True)
        ]
    ops_root = store / "ops"
    if not targets and ops_root.is_dir():
        targets = sorted(
            p.name for p in ops_root.iterdir()
            if p.is_dir() and (p / "torch_integration.py").is_file()
        )
    loaded: list[str] = []
    for op in targets:
        mod_name = f"kernelagent_ti_{op}"
        if mod_name in sys.modules:
            loaded.append(op)
            continue
        ti = store / "ops" / op / "torch_integration.py"
        if not ti.is_file():
            continue
        spec = importlib.util.spec_from_file_location(mod_name, ti)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        # Re-export any generated bridge Functions (e.g. _KA_relu_Fn) onto the
        # kernelagent_runtime package so training scaffolds can call
        # kernelagent_runtime._KA_<op>_Fn.apply(...) directly. The integration
        # template defines these classes; enable() makes them package-visible.
        _pkg = sys.modules.get("kernelagent_runtime")
        if _pkg is not None:
            for _attr in dir(mod):
                if _attr.startswith("_KA_"):
                    try:
                        setattr(_pkg, _attr, getattr(mod, _attr))
                    except AttributeError:
                        pass
        loaded.append(op)
    return loaded


# Activate on import so training code only needs ``import kernelagent_runtime``.
enable()
'''


def _bootstrap_runtime_package(target_store: Path, default_store: Path) -> bool:
    """Bootstrap kernelagent_runtime package into target_store.

    Always writes the embedded template so deploy_dir gets the current loader
    (manifest-optional scan, idempotent enable). Copies extra files from the
    default store when present.
    """
    target_rt = target_store / "kernelagent_runtime"
    existed = target_rt.exists()
    target_rt.mkdir(parents=True, exist_ok=True)
    src_rt = default_store / "kernelagent_runtime"
    if src_rt.exists() and src_rt.resolve() != target_rt.resolve():
        for item in src_rt.iterdir():
            dest = target_rt / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            elif item.name != "__init__.py":
                shutil.copy2(item, dest)
    (target_rt / "__init__.py").write_text(_RUNTIME_PACKAGE_TEMPLATE, encoding="utf-8")
    return not existed


# --------------------------------------------------------------------------- #
# Generated torch_integration.py template
# --------------------------------------------------------------------------- #

TORCH_INTEGRATION_TEMPLATE = '''\
# Auto-generated by kernelagent-injector (batch-1). DO NOT EDIT.
# Source: {example} @ {run_dir} (kind={kind})
"""Registers torch.ops.kernelagent.{op} for the deployed kernel bundle."""
import importlib.util
import os
import sys

import torch

_OP = "{op}"
_DISPATCH_KEY = "{dispatch_key}"
_FORWARD_ENTRY = "{forward_entry}"  # Solution-A: forward entry from metadata (default kernel_function)
_BACKWARD_STYLE = "{backward_style}"  # "autograd_function" | "separate"
_BACKWARD_OP = "{backward_op}"        # separate style only: the distinct backward op name
_BACKWARD_ENTRY = "{backward_entry}"  # separate style only: backward kernel function name
_BACKWARD_SCHEMA = "{backward_schema}"  # separate style only: torch.library schema for the backward op
_HERE = os.path.dirname(os.path.abspath(__file__))

# R9: dtype/device guard. The allowed sets are supplied by the operator's
# metadata (e.g. dtype=["float32","bf16"], device=["musa","cuda"]) and may be
# multi-valued; an empty list means "no guard" (fail-open) so we never silently
# reject a dtype/device the kernel actually supports.
_DTYPE_ALLOW = {dtype}        # list[str]; [] -> no guard
_DEVICE_ALLOW = {device}      # list[str]; [] -> no guard
_DTYPE_ALIASES = {{"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}}


def _ka_dtype_ok(dt):
    if not _DTYPE_ALLOW:
        return True
    _name = str(dt).replace("torch.", "")
    for _d in _DTYPE_ALLOW:
        _canon = _DTYPE_ALIASES.get(_d, _d)
        try:
            if dt == getattr(torch, _canon):
                return True
        except AttributeError:
            if _name == _canon:
                return True
    return False


def _guarded_forward(*args, **kwargs):
    _t = next((_a for _a in args if isinstance(_a, torch.Tensor)), None)
    if _t is not None:
        if not _ka_dtype_ok(_t.dtype):
            raise RuntimeError(
                f"[ka] {{_OP}} expects dtype {{_DTYPE_ALLOW}}, got {{_t.dtype}}")
        if _DEVICE_ALLOW and str(_t.device.type) not in _DEVICE_ALLOW:
            raise RuntimeError(
                f"[ka] {{_OP}} expects device {{_DEVICE_ALLOW}}, got {{_t.device.type}}")
    return forward_fn(*args, **kwargs)


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# kernel.py imports the locally built extension (e.g. "import relu_musa_ext");
# make ops/<op>/ (which contains the .so) importable first.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_kernel_mod = _load_module(f"kernelagent_ops_{{_OP}}_kernel", os.path.join(_HERE, "kernel.py"))
forward_fn = getattr(_kernel_mod, _FORWARD_ENTRY)


def _already_registered(name: str) -> bool:
    try:
        return getattr(torch.ops.kernelagent, name) is not None
    except AttributeError:
        return False


if not _already_registered(_OP):
    try:
        _lib = torch.library.Library("kernelagent", "DEF")
    except RuntimeError:
        _lib = torch.library.Library("kernelagent", "FRAGMENT")
    _lib.define("{schema}")
    # R9: wrap forward_fn in a dtype/device guard. The allowed sets come from the
    # operator's metadata; an empty set means no guard (fail-open).
    _lib.impl(_OP, _guarded_forward, _DISPATCH_KEY)

# --- backward wiring -------------------------------------------------------
# autograd_function: the backward is EMBEDDED inside the forward entry's
#   torch.autograd.Function, so registering the forward op alone (above) is
#   sufficient and nothing more is needed here.
# separate: the bundle ships the backward as a SEPARATE kernel (kernel_backward).
#   We register a second op (<OP>_bwd) and a bridge autograd.Function
#   (_KA_<OP>_Fn) that calls the forward op in forward() and the backward op in
#   backward(), "wiring" the two kernels into one differentiable operator.
if _BACKWARD_STYLE == "separate":
    if not _already_registered(_BACKWARD_OP):
        try:
            _blib = torch.library.Library("kernelagent", "DEF")
        except RuntimeError:
            _blib = torch.library.Library("kernelagent", "FRAGMENT")
        _blib.define(_BACKWARD_SCHEMA)
        _bwd_fn = getattr(_kernel_mod, _BACKWARD_ENTRY)
        _blib.impl(_BACKWARD_OP, _bwd_fn, _DISPATCH_KEY)

    class _KA_{op}_Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            ctx.save_for_backward(*args)
            # Route through the guarded wrapper so R9 dtype/device checks
            # apply to the separate-style forward as well.
            return _guarded_forward(*args)

        @staticmethod
        def backward(ctx, grad_output):
            saved = ctx.saved_tensors
            return getattr(_kernel_mod, _BACKWARD_ENTRY)(grad_output, *saved)

    def _ka_{op}(*a, **kw):
        return _KA_{op}_Fn.apply(*a, **kw)

_ops_registered = _OP  # introspection helper
'''


# --------------------------------------------------------------------------- #
# Verify: pull the real training task's run command/script and execute it to
# confirm the operator is actually wired in. This is a required step of the
# kernelagent_inject flow (per design): deploy alone is not enough evidence.
# --------------------------------------------------------------------------- #

# Convention: <KERNELAGENT_TRAIN_DIR>/verify_tasks.json maps op -> {steps:[...]}.
# Each step: {"name", "cmd", "expect_contains"?, "expect_kernel_calls_gt"?}.
# If the file is absent or has no entry for the op, fall back to these defaults.
DEFAULT_VERIFY_TASKS: dict[str, list[dict]] = {
    "relu": [
        {"name": "smoke", "cmd": "python test_relu_smoke.py", "expect_contains": "[PASS]"},
        {"name": "forward_only", "cmd": "python test_relu_forward_only.py", "expect_contains": "[PASS]"},
        {
            "name": "train",
            "cmd": "python train_mnist_musa_injected.py --epochs 1",
            "expect_contains": "[ok] training finished",
            "expect_kernel_calls_gt": 0,
        },
    ],
}

KERNEL_CALLS_RE = re.compile(r"(\w+)\s+kernel calls\s*=\s*(\d+)")
VERIFY_STEP_TIMEOUT_S = 1800


def _load_verify_tasks(train_dir: Path, op: str) -> list[dict]:
    """Fetch the real training task steps for the op from the train dir
    convention file, falling back to built-in defaults for known ops."""
    vf = train_dir / "verify_tasks.json"
    if vf.is_file():
        try:
            data = json.loads(vf.read_text(encoding="utf-8"))
            tasks = data.get(op)
            if isinstance(tasks, dict) and isinstance(tasks.get("steps"), list):
                return tasks["steps"]
            if isinstance(tasks, list):
                return tasks
        except (json.JSONDecodeError, AttributeError):
            pass
    if op in DEFAULT_VERIFY_TASKS:
        return DEFAULT_VERIFY_TASKS[op]
    raise BridgeError(
        f"no verify tasks for op '{op}' in {vf} and no built-in default; "
        f"add a verify_tasks.json entry or pass verify=false"
    )


def _resolve_train_python() -> str:
    return (
        os.environ.get("KERNELAGENT_TRAIN_PYTHON")
        or os.environ.get("KERNELAGENT_PYTHON")
        or sys.executable
    )


def _derive_steps_from_train_script(
    train_script: Path, op: str, train_args: str | None, store: Path
) -> list[dict] | None:
    """P2: derive verify steps from an R6-generated injected training script.
    Returns None when no injected script exists yet (R6 not run or not applicable).
    """
    injected = train_script.with_suffix("").name + "_ka_injected.py"
    injected_path = train_script.parent / injected
    if not injected_path.is_file():
        return None
    steps: list[dict] = []
    # NOTE: The forward numeric_check step was intentionally removed. KernelAgent
    # already performs numerical-precision verification on each generated operator,
    # so the injector no longer replicates that check (avoids a redundant, and for
    # some ops incorrect, F.{op}-based reference comparison). Verification now
    # relies solely on the injected-script training step below.
    # train step using the injected script
    cmd = f'python "{injected_path}"'
    if train_args:
        cmd += f" {train_args}"
    steps.append({
        "name": "train",
        "cmd": cmd,
        "expect_contains": f"[ka-inject] {op} kernel calls =",
        "expect_kernel_calls_gt": 0,
    })
    return steps


def _resolve_verify_steps(
    payload: dict,
    store: Path,
    op: str,
    train_dir: Path | None,
    trace: TraceCollector | None,
) -> list[dict]:
    """R5 four-level priority for verify step generation.

    P1: verify_cmd (user-provided, highest)
    P2: train_script derived (R6-generated injected script)
    P3: verify_tasks.json convention file
    P4: built-in DEFAULT_VERIFY_TASKS (lowest)
    """
    tc = trace

    # P1: user-provided verify_cmd
    verify_cmd = payload.get("verify_cmd")
    if verify_cmd:
        if tc:
            tc.add("verify.step", f"P1 verify_cmd={verify_cmd!r}")
        if isinstance(verify_cmd, str):
            return [{"name": "user_cmd", "cmd": verify_cmd}]
        if isinstance(verify_cmd, list):
            return [{"name": f"user_cmd_{i}", "cmd": c} for i, c in enumerate(verify_cmd)]
        raise BridgeError(f"verify_cmd must be str or list, got {type(verify_cmd).__name__}")

    # P2: train_script derived
    train_script_raw = payload.get("train_script")
    if train_script_raw:
        train_script = Path(train_script_raw)
        train_args = payload.get("train_args")
        p2_steps = _derive_steps_from_train_script(train_script, op, train_args, store)
        if p2_steps:
            if tc:
                tc.add("verify.step", f"P2 train_script={train_script_raw!r} -> {len(p2_steps)} steps")
            return p2_steps
        if tc:
            tc.add("verify.step", f"P2 train_script={train_script_raw!r} -> no injected script yet, fallback")

    # P3: verify_tasks.json
    if train_dir and train_dir.is_dir():
        vf = train_dir / "verify_tasks.json"
        if vf.is_file():
            try:
                data = json.loads(vf.read_text(encoding="utf-8"))
                tasks = data.get(op)
                if isinstance(tasks, dict) and isinstance(tasks.get("steps"), list):
                    if tc:
                        tc.add("verify.step", f"P3 verify_tasks.json -> {len(tasks['steps'])} steps")
                    return tasks["steps"]
                if isinstance(tasks, list):
                    if tc:
                        tc.add("verify.step", f"P3 verify_tasks.json -> {len(tasks)} steps")
                    return tasks
            except (json.JSONDecodeError, AttributeError):
                pass

    # P4: built-in defaults
    if op in DEFAULT_VERIFY_TASKS:
        if tc:
            tc.add("verify.step", f"P4 DEFAULT_VERIFY_TASKS -> {len(DEFAULT_VERIFY_TASKS[op])} steps")
        return DEFAULT_VERIFY_TASKS[op]

    raise BridgeError(
        f"no verify tasks for op '{op}'; "
        f"pass verify_cmd, train_script, add verify_tasks.json, or pass verify=false"
    )


def do_verify(
    store: Path,
    op: str,
    train_dir: Path | None,
    payload: dict,
    trace: TraceCollector | None,
) -> dict:
    """Run the real training task(s) for the op and report integration success."""
    tasks = _resolve_verify_steps(payload, store, op, train_dir, trace)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{store}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["KERNELAGENT_RUNTIME_STORE"] = str(store)
    py = _resolve_train_python()

    step_results: list[dict] = []
    overall_pass = True
    for step in tasks:
        name = step.get("name", step.get("cmd", "?"))
        cmd = step["cmd"]
        # Use the configured training python instead of whatever "python" resolves to.
        if cmd.startswith("python "):
            cmd = f'"{py}" {cmd[len("python "):]}'

        # Determine working directory: if cmd references an absolute script path,
        # use that script's parent; otherwise fall back to train_dir.
        cwd = str(train_dir) if train_dir else str(store)
        m = re.search(r'\b"([^"]+\.py)"', cmd)
        if m:
            script_path = Path(m.group(1))
            if script_path.is_absolute() and script_path.parent.is_dir():
                cwd = str(script_path.parent)

        if trace:
            trace.add("verify.step", f"name={name} cmd={cmd!r} cwd={cwd}")

        try:
            r = subprocess.run(
                cmd, shell=True, cwd=cwd, env=env,
                capture_output=True, text=True, timeout=VERIFY_STEP_TIMEOUT_S,
            )
            out = (r.stdout or "") + "\n" + (r.stderr or "")
            exit_code = r.returncode
        except subprocess.TimeoutExpired:
            out = f"TIMEOUT after {VERIFY_STEP_TIMEOUT_S}s"
            exit_code = -1
        except Exception as e:  # noqa: BLE001
            out = f"verify step error: {e!r}"
            exit_code = -2

        # Parse kernel-call counter from output (proves the injected path was taken).
        kernel_calls: int | None = None
        m = KERNEL_CALLS_RE.search(out)
        if m and m.group(1) == op:
            kernel_calls = int(m.group(2))

        # P1 user command: also check verify_expect if provided
        expect_met = True
        if step.get("expect_contains"):
            expect_met = step["expect_contains"] in out
        else:
            verify_expect = payload.get("verify_expect")
            if verify_expect and isinstance(verify_expect, str):
                expect_met = verify_expect in out
        kc_ok = True
        if step.get("expect_kernel_calls_gt") is not None:
            kc_ok = kernel_calls is not None and kernel_calls > step["expect_kernel_calls_gt"]
        passed = exit_code == 0 and expect_met and kc_ok
        overall_pass = overall_pass and passed

        if trace:
            trace.add(
                "verify.step",
                f"name={name} -> exit={exit_code} expect_met={expect_met} kc={kernel_calls} kc_ok={kc_ok} passed={passed}"
            )

        step_results.append({
            "name": name,
            "cmd": cmd,
            "passed": passed,
            "exit_code": exit_code,
            "expect_met": expect_met,
            "kernel_calls": kernel_calls,
            "kc_ok": kc_ok,
            "tail": out[-1500:],
        })

    return {"overall_pass": overall_pass, "steps": step_results}


# --------------------------------------------------------------------------- #
# Deploy
# --------------------------------------------------------------------------- #

def do_deploy(payload: dict, working_dir: Path) -> dict:
    tc = TraceCollector()

    default_store = Path(
        os.environ.get("KERNELAGENT_RUNTIME_STORE") or str(working_dir / "runtime_store")
    )
    # R3: deploy_dir allows user-specified target store; defaults to default_store
    store = Path(payload.get("deploy_dir") or payload.get("store") or default_store)
    tc.add("deploy.dir", f"store={store} (default={default_store})")
    validate_store(store)
    tc.add("deploy.dir", "validate_store ok")

    # Bootstrap kernelagent_runtime package if target store lacks it
    bootstrapped = _bootstrap_runtime_package(store, default_store)
    if bootstrapped:
        tc.add("deploy.dir", "bootstrapped kernelagent_runtime package from default store")

    staging_root = store / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)

    ref = resolve_source(
        str(payload["source"]), working_dir, staging_root,
        allow_baseline=payload.get("allow_baseline", True),
        trace=tc
    )
    # Enforce the baseline guard for every source form, not just example names.
    if ref["kind"] == "baseline" and not payload.get("allow_baseline", True):
        tc.add("source.gate", "kind=baseline with allow_baseline=false -> refused")
        raise BridgeError(
            f"source resolved to a baseline bundle ({ref['best_bundle_dir']}) and "
            f"allow_baseline=false; deploy a real generation run instead"
        )
    bundle_dir = Path(ref["best_bundle_dir"])
    backend = ref["backend"]
    files = MUSA_FILES if backend == "musa" else TRITON_FILES

    # S7-1: synthesise/read metadata (Solution-A unified entry point) and
    # cross-check the extension name against kernel.py (A8/R3).
    run_dir_for_meta = Path(ref["run_dir"]) if ref.get("run_dir") else (
        Path(ref["best_bundle_dir"]).parent if ref.get("best_bundle_dir") else None
    )
    meta = _read_metadata(bundle_dir, run_dir=run_dir_for_meta)
    ref["meta"] = meta
    tc.add("meta.read", f"_read_metadata => op={meta.get('op')!r} backward_mode={meta.get('backward_mode')!r}")
    _cross_check_extension(meta, bundle_dir / "kernel.py", tc)

    # R5: skip CUDA-backend operators (still in development); abort before build.
    if _is_cuda_backend(bundle_dir, meta):
        msg = (
            "[ka-inject] CUDA backend operator detected — "
            "CUDA branch still in development (doing...); "
            "deployment aborted. Use a MUSA/triton bundle for now."
        )
        print(msg)
        return {
            "status": "skipped",
            "reason": "cuda_backend_not_supported",
            "message": msg,
            "op": meta.get("op"),
            "trace": tc.to_list(),
        }

    # Op name resolution (Solution-A priority): payload.op_name (1) >
    # metadata.op (2) > problem.py semantic parse (3).
    # target_op (priority 0) is accepted if present but not exposed by the
    # batch-1 TS tool.
    op_input = payload.get("op_name")
    if op_input:
        tc.add("opname.input", f"payload.op_name={op_input!r} (priority 1)")
    meta_op = (ref.get("meta") or {}).get("op") if isinstance(ref, dict) else None
    if not op_input and meta_op:
        tc.add("opname.meta", f"metadata.op={meta_op!r} (priority 2)")
        op = meta_op
    else:
        op = op_input or (derive_op_from_problem(Path(ref["problem_py"]), trace=tc)
                          if ref["problem_py"] else "")
    if not op:
        tc.add("opname.check", "op_name not provided and problem.py not found; op_name required")
        raise BridgeError(
            "could not derive op name (no op_name and no usable problem.py); "
            "pass op_name explicitly"
        )
    if not re.match(r"^[a-z][a-z0-9_]*$", op):
        raise BridgeError(f"invalid op_name: {op!r} (lowercase letters/digits/underscore only)")
    target = payload.get("target_op")
    if target:
        tc.add("opname.check", f"target={target!r} -> consistency check")
    else:
        tc.add("opname.check", "target=None -> consistency check skipped")
    check_op_target_consistency(op, target)
    tc.add("opname.check", f"op='{op}' aligned -> ok")

    schema = derive_schema(bundle_dir / "kernel.py", op, meta=meta)
    tc.add("schema.parse", f"schema => '{schema}' (meta_priority={bool(meta.get('schema'))})")
    dispatch_key = "PrivateUse1" if backend == "musa" else "CUDA"

    op_dir = store / "ops" / op
    want_build = payload.get("build", True)
    if op_dir.exists():
        if want_build:
            shutil.rmtree(op_dir)  # clean rebuild
            op_dir.mkdir(parents=True)
        # when build is skipped, keep the existing dir (incl. the built .so) so
        # re-verify can run without rebuilding
    else:
        op_dir.mkdir(parents=True)

    # S7-4: persist the resolved metadata for downstream consumers
    # (generate_injected_train_script / verify) under ops/<op>/_meta.json.
    (op_dir / "_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tc.add("meta.write", f"ops/{op}/_meta.json written")

    hashes: dict[str, str] = {}
    for f in files:
        shutil.copy2(bundle_dir / f, op_dir / f)
        hashes[f] = sha256_of(op_dir / f)
        tc.add("deploy.copy", f"{f} <- {bundle_dir / f} sha256={hashes[f][:16]}...")

    # R2: resolve separate-style backward wiring parameters for the integration
    # template. For autograd_function style the backward is embedded, so the extra
    # op/schema are left empty (the template only uses them inside the separate
    # branch). backward_schema is synthesised from kernel_backward's AST when the
    # bundle does not ship one (no cross-file change to the generator needed).
    _bwd = _get_backward_info(op, op_dir, meta)
    _bwd_style = _bwd.get("style", "autograd_function")
    _bwd_op = f"{op}_bwd"
    _bwd_entry = _bwd.get("backward_entry") or "kernel_backward"
    _bwd_schema = ""
    if _bwd_style == "separate":
        _bwd_schema = _derive_backward_schema(bundle_dir / "kernel.py", op, meta=meta)

    (op_dir / "torch_integration.py").write_text(
        TORCH_INTEGRATION_TEMPLATE.format(
            example=ref["example"] or "<file>",
            run_dir=ref["run_dir"] or "<file>",
            kind=ref["kind"],
            op=op,
            dispatch_key=dispatch_key,
            forward_entry=meta.get("forward_entry", "kernel_function"),
            schema=schema,
            dtype=_normalize_list(meta.get("dtype")),
            device=_normalize_list(meta.get("device")),
            backward_style=_bwd_style,
            backward_op=_bwd_op,
            backward_entry=_bwd_entry,
            backward_schema=_bwd_schema,
        ),
        encoding="utf-8",
    )

    build_info: dict = {"performed": False, "success": True, "ext_name": None, "log": None}
    if want_build and backend == "musa":
        build_info["performed"] = True
        build_info["ext_name"] = meta.get("extension_module") or parse_ext_name(op_dir / "setup.py")
        log_path = op_dir / "build.log"
        try:
            r = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=op_dir, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S,
            )
            log_path.write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
            build_info["success"] = r.returncode == 0
            build_info["log"] = f"ops/{op}/build.log"
            build_info["tail"] = (r.stderr or r.stdout)[-2000:]
            tc.add("deploy.build",
                   f"cmd='python setup.py build_ext --inplace', exit={r.returncode}, "
                   f"ext_name={build_info['ext_name']}")
        except subprocess.TimeoutExpired:
            log_path.write_text("TIMEOUT", encoding="utf-8")
            build_info["success"] = False
            build_info["log"] = f"ops/{op}/build.log"
            build_info["tail"] = f"build timed out after {BUILD_TIMEOUT_S}s"
            tc.add("deploy.build", f"TIMEOUT after {BUILD_TIMEOUT_S}s")

    warn = ""
    if ref["status"] not in ("SUCCESS", "UNKNOWN", "BASELINE"):
        warn = (
            f"\n> [warn] source status = {ref['status']}; the kernel may have no "
            f"gain, double-check benchmarks.\n"
        )
    if ref["kind"] == "baseline":
        warn += "\n> [note] baseline bundle deployed (unoptimized reference).\n"
    elif ref["kind"] == "local":
        warn += ("\n> [note] caller-supplied bundle (kind=local); provenance is not an "
                 "examples/<example>/results/run_* layout.\n")

    store_str = str(store)
    train_dir_env = os.environ.get("KERNELAGENT_TRAIN_DIR", "").strip()
    train_dir = Path(train_dir_env) if train_dir_env else None
    want_verify = payload.get("verify", True)

    # ---- R6: Generate injected training script if train_script provided ----
    injected_path: Path | None = None
    train_script_raw = payload.get("train_script")
    if train_script_raw:
        try:
            injected_path = generate_injected_train_script(
                Path(train_script_raw), op, trace=tc, meta=meta, schema=schema
            )
        except BridgeError:
            raise
        except Exception as e:
            raise BridgeError(f"R6 train_script generation failed: {e}") from e

    # Collect files written before verify so the runtime loader can find the op.
    files_written: list[dict] = []
    for f in files:
        files_written.append({
            "path": f"ops/{op}/{f}",
            "sha256": hashes[f],
            "source": str(bundle_dir / f),
        })
    files_written.append({
        "path": f"ops/{op}/torch_integration.py",
        "source": "<generated>",
    })
    if build_info.get("log"):
        files_written.append({
            "path": f"ops/{op}/build.log",
            "source": "<build>",
        })
    if injected_path and injected_path.is_file():
        files_written.append({
            "path": str(injected_path),
            "source": "<generated>",
            "action": "generated",
        })

    # Write a pre-verify index so kernelagent_runtime.enable() can find the op.
    verify_result: dict | None = None
    _write_manifests(
        store, op, ref, schema, hashes, build_info, None, files_written, payload, tc
    )

    # ---- Required flow: run the real training task to confirm integration ----
    verify_block = ""
    if not want_verify:
        verify_block = "\n> [skipped] verify=false; integration not executed.\n"
    elif not build_info["success"]:
        verify_block = "\n> [skipped] build failed; cannot verify integration.\n"
    elif injected_path and injected_path.is_file():
        try:
            verify_result = do_verify(store, op, None, payload, tc)
            lines = ["\n### integration verify\n"]
            for s in verify_result["steps"]:
                mark = "PASS" if s["passed"] else "FAIL"
                lines.append(f"- [{mark}] {s['name']}: `{s['cmd']}`")
                if s.get("kernel_calls") is not None:
                    lines.append(f"  - kernel calls = {s['kernel_calls']}")
                if not s["passed"]:
                    lines.append(f"  - exit={s['exit_code']} expect_met={s['expect_met']} "
                                 f"kc_ok={s['kc_ok']}")
                    lines.append("  ```\n" + s["tail"] + "\n  ```")
            overall = "PASS" if verify_result["overall_pass"] else "FAIL"
            lines.append(f"\n**integration verify: {overall}**\n")
            verify_block = "\n".join(lines) + "\n"
        except BridgeError as e:
            verify_result = {"overall_pass": False, "error": str(e)}
            verify_block = f"\n> [verify error] {e}\n"
    elif train_dir and train_dir.is_dir():
        try:
            verify_result = do_verify(store, op, train_dir, payload, tc)
            lines = ["\n### integration verify\n"]
            for s in verify_result["steps"]:
                mark = "PASS" if s["passed"] else "FAIL"
                lines.append(f"- [{mark}] {s['name']}: `{s['cmd']}`")
                if s.get("kernel_calls") is not None:
                    lines.append(f"  - kernel calls = {s['kernel_calls']}")
                if not s["passed"]:
                    lines.append(f"  - exit={s['exit_code']} expect_met={s['expect_met']} "
                                 f"kc_ok={s['kc_ok']}")
                    lines.append("  ```\n" + s["tail"] + "\n  ```")
            overall = "PASS" if verify_result["overall_pass"] else "FAIL"
            lines.append(f"\n**integration verify: {overall}**\n")
            verify_block = "\n".join(lines) + "\n"
        except BridgeError as e:
            verify_result = {"overall_pass": False, "error": str(e)}
            verify_block = f"\n> [verify error] {e}\n"
    else:
        verify_block = (
            "\n> [skipped] KERNELAGENT_TRAIN_DIR not set or missing, and no train_script provided; "
            "cannot run the real training task. Set KERNELAGENT_TRAIN_DIR or pass train_script.\n"
        )

    # Rewrite three-layer manifests with the verify result included.
    manifest_files = _write_manifests(
        store, op, ref, schema, hashes, build_info, verify_result, files_written, payload, tc
    )
    for mf in manifest_files:
        files_written.append(mf)

    # Build run_block and activation_hint
    if injected_path and injected_path.is_file():
        train_args = payload.get("train_args", "")
        cmd = f'python "{injected_path}"'
        if train_args:
            cmd += f" {train_args}"
        run_block = (
            "```bash\n"
            f"export KERNELAGENT_RUNTIME_STORE={store_str}\n"
            f"export PYTHONPATH={store_str}:$PYTHONPATH\n"
            f"{cmd}\n"
            "```\n"
        )
        activation_hint = (
            f"export KERNELAGENT_RUNTIME_STORE={store_str} "
            f"PYTHONPATH={store_str}:$PYTHONPATH && {cmd}"
        )
    elif train_dir and train_dir.is_dir():
        run_block = (
            "```bash\n"
            f"cd {train_dir}\n"
            f"export PYTHONPATH={store_str}:$PYTHONPATH\n"
            f"python test_relu_smoke.py        # T1: numeric check vs F.relu\n"
            f"python test_relu_forward_only.py  # T2-F: forward-only safety\n"
            f"python train_mnist_musa_injected.py --epochs 1  # T2-B: full training\n"
            "```\n"
        )
        activation_hint = (
            f"cd {train_dir} && export PYTHONPATH={store_str}:$PYTHONPATH && "
            f"python train_mnist_musa_injected.py --epochs 1"
        )
    else:
        run_block = (
            "```bash\n"
            f"export PYTHONPATH={store_str}:$PYTHONPATH\n"
            f'python -c "import kernelagent_runtime, torch; print(torch.ops.kernelagent.{op})"\n'
            "```\n"
        )
        activation_hint = f"export PYTHONPATH={store_str}:$PYTHONPATH; import kernelagent_runtime"

    verified = bool(verify_result and verify_result.get("overall_pass"))

    # Build files-written section for report
    files_block_lines = ["\n### files written this run\n"]
    for fw in files_written:
        action = fw.get("action", "written")
        path = fw["path"]
        files_block_lines.append(f"- `{path}`  ({action})")
    files_block = "\n".join(files_block_lines) + "\n"

    report = (
        "## kernelagent-inject deploy report\n\n"
        f"- op: `kernelagent::{op}` (backend={backend}, dispatch={dispatch_key})\n"
        f"- source kind: `{ref['kind']}`"
        + (f"  example: `{ref['example']}`" if ref["example"] else "") + "\n"
        f"- run status: `{ref['status']}`\n"
        f"- deploy dir: `{op_dir}`\n"
        f"- build: {'ok' if build_info['success'] else 'FAILED (see build.log)'}"
        + (f", ext_name=`{build_info['ext_name']}`" if build_info.get("ext_name") else "") + "\n"
        f"- schema: `{schema}`\n"
        f"- integration verified: `{'yes' if verified else 'no'}`\n{warn}\n"
        f"{verify_block}\n"
        f"{files_block}\n"
        "Activate / verify in the training environment:\n\n"
        f"{run_block}"
    )

    return {
        "success": bool(build_info["success"]),
        "verified": verified,
        "op": op,
        "op_dir": str(op_dir),
        "backend": backend,
        "schema": schema,
        "kind": ref["kind"],
        "build": build_info,
        "verify": verify_result,
        "files_written": files_written,
        "trace": tc.to_list(),
        "report": report,
        "activation_hint": activation_hint,
    }


# --------------------------------------------------------------------------- #
# Three-layer manifest (R2): L3 history -> L2 per-op meta -> L1 global index
# --------------------------------------------------------------------------- #

def _write_manifests(
    store: Path,
    op: str,
    ref: dict,
    schema: str,
    hashes: dict[str, str],
    build_info: dict,
    verify_result: dict | None,
    files_written: list[dict],
    payload: dict,
    trace: TraceCollector,
) -> list[dict]:
    """Write L3 history (new), L2 per-op meta (overwrite), L1 global index (merge).
    Returns the list of written files for the report.
    """
    now = datetime.now(timezone.utc).astimezone()
    ts = now.strftime("%Y%m%d_%H%M%S")
    written: list[dict] = []

    # --- L3: immutable trigger history ---
    history_dir = store / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    l3_path = history_dir / f"inj_{ts}_{op}.json"
    l3 = {
        "schema_version": 2,
        "trigger_at": now.isoformat(timespec="seconds"),
        "op": op,
        "input": {k: v for k, v in payload.items() if k != "store"},
        "resolved": ref,
        "trace": trace.to_list(),
        "schema": schema,
        "bundle_sha256": hashes,
        "build": build_info,
        "verify": verify_result,
        "files_written": files_written,
    }
    l3_path.write_text(json.dumps(l3, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append({"path": str(l3_path.relative_to(store)), "action": "new"})

    # --- L2: per-op metadata (overwrite) ---
    meta_dir = store / "ops" / op
    meta_dir.mkdir(parents=True, exist_ok=True)
    l2_path = meta_dir / "_meta.json"
    l2 = {
        "schema_version": 2,
        "updated_at": now.isoformat(timespec="seconds"),
        "op": op,
        "schema": schema,
        "dispatch_key": "PrivateUse1" if ref["backend"] == "musa" else "CUDA",
        "bundle_sha256": hashes,
        "source": ref,
        "build": build_info,
        "verify": verify_result,
        "files_in_op_dir": [f.name for f in meta_dir.iterdir() if f.is_file()],
    }
    l2_path.write_text(json.dumps(l2, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append({"path": str(l2_path.relative_to(store)), "action": "updated"})

    # --- L1: global index (merge) ---
    l1_path = store / "manifest.json"
    manifest: dict = {"schema_version": 2, "ops": {}}
    if l1_path.is_file():
        try:
            manifest = json.loads(l1_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {"schema_version": 2, "ops": {}}
    manifest.setdefault("ops", {})[op] = {
        "backend": ref["backend"],
        "source": {
            "kind": ref["kind"],
            "example": ref["example"],
            "run_dir": ref["run_dir"],
            "best_bundle_dir": ref["best_bundle_dir"],
            "status": ref["status"],
            "problem_py": ref["problem_py"],
        },
        "schema": schema,
        "bundle_sha256": hashes,
        "build": build_info,
        "verify_summary": {
            "overall_pass": verify_result.get("overall_pass") if verify_result else None,
            "step_count": len(verify_result.get("steps", [])) if verify_result else 0,
        } if verify_result else None,
        "latest_history": str(l3_path.relative_to(store)),
        "updated_at": now.isoformat(timespec="seconds"),
    }
    l1_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append({"path": str(l1_path.relative_to(store)), "action": "updated"})

    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="kernelagent-injector bridge (batch-1)")
    parser.add_argument("--mode", choices=["deploy"], required=True)
    parser.add_argument("--input", required=True, help="payload JSON file path")
    parser.add_argument("--output", help="output JSON file path (default stdout)")
    args = parser.parse_args()

    working_dir_env = os.environ.get("KERNELAGENT_WORKING_DIR")
    if not working_dir_env:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": "KERNELAGENT_WORKING_DIR env var is not set "
                             "(should point to the KernelAgent project root with examples/)",
                },
                indent=2,
            )
        )
        return 1
    working_dir = Path(working_dir_env)

    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = do_deploy(payload, working_dir)
    except BridgeError as e:
        result = {"success": False, "error": str(e), "report": f"## deploy failed\n\n{e}"}
    except Exception as e:  # noqa: BLE001
        result = {"success": False, "error": f"unexpected: {e!r}"}

    out = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
