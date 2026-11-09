"""Fallback runner when pytest is unavailable (tests are plain pytest-style)."""

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError("did not raise")
        return issubclass(t, self.exc)


class _Pytest:  # minimal shim
    raises = staticmethod(lambda e: _Raises(e))


sys.modules.setdefault("pytest", _Pytest)

ok = fail = 0
for f in sorted(Path(__file__).parent.rglob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for name, fn in inspect.getmembers(m, inspect.isfunction):
        if name.startswith("test_"):
            try:
                fn(*([None] if "monkeypatch" in inspect.signature(fn).parameters else []))
                ok += 1
                print("PASS", f.stem, name)
            except Exception:
                fail += 1
                print("FAIL", f.stem, name)
                traceback.print_exc()
print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
