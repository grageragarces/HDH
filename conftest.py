"""Root pytest configuration.

`pytest.ini` also collects the docstring examples under `hdh/` as doctests.
Converters for optional SDKs can't be imported without them, so their
modules are left out of collection when the SDK is missing, the same way
their tests in `tests/` skip.
"""
import importlib.util

_OPTIONAL = {
    "cirq": "hdh/converters/cirq_converter.py",
    "pennylane": "hdh/converters/pennylane_converter.py",
    "braket": "hdh/converters/braket_converter.py",
}

# Scripts under tests/ that aren't test modules. Doctest collection would
# import them, and private_test.py plots (writing hdh_plot.svg) on import.
collect_ignore = ["tests/private_test.py", "tests/diagnose.py"]

collect_ignore += [
    path for module, path in _OPTIONAL.items()
    if importlib.util.find_spec(module) is None
]
