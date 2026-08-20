# Contributing to HDH

Thanks for your interest in HDH. This document covers how to contribute code,
report problems, and get help.

## Reporting issues

Please open an issue at https://github.com/grageragarces/HDH/issues. For bugs,
include:

- the HDH version (`pip show hdh`) and Python version,
- which optional extras are installed (`cirq`, `pennylane`, `braket`, `kahypar`, `metis`),
- a minimal snippet that reproduces the problem, and
- the full traceback or the incorrect output you observed.

## Seeking support

Usage questions are welcome as GitHub issues, or via GitHub Discussions if you
would rather not file a bug. The documentation at
https://grageragarces.github.io/HDH/ covers installation, the model-to-HDH
mappings, visualization, and the partitioning passes, and the API reference is
generated from the library's docstrings.

## Contributing code

1. Fork the repository and create a branch off `main`.
2. Install the development environment:

   ```bash
   pip install -e ".[all]"
   pip install pytest pytest-cov
   ```

   The `cirq`, `pennylane`, and `braket` extras require Python >= 3.11. On
   Python 3.10 the corresponding converters are unavailable and their tests
   skip.

3. Add tests under `tests/` for any behaviour you change. New converters or
   partitioning passes should come with tests covering both the expected
   output and the failure modes (unsupported gates, capacity violations).
4. Run the suite before opening a pull request:

   ```bash
   pytest
   pytest --cov=hdh --cov-report=term-missing   # optional, to check coverage
   ```

5. Open a pull request describing what changed and why. CI runs the test suite
   on Python 3.10, 3.11, and 3.12.

Areas where contributions are particularly welcome:

- additional SDK conversions, especially HDH -> SDK directions
  (`to_cirq`, `to_braket`),
- partitioning heuristics built on the HDH data structure,
- benchmark workloads for the non-circuit models (MBQC, quantum walks, QCA),
- visualization and frontend tooling.

## Code of conduct

Participation in this project is governed by the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Please report unacceptable behaviour to the maintainer at the address listed on
the repository.
