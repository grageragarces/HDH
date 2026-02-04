""" TWO POSSIBLE USAGE MODES:
    
    1) Append mode (default): scan existing CSV once, skip already-done tests -- good for adding a method, slow if working in parallel from scratch
    ----------------------------------------------------------------------------
# first time, or "just pick up where I left off"
python -m database_generators.run_partition_tests \
  --database-root ../database \
  --methods greedy_hdh metis_telegate kahypar \
  --k-values 2 3 4 5 6 \
  --overhead 0.0 \
  --workers 10 \  # will depend on your available cores
  --contributor “INSERT GITHUB USERNAME” \
  --library-version "INSERT LATEST “V

    2) Scratch mode: back up existing CSV, start fresh, run every test from - for starting the database from scratch
    ----------------------------------------------------------------------------
# nuke and redo everything from zero
python -m database_generators.run_partition_tests \
  --database-root ../database \
  --methods greedy_hdh metis_telegate kahypar \
  --k-values 2 3 4 5 6 \
  --overhead 0.0 \
  --workers 10 \ # will depend on your available cores
  --contributor “INSERT GITHUB USERNAME” \
  --library-version "INSERT LATEST “V
  --mode scratch
    """

import argparse
import json
import pickle
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import sys
import csv
import hashlib
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm
import re

# CSV field size limit 
_maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(_maxInt)
        break
    except OverflowError:
        _maxInt = int(_maxInt / 10)

FIELDNAMES = [
    'workload_file', 'model', 'origin', 'n_qubits', 'k_partitions', 'capacity',
    'method_name', 'method_version', 'config_hash',
    'bins', 'cost', 'respects_capacity', 'method_metadata',
    'time_seconds', 'memory_mb',
    'date_run', 'library_version', 'contributor', 'notes',
]

def hash_config(params: Dict[str, Any]) -> str:
    """Deterministic short hash of a config dict (or 'default' when empty)."""
    if not params:
        return "default"
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def count_unique_qubits(bin_nodes: List[str]) -> int:
    """Count distinct qubit indices in a bin, ignoring time-step suffixes."""
    qubits: set = set()
    for node in bin_nodes:
        m = re.match(r"^q(\d+)_t\d+$", node)
        if m:
            qubits.add(int(m.group(1)))
    return len(qubits)


def calculate_capacity(n_qubits: int, k_partitions: int, overhead: float = 0.0) -> int:
    """Ceiling-division capacity with an optional overhead multiplier."""
    base = (n_qubits + k_partitions - 1) // k_partitions
    return int(base * (1.0 + overhead))


def _dedup_key(workload_file: str, k: int, method_name: str, cfg_hash: str) -> tuple:
    """The tuple we use to decide 'have we already run this exact config?'"""
    return (workload_file, k, method_name, cfg_hash)


def load_existing_keys(results_csv: Path) -> set:
    """
    Read the CSV *once* at startup and return the set of de-dup keys already
    present.  This is the only full CSV scan the process will ever do.
    """
    keys: set = set()
    if not results_csv.exists():
        return keys
    try:
        with open(results_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                keys.add(_dedup_key(
                    row['workload_file'],
                    int(row['k_partitions']),
                    row['method_name'],
                    row['config_hash'],
                ))
    except Exception as e:
        print(f"Warning: could not read {results_csv} ({e}). "
              "Treating DB as empty — duplicates may be appended.")
    return keys


def _ensure_header(results_csv: Path):
    """Create the file with a header row if it does not exist yet."""
    ensure_dir(results_csv.parent)
    if not results_csv.exists():
        with open(results_csv, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def append_result(results_csv: Path, entry: Dict, lock) -> None:
    """Append a single row to the CSV.  Thread-safe via the supplied lock.
    No read, no rewrite — just an open-in-append + writerow."""
    with lock:
        # Edge case: file was deleted between startup and now
        if not results_csv.exists():
            _ensure_header(results_csv)
        with open(results_csv, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(entry)


def build_csv_entry(result: Dict) -> Dict:
    """Flatten a result dict into the row shape expected by FIELDNAMES."""
    return {
        'workload_file':     result['workload_file'],
        'model':             result['model'],
        'origin':            result['origin'],
        'n_qubits':          result['n_qubits'],
        'k_partitions':      result['k_partitions'],
        'capacity':          result['capacity'],
        'method_name':       result['method_name'],
        'method_version':    result['method_version'],
        'config_hash':       hash_config(result.get('config_params')),
        'bins':              json.dumps(result['bins']),
        'cost':              result['cost'],
        'respects_capacity': str(result['respects_capacity']).lower(),
        'method_metadata':   json.dumps(result.get('method_metadata') or {}),
        'time_seconds':      f"{result['time_seconds']:.6f}" if result.get('time_seconds') is not None else '',
        'memory_mb':         f"{result['memory_mb']:.2f}"    if result.get('memory_mb')    is not None else '',
        'date_run':          datetime.now().isoformat()[:10],
        'library_version':   result.get('library_version', 'unknown'),
        'contributor':       result.get('contributor', 'unknown'),
        'notes':             result.get('notes', ''),
    }


# HDH loading 

def load_hdh(hdh_path: Path):
    with open(hdh_path, 'rb') as f:
        return pickle.load(f)

def get_n_qubits(hdh_graph) -> int:
    qubits: set = set()
    for node in hdh_graph.S:
        m = re.match(r"^q(\d+)_t\d+$", node)
        if m:
            qubits.add(int(m.group(1)))
    return len(qubits)


# Partitioning methods
class MethodSkipError(Exception):
    """Raised at runtime when a method cannot handle the given parameters.
    The worker catches this and records a clean skip instead of a crash."""
    pass


def run_greedy_hdh(hdh_graph, k: int, capacity: int, config: Dict) -> Tuple[List, int, bool, Dict]:
    from hdh.passes.cut import compute_cut

    beam_k       = config.get('beam_k', 3)
    reserve_frac = config.get('reserve_frac', 0.08)
    restarts     = config.get('restarts', 1)
    seed         = config.get('seed', 0)

    bins, cost = compute_cut(
        hdh_graph,
        k=k,
        cap=capacity,
        beam_k=beam_k,
        reserve_frac=reserve_frac,
        restarts=restarts,
        seed=seed,
    )

    bins_list         = [sorted(list(b)) for b in bins]
    respects_capacity = all(count_unique_qubits(b) <= capacity for b in bins_list)

    metadata = {
        'beam_k': beam_k, 'reserve_frac': reserve_frac,
        'restarts': restarts, 'seed': seed,
    }
    return bins_list, cost, respects_capacity, metadata


def run_metis_telegate(hdh_graph, k: int, capacity: int, config: Dict) -> Tuple[List, int, bool, Dict]:
    from hdh.passes.cut import metis_telegate

    bins_qubits, cost, respects_capacity, method = metis_telegate(
        hdh_graph, partitions=k, capacities=capacity
    )

    bins_list = [sorted(list(b)) for b in bins_qubits]
    metadata  = {'metis_method': method, 'failed': not respects_capacity}
    return bins_list, cost, respects_capacity, metadata


def run_kahypar(hdh_graph, k: int, capacity: int, config: Dict) -> Tuple[List, int, bool, Dict]:
    """
    Run KaHyPar via the hdh wrapper.

    ``kahypar_cutter`` only supports k == 2.  Tasks with other k values are
    filtered out *before* they reach this function (see ``supported_k`` in the
    METHODS registry), but we keep a runtime guard here as a safety net.
    """
    if k != 2:
        raise MethodSkipError(
            f"kahypar_cutter only supports k=2 (got k={k})"
        )

    from hdh.passes.cut import kahypar_cutter

    seed            = config.get('seed', 0)
    config_path     = config.get('config_path', None)
    suppress_output = config.get('suppress_output', True)

    bins, cost = kahypar_cutter(
        hdh_graph, k=k, cap=capacity,
        seed=seed, config_path=config_path, suppress_output=suppress_output,
    )

    bins_list         = [sorted(list(b)) for b in bins]
    respects_capacity = all(count_unique_qubits(b) <= capacity for b in bins_list)

    metadata = {
        'seed': seed,
        'config_path': config_path or 'default',
        'method': 'kahypar_qubit_level',
    }
    return bins_list, cost, respects_capacity, metadata


# ``supported_k``: a set of ints the method can handle, or *None* for "any k".
# Tasks with k not in this set are dropped at task-build time — nothing is
# even spawned to the pool.
METHODS: Dict[str, Dict[str, Any]] = {
    'greedy_hdh': {
        'function':       run_greedy_hdh,
        'default_config': {'beam_k': 3, 'reserve_frac': 0.08, 'restarts': 1, 'seed': 0},
        'supported_k':    None,   # all k
    },
    'metis_telegate': {
        'function':       run_metis_telegate,
        'default_config': {},
        'supported_k':    None,   # all k
    },
    'kahypar': {
        'function':       run_kahypar,
        'default_config': {'seed': 0, 'config_path': None, 'suppress_output': True},
        'supported_k':    {2},    # only bisection
    },
}

# File discovery

def find_hdh_files(
    database_root: Path,
    model: Optional[str] = None,
    origin: Optional[str] = None,
    max_qubits: Optional[int] = None,
) -> List[Dict]:
    hdhs_dir = database_root / "HDHs"
    if not hdhs_dir.exists():
        print(f"Error: HDHs directory not found: {hdhs_dir}")
        return []

    if model and origin:
        search_path = hdhs_dir / model / origin / "pkl"
        pkl_files   = list(search_path.glob("*.pkl")) if search_path.exists() else []
    elif model:
        pkl_files = list((hdhs_dir / model).rglob("*.pkl"))
    else:
        pkl_files = list(hdhs_dir.rglob("*.pkl"))

    found: List[Dict] = []
    for pkl_path in pkl_files:
        parts       = pkl_path.relative_to(hdhs_dir).parts
        file_model  = parts[0]
        file_origin = parts[1] if len(parts) > 1 else "Unknown"

        if max_qubits is not None:
            try:
                n = get_n_qubits(load_hdh(pkl_path))
                if n > max_qubits:
                    continue
            except Exception as e:
                print(f"Warning: could not load {pkl_path}: {e}")
                continue

        found.append({
            'hdh_path':      pkl_path,
            'workload_file': pkl_path.stem + ".qasm",
            'model':         file_model,
            'origin':        file_origin,
        })
    return found


# Test runner  (called inside onr worker process)
def run_test(
    hdh_path: Path,
    workload_file: str,
    model: str,
    origin: str,
    k: int,
    overhead: float,
    method_name: str,
    method_config: Dict,
    library_version: str,
    contributor: str,
) -> Dict:
    hdh_graph = load_hdh(hdh_path)
    n_qubits  = get_n_qubits(hdh_graph)
    capacity  = calculate_capacity(n_qubits, k, overhead)

    method_info = METHODS[method_name]
    config      = method_info['default_config'].copy()
    config.update(method_config)

    tracemalloc.start()
    t0 = time.time()

    bins, cost, respects_capacity, metadata = method_info['function'](
        hdh_graph, k, capacity, config
    )

    elapsed   = time.time() - t0
    _, peak   = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metadata['overhead']      = overhead
    metadata['base_capacity'] = (n_qubits + k - 1) // k

    # config_params now includes overhead so that different overhead values
    # produce distinct config_hashes and are stored as separate rows.
    config_params = {**config, 'overhead': overhead}

    return {
        'workload_file':     workload_file,
        'model':             model,
        'origin':            origin,
        'n_qubits':          n_qubits,
        'k_partitions':      k,
        'capacity':          capacity,
        'method_name':       method_name,
        'method_version':    '1.0.0',
        'bins':              bins,
        'cost':              cost,
        'respects_capacity': respects_capacity,
        'method_metadata':   metadata,
        'time_seconds':      elapsed,
        'memory_mb':         peak / (1024 * 1024),
        'config_params':     config_params,
        'library_version':   library_version,
        'contributor':       contributor,
    }


# Worker  (multiprocessing)

def run_test_worker(task: Dict) -> Tuple[str, Optional[Dict], str]:
    """
    Returns (status, result_or_None, message).
    status ∈ {'ok', 'skip', 'error'}
    """
    try:
        result = run_test(
            hdh_path=task['hdh_path'],
            workload_file=task['workload_file'],
            model=task['model'],
            origin=task['origin'],
            k=task['k'],
            overhead=task['overhead'],
            method_name=task['method_name'],
            method_config=task['method_config'],
            library_version=task['library_version'],
            contributor=task['contributor'],
        )
        return ('ok', result, "")

    except MethodSkipError as e:
        # Runtime safety-net skip (should rarely fire thanks to pre-filter)
        return ('skip', None, str(e))

    except Exception as e:
        import traceback
        return ('error', None, f"{e}\n{traceback.format_exc()}")


# CLI
def main():
    parser = argparse.ArgumentParser(
        description='Run automated partitioning tests on HDH database (PARALLEL | modes: append / scratch)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--database-root', type=Path, required=True,
                        help='Root directory of the database')

    # Filtering
    parser.add_argument('--model',      help='Filter by model (e.g. Circuit)')
    parser.add_argument('--origin',     help='Filter by origin (e.g. MQTBench)')
    parser.add_argument('--max-qubits', type=int,
                        help='Only test circuits with <= this many qubits')

    # Test parameters
    parser.add_argument('--methods', nargs='+', required=True,
                        choices=list(METHODS.keys()),
                        help='Partitioning methods to test')
    parser.add_argument('--k-values', nargs='+', type=int, required=True,
                        help='Number of partitions (e.g. 2 3 4 5 6)')
    parser.add_argument('--overhead', nargs='+', type=float, default=[0.0],
                        help='Capacity overhead fraction(s). Can pass multiple.')

    # Configuration
    parser.add_argument('--config', type=str,
                        help='JSON string with method config')
    parser.add_argument('--library-version', default='0.2.1')
    parser.add_argument('--contributor',     default='unknown')

    # Parallel
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel workers (default: CPU count − 2)')

    # Mode
    parser.add_argument('--mode', choices=['append', 'scratch'], default='append',
                        help=(
                            'append  – scan the existing CSV once, skip anything '
                            'already there, and append new rows (safe to re-run). '
                            'scratch – back up the current CSV, start a fresh one, '
                            'and run every task from zero.'
                        ))

    # Behaviour flags
    parser.add_argument('--dry-run',     action='store_true',
                        help='Show what would run without executing')
    parser.add_argument('--output-json', type=Path,
                        help='Write results to a JSON file instead of the DB CSV')

    args = parser.parse_args()

    # ---- workers ----
    if args.workers is None:
        args.workers = max(1, mp.cpu_count() - 2)
    print(f"Using {args.workers} parallel workers")

    # ---- parse user config ----
    method_config: Dict = {}
    if args.config:
        try:
            method_config = json.loads(args.config)
        except json.JSONDecodeError:
            print(f"Error: invalid JSON in --config: {args.config}")
            sys.exit(1)

    # ---- discover HDH files ----
    print("Finding HDH files …")
    hdh_files = find_hdh_files(
        args.database_root,
        model=args.model,
        origin=args.origin,
        max_qubits=args.max_qubits,
    )
    if not hdh_files:
        print("No HDH files found!")
        sys.exit(1)
    print(f"Found {len(hdh_files)} HDH files")

    results_csv = args.database_root / "Partitions" / "results" / "partitions.csv"

    # ---- mode: scratch  →  back up the old CSV and start fresh -------------
    # ---- mode: append   →  one-time scan so we can skip already-done tasks --
    existing_keys: set = set()

    if args.mode == 'scratch' and not args.output_json:
        import shutil
        ensure_dir(results_csv.parent)
        if results_csv.exists():
            timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = results_csv.with_suffix(f".csv.backup_{timestamp}")
            shutil.copy(results_csv, backup_path)
            print(f"[scratch] Backed up existing CSV → {backup_path}")
        # Write a fresh header — overwrites whatever was there
        with open(results_csv, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        print("[scratch] Wiped CSV. Every task will run from zero.")
        # existing_keys stays empty — nothing will be skipped

    elif args.mode == 'append' and not args.output_json:
        print("Scanning existing results (one-time) …")
        existing_keys = load_existing_keys(results_csv)
        print(f"  {len(existing_keys)} result(s) already on disk — will be skipped.")

    # ---- build task list, dropping already-done and unsupported-k tasks -----
    tasks:          List[Dict] = []
    skipped_done:   int        = 0   # already in DB
    skipped_k:      int        = 0   # method doesn't support this k

    for hdh_info in hdh_files:
        for method_name in args.methods:
            supported_k = METHODS[method_name].get('supported_k')   # None = any

            # Merge the method's default config with user overrides so we can
            # compute the hash for dedup *before* spawning any workers.
            merged_base = METHODS[method_name]['default_config'].copy()
            merged_base.update(method_config)

            for k in args.k_values:
                # --- drop unsupported k values for this method ---
                if supported_k is not None and k not in supported_k:
                    skipped_k += len(args.overhead)
                    continue

                for overhead in args.overhead:
                    # overhead is part of the hash so distinct overheads are
                    # stored as separate rows (fixes silent collision in the
                    # original code).
                    cfg_hash = hash_config({**merged_base, 'overhead': overhead})
                    key      = _dedup_key(hdh_info['workload_file'], k, method_name, cfg_hash)

                    if key in existing_keys:
                        skipped_done += 1
                        continue

                    tasks.append({
                        'hdh_path':        hdh_info['hdh_path'],
                        'workload_file':   hdh_info['workload_file'],
                        'model':           hdh_info['model'],
                        'origin':          hdh_info['origin'],
                        'k':               k,
                        'overhead':        overhead,
                        'method_name':     method_name,
                        'method_config':   method_config,
                        'library_version': args.library_version,
                        'contributor':     args.contributor,
                    })

    total_tests = len(tasks)

    # ---- summary ------------------------------------------------
    print(f"\nTask summary  (mode={args.mode}):")
    print(f"  Queued to run              : {total_tests}")
    if skipped_done:
        print(f"  Skipped (already in DB)    : {skipped_done}")
    if skipped_k:
        print(f"  Skipped (unsupported k)    : {skipped_k}")
        # Tell the user which method(s) caused it
        for mname in args.methods:
            sk = METHODS[mname].get('supported_k')
            if sk is not None:
                unsupported = [k for k in args.k_values if k not in sk]
                if unsupported:
                    print(f"      → {mname} only supports k ∈ {sorted(sk)}, "
                          f"skipped k = {unsupported}")

    # ---- dry run ------------------------------------------------------------
    if args.dry_run:
        print("\n[DRY RUN] first 10 queued tasks:")
        for t in tasks[:10]:
            print(f"  {t['model']}/{t['origin']}/{t['workload_file']}  "
                  f"k={t['k']}  overhead={t['overhead']}  method={t['method_name']}")
        if total_tests > 10:
            print(f"  … and {total_tests - 10} more")
        sys.exit(0)

    if total_tests == 0:
        print("\nNothing to do.  Use --mode scratch to re-run everything from zero.")
        sys.exit(0)

    # ---- ensure the CSV exists with a header before workers start -----------
    if not args.output_json:
        _ensure_header(results_csv)

    # ---- run ----------------------------------------------------------------
    print(f"\nRunning {total_tests} tests …")

    lock        = mp.Manager().Lock()
    all_results: List[Dict] = []
    completed   = 0
    failed      = 0
    skipped_rt  = 0   # runtime skips (MethodSkipError safety-net)

    with mp.Pool(processes=args.workers) as pool:
        for status, result, msg in tqdm(
            pool.imap_unordered(run_test_worker, tasks),
            total=total_tests, desc="Processing", unit="test",
        ):
            if status == 'ok' and result:
                all_results.append(result)
                if not args.output_json:
                    try:
                        append_result(results_csv, build_csv_entry(result), lock)
                        completed += 1
                    except Exception as e:
                        tqdm.write(f"[SAVE ERROR] {e}")
                        failed += 1
                else:
                    completed += 1

            elif status == 'skip':
                skipped_rt += 1
                tqdm.write(f"[SKIP] {msg}")

            else:   # 'error'
                failed += 1
                tqdm.write(f"[ERROR] {msg}")

    # ---- optional JSON dump -------------------------------------------------
    if args.output_json:
        print(f"\nSaving {len(all_results)} results → {args.output_json}")
        with open(args.output_json, 'w') as f:
            json.dump({'results': all_results}, f, indent=2)

    # ---- final summary ------------------------------------------------------
    print("\n" + "=" * 60)
    print("Tests complete:")
    print(f"  Completed          : {completed}")
    print(f"  Failed             : {failed}")
    print(f"  Skipped (pre-run)  : {skipped_done + skipped_k}")
    print(f"  Skipped (runtime)  : {skipped_rt}")
    if total_tests > 0:
        print(f"  Success rate       : {100 * completed / total_tests:.1f} %")

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()