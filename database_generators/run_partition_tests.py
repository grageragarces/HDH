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
from functools import partial
from tqdm import tqdm

# CSV field size limit error
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

# Import add_result function inline to avoid module issues
def hash_config(params: Dict[str, Any]) -> str:
    """Generate hash for configuration parameters."""
    if not params:
        return "default"
    config_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def ensure_dir(p: Path):
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def add_result_safe(
    results_csv: Path,
    result: Dict,
    lock: mp.Lock,
    allow_overwrite: bool = False
) -> bool:
    """Thread-safe version of add_result."""
    with lock:
        return add_result(
            results_csv=results_csv,
            workload_file=result['workload_file'],
            model=result['model'],
            origin=result['origin'],
            n_qubits=result['n_qubits'],
            k_partitions=result['k_partitions'],
            capacity=result['capacity'],
            method_name=result['method_name'],
            method_version=result['method_version'],
            bins=result['bins'],
            cost=result['cost'],
            respects_capacity=result['respects_capacity'],
            method_metadata=result['method_metadata'],
            time_seconds=result['time_seconds'],
            memory_mb=result['memory_mb'],
            config_params=result.get('config_params'),  # ADD THIS
            library_version=result['library_version'],
            contributor=result['contributor'],
            notes=result.get('notes', ''),  # ADD THIS
            allow_overwrite=allow_overwrite
        )

def add_result(
    results_csv: Path,
    workload_file: str,
    model: str,
    origin: str,
    n_qubits: int,
    k_partitions: int,
    capacity: int,
    method_name: str,
    method_version: str,
    bins: List[List[str]],
    cost: int,
    respects_capacity: bool = True,
    method_metadata: Optional[Dict] = None,
    time_seconds: Optional[float] = None,
    memory_mb: Optional[float] = None,
    config_params: Optional[Dict] = None,
    library_version: str = "unknown",
    contributor: str = "unknown",
    notes: str = "",
    allow_overwrite: bool = False
) -> bool:
    """Add a single result to the database."""
    
    # Generate config hash
    config_hash = hash_config(config_params)
    
    # Build entry
    entry = {
        'workload_file': workload_file,
        'model': model,
        'origin': origin,
        'n_qubits': n_qubits,
        'k_partitions': k_partitions,
        'capacity': capacity,
        'method_name': method_name,
        'method_version': method_version,
        'config_hash': config_hash,
        'bins': json.dumps(bins),
        'cost': cost,
        'respects_capacity': str(respects_capacity).lower(),
        'method_metadata': json.dumps(method_metadata or {}),
        'time_seconds': f"{time_seconds:.6f}" if time_seconds is not None else '',
        'memory_mb': f"{memory_mb:.2f}" if memory_mb is not None else '',
        'date_run': datetime.now().isoformat()[:10],
        'library_version': library_version,
        'contributor': contributor,
        'notes': notes
    }
    
    # Ensure directory exists
    ensure_dir(results_csv.parent)
    
    # Check if entry already exists
    existing = []
    if results_csv.exists():
        try:
            with open(results_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing = list(reader)
        except Exception as e:
            print(f"Warning: Error reading CSV: {e}")
            # Backup corrupted file
            backup_path = results_csv.with_suffix('.csv.backup')
            if results_csv.exists():
                import shutil
                shutil.copy(results_csv, backup_path)
                print(f"Backed up corrupted CSV to {backup_path}")
            existing = []
    
    # Check for duplicate
    dup_key = (workload_file, k_partitions, method_name, config_hash)
    duplicate_found = False
    for ex in existing:
        try:
            ex_key = (ex['workload_file'], int(ex['k_partitions']), 
                      ex['method_name'], ex['config_hash'])
            if dup_key == ex_key:
                duplicate_found = True
                if not allow_overwrite:
                    return False
                
                # Remove old entry
                existing = [e for e in existing if (e['workload_file'], int(e['k_partitions']), 
                           e['method_name'], e['config_hash']) != dup_key]
                break
        except (KeyError, ValueError) as e:
            # Skip malformed entries
            print(f"Warning: Skipping malformed entry: {e}")
            continue
    
    # Add new entry
    existing.append(entry)
    
    # Write back
    fieldnames = [
        'workload_file', 'model', 'origin', 'n_qubits', 'k_partitions', 'capacity',
        'method_name', 'method_version', 'config_hash',
        'bins', 'cost', 'respects_capacity', 'method_metadata',
        'time_seconds', 'memory_mb',
        'date_run', 'library_version', 'contributor', 'notes'
    ]
    
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
    
    return True

def load_hdh(hdh_path: Path):
    """Load HDH from pickle file."""
    with open(hdh_path, 'rb') as f:
        return pickle.load(f)

def calculate_capacity(n_qubits: int, k_partitions: int, overhead: float = 0.0) -> int:
    """
    Calculate capacity per partition with optional overhead.
    
    Args:
        n_qubits: Total number of qubits
        k_partitions: Number of partitions
        overhead: Fraction of extra capacity (0.0 = no overhead, 0.1 = 10% extra)
    
    Returns:
        Capacity per partition (integer)
    
    Examples:
        n_qubits=10, k=2, overhead=0.0  -> capacity=5
        n_qubits=10, k=2, overhead=0.2  -> capacity=6  (20% extra)
        n_qubits=10, k=3, overhead=0.0  -> capacity=4  (ceiling division)
    """
    base_capacity = (n_qubits + k_partitions - 1) // k_partitions  # Ceiling division
    capacity_with_overhead = int(base_capacity * (1.0 + overhead))
    return capacity_with_overhead

# THIS IS MY METHOD - Replace it by yours!
def run_greedy_hdh(hdh_graph, k: int, capacity: int, config: Dict) -> Tuple[List, int, Dict]:
    """
    Run greedy HDH partitioner.
    
    Returns:
        (bins, cost, metadata)
    """
    from hdh.passes.cut import compute_cut
    
    beam_k = config.get('beam_k', 3)
    reserve_frac = config.get('reserve_frac', 0.08)
    restarts = config.get('restarts', 1)
    seed = config.get('seed', 0)
    
    bins, cost = compute_cut(
        hdh_graph, 
        k=k, 
        cap=capacity,
        beam_k=beam_k,
        reserve_frac=reserve_frac,
        restarts=restarts,
        seed=seed
    )
    
    # Convert bins from sets to lists of qubit strings
    bins_list = []
    for bin_set in bins:
        bin_list = sorted(list(bin_set))
        bins_list.append(bin_list)
    
    # Check if capacity is respected
    respects_capacity = all(len(b) <= capacity for b in bins_list)
    
    metadata = {
        'beam_k': beam_k,
        'reserve_frac': reserve_frac,
        'restarts': restarts,
        'seed': seed
    }
    
    return bins_list, cost, respects_capacity, metadata

def run_metis_telegate(hdh_graph, k: int, capacity: int, config: Dict) -> Tuple[List, int, Dict]:
    """
    Run METIS telegate partitioner.
    
    Returns:
        (bins, cost, respects_capacity, metadata)
    """
    from hdh.passes.cut import metis_telegate
    
    bins_qubits, cost, respects_capacity, method = metis_telegate(
        hdh_graph,
        partitions=k,
        capacities=capacity
    )
    
    # Convert to list of lists
    bins_list = [sorted(list(b)) for b in bins_qubits]
    
    metadata = {
        'metis_method': method,
        'failed': not respects_capacity
    }
    
    return bins_list, cost, respects_capacity, metadata

# Registry of available methods
METHODS = {
    'greedy_hdh': {
        'function': run_greedy_hdh,
        'default_config': {
            'beam_k': 3,
            'reserve_frac': 0.08,
            'restarts': 1,
            'seed': 0
        }
    },
    'metis_telegate': {
        'function': run_metis_telegate,
        'default_config': {}
    }
}

def get_n_qubits(hdh_graph) -> int:
    """Extract number of qubits from HDH graph."""
    import re
    qubits = set()
    for node in hdh_graph.S:
        match = re.match(r"^q(\d+)_t\d+$", node)
        if match:
            qubits.add(int(match.group(1)))
    return len(qubits)

def find_hdh_files(
    database_root: Path,
    model: str = None,
    origin: str = None,
    max_qubits: int = None
) -> List[Dict]:
    """
    Find all HDH pickle files in the database.
    
    Returns:
        List of dicts with keys: hdh_path, workload_file, model, origin
    """
    hdhs_dir = database_root / "HDHs"
    
    if not hdhs_dir.exists():
        print(f"Error: HDHs directory not found: {hdhs_dir}")
        return []
    
    found = []
    
    # Search for pickle files
    if model and origin:
        search_path = hdhs_dir / model / origin / "pkl"
        pkl_files = list(search_path.glob("*.pkl")) if search_path.exists() else []
    elif model:
        pkl_files = list((hdhs_dir / model).rglob("*.pkl"))
    else:
        pkl_files = list(hdhs_dir.rglob("*.pkl"))
    
    for pkl_path in pkl_files:
        # Extract model, origin, and workload name from path
        parts = pkl_path.relative_to(hdhs_dir).parts
        file_model = parts[0]
        file_origin = parts[1] if len(parts) > 1 else "Unknown"
        workload_file = pkl_path.stem + ".qasm"
        
        # Filter by max_qubits if specified
        if max_qubits:
            try:
                hdh_graph = load_hdh(pkl_path)
                n_qubits = get_n_qubits(hdh_graph)
                if n_qubits > max_qubits:
                    continue
            except Exception as e:
                print(f"Warning: Could not load {pkl_path}: {e}")
                continue
        
        found.append({
            'hdh_path': pkl_path,
            'workload_file': workload_file,
            'model': file_model,
            'origin': file_origin
        })
    
    return found

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
    contributor: str
) -> Optional[Dict]:
    """
    Run a single partitioning test.
    
    Returns:
        Result dict or None if failed
    """
    try:
        # Load HDH
        hdh_graph = load_hdh(hdh_path)
        n_qubits = get_n_qubits(hdh_graph)
        
        # Calculate capacity
        capacity = calculate_capacity(n_qubits, k, overhead)
        
        # Get method function
        method_info = METHODS[method_name]
        method_fn = method_info['function']
        
        # Merge default config with user config
        config = method_info['default_config'].copy()
        config.update(method_config)
        
        # Run partitioning (with timing and memory tracking)
        tracemalloc.start()
        start_time = time.time()
        
        bins, cost, respects_capacity, metadata = method_fn(hdh_graph, k, capacity, config)
        
        elapsed_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak / (1024 * 1024)
        
        # Add overhead to metadata
        metadata['overhead'] = overhead
        metadata['base_capacity'] = (n_qubits + k - 1) // k
        
        # Build result
        result = {
            'workload_file': workload_file,
            'model': model,
            'origin': origin,
            'n_qubits': n_qubits,
            'k_partitions': k,
            'capacity': capacity,
            'method_name': method_name,
            'method_version': '1.0.0',
            'bins': bins,
            'cost': cost,
            'respects_capacity': respects_capacity,
            'method_metadata': metadata,
            'time_seconds': elapsed_time,
            'memory_mb': memory_mb,
            'library_version': library_version,
            'contributor': contributor
        }
        
        return result
        
    except Exception as e:
        return None

def run_test_worker(task: Dict) -> Tuple[bool, Optional[Dict], str]:
    """
    Worker function for parallel test execution.
    
    Args:
        task: Dictionary with test parameters
    
    Returns:
        (success, result_dict, error_msg)
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
            contributor=task['contributor']
        )
        
        if result:
            return True, result, ""
        else:
            return False, None, "Test execution failed"
            
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return False, None, error_msg

def main():
    parser = argparse.ArgumentParser(
        description='Run automated partitioning tests on HDH database (PARALLEL VERSION)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--database-root', type=Path, required=True,
                       help='Root directory of the database')
    
    # Filtering
    parser.add_argument('--model', help='Filter by model (e.g., Circuit)')
    parser.add_argument('--origin', help='Filter by origin (e.g., MQTBench)')
    parser.add_argument('--max-qubits', type=int, 
                       help='Only test circuits with <= this many qubits')
    
    # Test parameters
    parser.add_argument('--methods', nargs='+', required=True,
                       choices=list(METHODS.keys()),
                       help='Partitioning methods to test')
    parser.add_argument('--k-values', nargs='+', type=int, required=True,
                       help='Number of partitions to test (e.g., 2 3 4 5 6)')
    parser.add_argument('--overhead', nargs='+', type=float, default=[0.0],
                       help='Capacity overhead fraction(s) (0.0=none, 0.1=10%% extra). Can specify multiple.')
    
    # Configuration
    parser.add_argument('--config', type=str,
                       help='JSON string with method configuration (e.g., \'{"beam_k": 5}\')')
    parser.add_argument('--library-version', default='0.2.1',
                       help='HDH library version')
    parser.add_argument('--contributor', default='unknown',
                       help='Your name/username')
    
    # Parallel options
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count - 2)')
    
    # Output options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be tested without running')
    parser.add_argument('--output-json', type=Path,
                       help='Save results to JSON file instead of database')
    
    args = parser.parse_args()
    
    # Determine number of workers
    if args.workers is None:
        # Leave 2 cores free for system
        args.workers = max(1, mp.cpu_count() - 2)
    
    print(f"Using {args.workers} parallel workers")
    
    # Parse config
    method_config = {}
    if args.config:
        try:
            method_config = json.loads(args.config)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in --config: {args.config}")
            sys.exit(1)
    
    # Find HDH files
    print("Finding HDH files...")
    hdh_files = find_hdh_files(
        args.database_root,
        model=args.model,
        origin=args.origin,
        max_qubits=args.max_qubits
    )
    
    if not hdh_files:
        print("No HDH files found!")
        sys.exit(1)
    
    print(f"Found {len(hdh_files)} HDH files")
    
    # Build task list
    tasks = []
    for hdh_info in hdh_files:
        for method_name in args.methods:
            for k in args.k_values:
                for overhead in args.overhead:
                    tasks.append({
                        'hdh_path': hdh_info['hdh_path'],
                        'workload_file': hdh_info['workload_file'],
                        'model': hdh_info['model'],
                        'origin': hdh_info['origin'],
                        'k': k,
                        'overhead': overhead,
                        'method_name': method_name,
                        'method_config': method_config,
                        'library_version': args.library_version,
                        'contributor': args.contributor
                    })
    
    total_tests = len(tasks)
    print(f"Will run {total_tests} tests:")
    print(f"  {len(hdh_files)} circuits")
    print(f"  {len(args.methods)} methods: {', '.join(args.methods)}")
    print(f"  {len(args.k_values)} k values: {args.k_values}")
    print(f"  {len(args.overhead)} overhead values: {[f'{oh*100:.1f}%' for oh in args.overhead]}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would test:")
        for task in tasks[:5]:
            print(f"  - {task['model']}/{task['origin']}/{task['workload_file']} "
                  f"(k={task['k']}, method={task['method_name']})")
        if len(tasks) > 5:
            print(f"  ... and {len(tasks) - 5} more")
        sys.exit(0)
    
    # Run tests in parallel
    print("\nRunning tests in parallel...")
    results_csv = args.database_root / "Partitions" / "results" / "partitions.csv"
    
    all_results = []
    completed = 0
    failed = 0
    
    # Create a lock for thread-safe CSV writing
    lock = mp.Manager().Lock()
    
    # Run with multiprocessing pool
    with mp.Pool(processes=args.workers) as pool:
        results_iter = pool.imap_unordered(run_test_worker, tasks)
        
        for success, result, error_msg in tqdm(results_iter, total=total_tests, 
                                               desc="Processing", unit="test"):
            if success and result:
                all_results.append(result)
                
                if not args.output_json:
                    # Add directly to database (thread-safe)
                    try:
                        add_success = add_result_safe(
                            results_csv=results_csv,
                            result=result,
                            lock=lock,
                            allow_overwrite=True
                        )
                        if add_success:
                            completed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        tqdm.write(f"Error saving result: {e}")
                        failed += 1
                else:
                    completed += 1
            else:
                if error_msg:
                    tqdm.write(f"Test failed: {error_msg[:100]}")
                failed += 1
    
    # Save to JSON if requested
    if args.output_json:
        print(f"\nSaving results to {args.output_json}...")
        with open(args.output_json, 'w') as f:
            json.dump({'results': all_results}, f, indent=2)
        print(f"✓ Saved {len(all_results)} results")
    
    # Summary
    print("\n" + "="*60)
    print(f"Tests complete:")
    print(f"  Completed: {completed}/{total_tests}")
    print(f"  Failed: {failed}/{total_tests}")
    print(f"  Success rate: {100*completed/total_tests:.1f}%")
    
    sys.exit(0 if failed == 0 else 1)

if __name__ == '__main__':
    main()