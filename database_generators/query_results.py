"""
Query and analyze partitioning results.

Commands:
  best      - Get best method for a specific workload and k
  compare   - Compare two methods across workloads
  stats     - Get statistics for a method
  list      - List all methods, workloads, or origins
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import sys

def load_results(results_csv: Path) -> List[Dict[str, Any]]:
    """Load results from CSV."""
    results = []
    with open(results_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['n_qubits'] = int(row['n_qubits'])
            row['k_partitions'] = int(row['k_partitions'])
            row['capacity'] = int(row['capacity'])
            row['cost'] = int(row['cost'])
            row['respects_capacity'] = row['respects_capacity'].lower() == 'true'
            
            # Convert time and memory if present
            if row.get('time_seconds'):
                try:
                    row['time_seconds'] = float(row['time_seconds'])
                except ValueError:
                    row['time_seconds'] = None
            else:
                row['time_seconds'] = None
            
            if row.get('memory_mb'):
                try:
                    row['memory_mb'] = float(row['memory_mb'])
                except ValueError:
                    row['memory_mb'] = None
            else:
                row['memory_mb'] = None
            
            results.append(row)
    return results

def get_best_method(results: List[Dict], workload: str, k: int):
    """Get best method for specific workload and k."""
    filtered = [r for r in results 
                if r['workload_file'] == workload and r['k_partitions'] == k]
    
    if not filtered:
        print(f"No results found for {workload} with k={k}")
        return
    
    # Filter to those respecting capacity
    valid = [r for r in filtered if r['respects_capacity']]
    if not valid:
        print("Warning: No methods respect capacity constraint")
        valid = filtered
    
    # Sort by cost
    valid.sort(key=lambda r: r['cost'])
    
    print(f"\nResults for {workload} (k={k}):")
    print(f"{'Method':<25} {'Version':<10} {'Cost':<8} {'Respects Cap':<15} {'Time (s)':<12}")
    print("-" * 80)
    for r in valid:
        time_str = f"{r['time_seconds']:.3f}" if r['time_seconds'] is not None else "N/A"
        print(f"{r['method_name']:<25} {r['method_version']:<10} {r['cost']:<8} "
              f"{str(r['respects_capacity']):<15} {time_str:<12}")
    
    print(f"\n→ Best: {valid[0]['method_name']} (cost={valid[0]['cost']})")
    
    # Show bins for best method
    try:
        bins = json.loads(valid[0]['bins'])
        print(f"  Bins: {bins}")
    except json.JSONDecodeError:
        pass

def compare_methods(results: List[Dict], method_a: str, method_b: str, 
                   origin: str = None, k: int = None):
    """Compare two methods across workloads."""
    filtered = results
    
    if origin:
        filtered = [r for r in filtered if r['origin'] == origin]
    
    if k:
        filtered = [r for r in filtered if r['k_partitions'] == k]
    
    a_results = [r for r in filtered if r['method_name'] == method_a]
    b_results = [r for r in filtered if r['method_name'] == method_b]
    
    if not a_results or not b_results:
        print(f"Insufficient results to compare {method_a} vs {method_b}")
        if origin:
            print(f"  (filtered by origin: {origin})")
        if k:
            print(f"  (filtered by k: {k})")
        return
    
    # Match up results by (workload, k)
    a_dict = {(r['workload_file'], r['k_partitions']): r for r in a_results}
    b_dict = {(r['workload_file'], r['k_partitions']): r for r in b_results}
    
    common_keys = set(a_dict.keys()) & set(b_dict.keys())
    
    if not common_keys:
        print(f"No common workloads between {method_a} and {method_b}")
        return
    
    a_wins = 0
    b_wins = 0
    ties = 0
    total_a_cost = 0
    total_b_cost = 0
    
    print(f"\n{'='*90}")
    print(f"Comparing {method_a} vs {method_b}")
    if origin:
        print(f"Origin: {origin}")
    if k:
        print(f"k: {k}")
    print(f"{'='*90}")
    print(f"\n{'Workload':<35} {'k':<5} {method_a:<12} {method_b:<12} {'Winner':<15}")
    print("-" * 90)
    
    for key in sorted(common_keys):
        workload, k_val = key
        a_cost = a_dict[key]['cost']
        b_cost = b_dict[key]['cost']
        
        total_a_cost += a_cost
        total_b_cost += b_cost
        
        if a_cost < b_cost:
            winner = method_a
            a_wins += 1
        elif b_cost < a_cost:
            winner = method_b
            b_wins += 1
        else:
            winner = "tie"
            ties += 1
        
        # Truncate long workload names
        workload_display = workload if len(workload) <= 33 else workload[:30] + "..."
        print(f"{workload_display:<35} {k_val:<5} {a_cost:<12} {b_cost:<12} {winner:<15}")
    
    avg_a = total_a_cost / len(common_keys)
    avg_b = total_b_cost / len(common_keys)
    
    print("\n" + "="*90)
    print(f"Results across {len(common_keys)} workloads:")
    print(f"\n{method_a}:")
    print(f"  Wins: {a_wins} ({100*a_wins/len(common_keys):.1f}%)")
    print(f"  Average cost: {avg_a:.2f}")
    
    print(f"\n{method_b}:")
    print(f"  Wins: {b_wins} ({100*b_wins/len(common_keys):.1f}%)")
    print(f"  Average cost: {avg_b:.2f}")
    
    print(f"\nTies: {ties}")
    
    if a_wins > b_wins:
        print(f"\n→ {method_a} performs better overall")
    elif b_wins > a_wins:
        print(f"\n→ {method_b} performs better overall")
    else:
        print(f"\n→ Methods are evenly matched")

def get_method_stats(results: List[Dict], method_name: str):
    """Get statistics for a method."""
    method_results = [r for r in results if r['method_name'] == method_name]
    
    if not method_results:
        print(f"No results found for method: {method_name}")
        return
    
    total = len(method_results)
    respects_cap = sum(1 for r in method_results if r['respects_capacity'])
    total_cost = sum(r['cost'] for r in method_results)
    avg_cost = total_cost / total
    
    # Time statistics
    times = [r['time_seconds'] for r in method_results if r['time_seconds'] is not None]
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
    else:
        avg_time = min_time = max_time = None
    
    # Memory statistics
    mems = [r['memory_mb'] for r in method_results if r['memory_mb'] is not None]
    if mems:
        avg_mem = sum(mems) / len(mems)
        max_mem = max(mems)
    else:
        avg_mem = max_mem = None
    
    # Count wins (best cost for each workload/k)
    workload_groups = defaultdict(list)
    for r in results:
        key = (r['workload_file'], r['k_partitions'])
        workload_groups[key].append(r)
    
    wins = 0
    for key, group in workload_groups.items():
        valid = [r for r in group if r['respects_capacity']]
        if not valid:
            valid = group
        if not valid:
            continue
        best = min(valid, key=lambda r: r['cost'])
        if best['method_name'] == method_name:
            wins += 1
    
    # Cost distribution
    costs = [r['cost'] for r in method_results]
    min_cost = min(costs)
    max_cost = max(costs)
    
    # Group by origin
    by_origin = defaultdict(list)
    for r in method_results:
        by_origin[r['origin']].append(r)
    
    print(f"\n{'='*60}")
    print(f"Statistics for {method_name}")
    print(f"{'='*60}")
    
    print(f"\nGeneral:")
    print(f"  Total results: {total}")
    print(f"  Respects capacity: {respects_cap}/{total} ({100*respects_cap/total:.1f}%)")
    print(f"  Wins (best cost): {wins}/{len(workload_groups)} ({100*wins/len(workload_groups):.1f}%)")
    
    print(f"\nCost statistics:")
    print(f"  Average: {avg_cost:.2f}")
    print(f"  Min: {min_cost}")
    print(f"  Max: {max_cost}")
    print(f"  Total: {total_cost}")
    
    if avg_time is not None:
        print(f"\nTime statistics (seconds):")
        print(f"  Average: {avg_time:.3f}")
        print(f"  Min: {min_time:.3f}")
        print(f"  Max: {max_time:.3f}")
        print(f"  (Available for {len(times)}/{total} results)")
    
    if avg_mem is not None:
        print(f"\nMemory statistics (MB):")
        print(f"  Average: {avg_mem:.2f}")
        print(f"  Max: {max_mem:.2f}")
        print(f"  (Available for {len(mems)}/{total} results)")
    
    print(f"\nBy origin:")
    for origin in sorted(by_origin.keys()):
        origin_results = by_origin[origin]
        origin_avg = sum(r['cost'] for r in origin_results) / len(origin_results)
        print(f"  {origin}: {len(origin_results)} results, avg cost {origin_avg:.2f}")

def list_items(results: List[Dict], item_type: str):
    """List methods, workloads, or origins."""
    if item_type == 'methods':
        methods = defaultdict(int)
        for r in results:
            methods[r['method_name']] += 1
        
        print(f"\nMethods ({len(methods)} total):")
        print(f"{'Method':<30} {'Results':<10}")
        print("-" * 40)
        for method, count in sorted(methods.items()):
            print(f"{method:<30} {count:<10}")
    
    elif item_type == 'workloads':
        workloads = defaultdict(lambda: {'count': 0, 'n_qubits': 0, 'origin': ''})
        for r in results:
            key = r['workload_file']
            workloads[key]['count'] += 1
            workloads[key]['n_qubits'] = r['n_qubits']
            workloads[key]['origin'] = r['origin']
        
        print(f"\nWorkloads ({len(workloads)} total):")
        print(f"{'Workload':<40} {'Qubits':<10} {'Origin':<20} {'Results':<10}")
        print("-" * 80)
        for workload, info in sorted(workloads.items()):
            w_display = workload if len(workload) <= 38 else workload[:35] + "..."
            print(f"{w_display:<40} {info['n_qubits']:<10} {info['origin']:<20} {info['count']:<10}")
    
    elif item_type == 'origins':
        origins = defaultdict(lambda: {'count': 0, 'workloads': set()})
        for r in results:
            origins[r['origin']]['count'] += 1
            origins[r['origin']]['workloads'].add(r['workload_file'])
        
        print(f"\nOrigins ({len(origins)} total):")
        print(f"{'Origin':<30} {'Workloads':<12} {'Results':<10}")
        print("-" * 52)
        for origin, info in sorted(origins.items()):
            print(f"{origin:<30} {len(info['workloads']):<12} {info['count']:<10}")

def main():
    parser = argparse.ArgumentParser(
        description='Query partitioning results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get best method for a workload
  python query_results.py --database-root ./Database \\
    best --workload "circuit.qasm" --k 2

  # Compare two methods
  python query_results.py --database-root ./Database \\
    compare --method-a greedy_hdh --method-b metis_telegate

  # Compare on specific origin
  python query_results.py --database-root ./Database \\
    compare --method-a greedy_hdh --method-b metis_telegate --origin MQTBench

  # Get method statistics
  python query_results.py --database-root ./Database \\
    stats --method greedy_hdh

  # List all methods
  python query_results.py --database-root ./Database list --type methods
        """
    )
    
    parser.add_argument('--database-root', type=Path, required=True,
                       help='Root directory of the database')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # best command
    best_parser = subparsers.add_parser('best', help='Get best method for workload')
    best_parser.add_argument('--workload', required=True, help='Workload filename')
    best_parser.add_argument('--k', type=int, required=True, help='Number of partitions')
    
    # compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two methods')
    compare_parser.add_argument('--method-a', required=True, help='First method name')
    compare_parser.add_argument('--method-b', required=True, help='Second method name')
    compare_parser.add_argument('--origin', help='Filter by origin')
    compare_parser.add_argument('--k', type=int, help='Filter by k partitions')
    
    # stats command
    stats_parser = subparsers.add_parser('stats', help='Get method statistics')
    stats_parser.add_argument('--method', required=True, help='Method name')
    
    # list command
    list_parser = subparsers.add_parser('list', help='List methods, workloads, or origins')
    list_parser.add_argument('--type', required=True, 
                            choices=['methods', 'workloads', 'origins'],
                            help='What to list')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    results_csv = args.database_root / "Partitions" / "results" / "partitions.csv"
    if not results_csv.exists():
        print(f"Error: Results file not found: {results_csv}")
        sys.exit(1)
    
    print("Loading results...")
    results = load_results(results_csv)
    print(f"Loaded {len(results)} results\n")
    
    if args.command == 'best':
        get_best_method(results, args.workload, args.k)
    elif args.command == 'compare':
        compare_methods(results, args.method_a, args.method_b, args.origin, args.k)
    elif args.command == 'stats':
        get_method_stats(results, args.method)
    elif args.command == 'list':
        list_items(results, args.type)

if __name__ == '__main__':
    main()
