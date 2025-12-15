"""
Generate leaderboards from consolidated results.

This script reads the main partitions.csv file and generates several
pre-computed leaderboard views for quick access.
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
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
            results.append(row)
    return results

def generate_overall_leaderboard(results: List[Dict], output_path: Path):
    """Generate overall leaderboard showing best method for each (workload, k)."""
    # Group by (workload_file, k_partitions)
    grouped = defaultdict(list)
    for r in results:
        key = (r['workload_file'], r['k_partitions'])
        grouped[key].append(r)
    
    leaderboard = []
    for (workload, k), entries in sorted(grouped.items()):
        # Filter to only entries that respect capacity
        valid = [e for e in entries if e['respects_capacity']]
        if not valid:
            valid = entries  # If none respect capacity, include all
        
        # Sort by cost (ascending)
        valid.sort(key=lambda e: e['cost'])
        
        if len(valid) == 0:
            continue
        
        best = valid[0]
        second = valid[1] if len(valid) > 1 else None
        
        entry = {
            'workload_file': workload,
            'model': best['model'],
            'origin': best['origin'],
            'n_qubits': best['n_qubits'],
            'k_partitions': k,
            'best_method': best['method_name'],
            'best_cost': best['cost'],
            'second_method': second['method_name'] if second else '',
            'second_cost': second['cost'] if second else '',
            'methods_tested': len(set(e['method_name'] for e in entries))
        }
        leaderboard.append(entry)
    
    # Write to CSV
    fieldnames = ['workload_file', 'model', 'origin', 'n_qubits', 'k_partitions',
                  'best_method', 'best_cost', 'second_method', 'second_cost', 'methods_tested']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(leaderboard)
    
    print(f"✓ Generated overall leaderboard: {output_path} ({len(leaderboard)} entries)")

def generate_by_k_leaderboards(results: List[Dict], output_dir: Path):
    """Generate leaderboards for each k value."""
    # Group by k_partitions
    by_k = defaultdict(list)
    for r in results:
        by_k[r['k_partitions']].append(r)
    
    for k, entries in sorted(by_k.items()):
        # Group by workload
        grouped = defaultdict(list)
        for e in entries:
            grouped[e['workload_file']].append(e)
        
        leaderboard = []
        for workload, workload_entries in sorted(grouped.items()):
            valid = [e for e in workload_entries if e['respects_capacity']]
            if not valid:
                valid = workload_entries
            
            valid.sort(key=lambda e: e['cost'])
            if not valid:
                continue
            
            best = valid[0]
            second = valid[1] if len(valid) > 1 else None
            
            entry = {
                'workload_file': workload,
                'model': best['model'],
                'origin': best['origin'],
                'n_qubits': best['n_qubits'],
                'best_method': best['method_name'],
                'best_cost': best['cost'],
                'win_margin': (second['cost'] - best['cost']) if second else 0,
                'methods_tested': len(set(e['method_name'] for e in workload_entries))
            }
            leaderboard.append(entry)
        
        # Write to CSV
        output_path = output_dir / f"k{k}.csv"
        fieldnames = ['workload_file', 'model', 'origin', 'n_qubits',
                     'best_method', 'best_cost', 'win_margin', 'methods_tested']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(leaderboard)
        
        print(f"✓ Generated k={k} leaderboard: {output_path} ({len(leaderboard)} entries)")

def generate_by_origin_leaderboards(results: List[Dict], output_dir: Path):
    """Generate leaderboards for each origin."""
    # Group by origin
    by_origin = defaultdict(list)
    for r in results:
        by_origin[r['origin']].append(r)
    
    for origin, entries in sorted(by_origin.items()):
        # Group by (workload, k)
        grouped = defaultdict(list)
        for e in entries:
            key = (e['workload_file'], e['k_partitions'])
            grouped[key].append(e)
        
        leaderboard = []
        for (workload, k), workload_entries in sorted(grouped.items()):
            valid = [e for e in workload_entries if e['respects_capacity']]
            if not valid:
                valid = workload_entries
            
            valid.sort(key=lambda e: e['cost'])
            if not valid:
                continue
            
            best = valid[0]
            entry = {
                'workload_file': workload,
                'n_qubits': best['n_qubits'],
                'k_partitions': k,
                'best_method': best['method_name'],
                'best_cost': best['cost'],
                'methods_tested': len(set(e['method_name'] for e in workload_entries))
            }
            leaderboard.append(entry)
        
        # Write to CSV
        output_path = output_dir / f"{origin}.csv"
        fieldnames = ['workload_file', 'n_qubits', 'k_partitions',
                     'best_method', 'best_cost', 'methods_tested']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(leaderboard)
        
        print(f"✓ Generated {origin} leaderboard: {output_path} ({len(leaderboard)} entries)")

def generate_by_qubit_count_leaderboards(results: List[Dict], output_dir: Path):
    """Generate leaderboards by qubit count ranges."""
    ranges = [
        ('small_5-10', 5, 10),
        ('medium_11-20', 11, 20),
        ('large_20plus', 21, float('inf'))
    ]
    
    for range_name, min_q, max_q in ranges:
        # Filter results by qubit range
        filtered = [r for r in results if min_q <= r['n_qubits'] <= max_q]
        if not filtered:
            continue
        
        # Group by (workload, k)
        grouped = defaultdict(list)
        for r in filtered:
            key = (r['workload_file'], r['k_partitions'])
            grouped[key].append(r)
        
        leaderboard = []
        for (workload, k), entries in sorted(grouped.items()):
            valid = [e for e in entries if e['respects_capacity']]
            if not valid:
                valid = entries
            
            valid.sort(key=lambda e: e['cost'])
            if not valid:
                continue
            
            best = valid[0]
            entry = {
                'workload_file': workload,
                'origin': best['origin'],
                'n_qubits': best['n_qubits'],
                'k_partitions': k,
                'best_method': best['method_name'],
                'best_cost': best['cost']
            }
            leaderboard.append(entry)
        
        # Write to CSV
        output_path = output_dir / f"{range_name}.csv"
        fieldnames = ['workload_file', 'origin', 'n_qubits', 'k_partitions',
                     'best_method', 'best_cost']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(leaderboard)
        
        print(f"✓ Generated {range_name} leaderboard: {output_path} ({len(leaderboard)} entries)")

def generate_method_summary(results: List[Dict], output_path: Path):
    """Generate summary statistics for each method."""
    # Group by method
    by_method = defaultdict(list)
    for r in results:
        by_method[r['method_name']].append(r)
    
    # Calculate wins for each method
    workload_groups = defaultdict(list)
    for r in results:
        key = (r['workload_file'], r['k_partitions'])
        workload_groups[key].append(r)
    
    method_stats = []
    for method_name, entries in sorted(by_method.items()):
        total = len(entries)
        respects_cap = sum(1 for r in entries if r['respects_capacity'])
        avg_cost = sum(r['cost'] for r in entries) / total if total > 0 else 0
        
        # Count wins
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
        
        total_workloads = len(workload_groups)
        win_rate = (wins / total_workloads * 100) if total_workloads > 0 else 0
        
        method_stats.append({
            'method_name': method_name,
            'total_results': total,
            'respects_capacity_count': respects_cap,
            'respects_capacity_rate': f"{100*respects_cap/total:.1f}%" if total > 0 else "0%",
            'average_cost': f"{avg_cost:.2f}",
            'wins': wins,
            'total_workloads': total_workloads,
            'win_rate': f"{win_rate:.1f}%"
        })
    
    # Write to CSV
    fieldnames = ['method_name', 'total_results', 'respects_capacity_count', 
                  'respects_capacity_rate', 'average_cost', 'wins', 'total_workloads', 'win_rate']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(method_stats)
    
    print(f"✓ Generated method summary: {output_path} ({len(method_stats)} methods)")

def main():
    parser = argparse.ArgumentParser(description='Generate leaderboards from results')
    parser.add_argument('--database-root', type=Path, required=True,
                       help='Root directory of the database')
    
    args = parser.parse_args()
    
    results_csv = args.database_root / "Partitions" / "results" / "partitions.csv"
    if not results_csv.exists():
        print(f"Error: Results file not found: {results_csv}")
        print("Make sure you have created the results file first!")
        sys.exit(1)
    
    print("Loading results...")
    results = load_results(results_csv)
    print(f"Loaded {len(results)} results")
    
    if len(results) == 0:
        print("No results to process!")
        sys.exit(0)
    
    leaderboards_root = args.database_root / "Partitions" / "leaderboards"
    
    # Ensure directories exist
    leaderboards_root.mkdir(parents=True, exist_ok=True)
    for subdir in ['by_k', 'by_origin', 'by_qubit_count']:
        (leaderboards_root / subdir).mkdir(parents=True, exist_ok=True)
    
    # Generate leaderboards
    print("\nGenerating leaderboards...")
    generate_overall_leaderboard(results, leaderboards_root / "overall.csv")
    generate_by_k_leaderboards(results, leaderboards_root / "by_k")
    generate_by_origin_leaderboards(results, leaderboards_root / "by_origin")
    generate_by_qubit_count_leaderboards(results, leaderboards_root / "by_qubit_count")
    generate_method_summary(results, leaderboards_root / "method_summary.csv")
    
    print("\n✓ All leaderboards generated!")

if __name__ == '__main__':
    main()
