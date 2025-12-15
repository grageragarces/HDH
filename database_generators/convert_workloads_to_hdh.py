"""
Convert workload files (QASM, etc.) to HDH format.

This script:
1. Finds all workload files in Database/Workloads/
2. Converts them to HDH using the appropriate converter
3. Saves as pickle files in Database/HDHs/
4. Optionally exports to CSV format (text/)
"""

import argparse
import pickle
import csv
import json
from pathlib import Path
from typing import Dict, List, Any
import sys
from tqdm import tqdm

def ensure_dir(p: Path):
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def save_pkl(hdh_graph, output_path: Path):
    """Save HDH as pickle."""
    ensure_dir(output_path.parent)
    with open(output_path, 'wb') as f:
        pickle.dump(hdh_graph, f)

def save_nodes_csv(hdh_graph, output_path: Path):
    """Save nodes to CSV."""
    ensure_dir(output_path.parent)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'type', 'time', 'realisation'])
        for nid in sorted(hdh_graph.S):
            writer.writerow([
                nid,
                hdh_graph.sigma.get(nid, ''),
                getattr(hdh_graph, 'time_map', {}).get(nid, ''),
                hdh_graph.upsilon.get(nid, '')
            ])

def save_edges_csvs(hdh_graph, edges_path: Path, members_path: Path):
    """Save edges and edge members to CSV."""
    ensure_dir(edges_path.parent)
    
    # Stable ordering for edge indices
    edges_sorted = sorted(hdh_graph.C, key=lambda e: tuple(sorted(e)))
    
    # Edges table
    with open(edges_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'edge_index', 'type', 'realisation', 'gate_name', 'role',
            'edge_args', 'edge_metadata'
        ])
        for idx, e in enumerate(edges_sorted):
            writer.writerow([
                idx,
                hdh_graph.tau.get(e, ''),
                hdh_graph.phi.get(e, ''),
                getattr(hdh_graph, 'gate_name', {}).get(e, ''),
                getattr(hdh_graph, 'edge_role', {}).get(e, ''),
                json.dumps(getattr(hdh_graph, 'edge_args', {}).get(e, None)) if e in getattr(hdh_graph, 'edge_args', {}) else '',
                json.dumps(getattr(hdh_graph, 'edge_metadata', {}).get(e, None)) if e in getattr(hdh_graph, 'edge_metadata', {}) else ''
            ])
    
    # Edge members table
    with open(members_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['edge_index', 'node_id'])
        for idx, e in enumerate(edges_sorted):
            for nid in sorted(e):
                writer.writerow([idx, nid])

def convert_qasm_file(qasm_path: Path, output_base: Path, export_csv: bool = True):
    """
    Convert a single QASM file to HDH.
    
    Args:
        qasm_path: Path to QASM file
        output_base: Base path for outputs (without extension)
        export_csv: Whether to export CSV files
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from hdh.converters import from_qasm
        
        # Convert QASM to HDH
        hdh_graph = from_qasm('file', str(qasm_path))
        
        # Save pickle
        pkl_path = output_base.parent / 'pkl' / (output_base.name + '.pkl')
        save_pkl(hdh_graph, pkl_path)
        
        # Save CSV if requested
        if export_csv:
            text_dir = output_base.parent / 'text'
            save_nodes_csv(hdh_graph, text_dir / f"{output_base.name}__nodes.csv")
            save_edges_csvs(
                hdh_graph,
                text_dir / f"{output_base.name}__edges.csv",
                text_dir / f"{output_base.name}__edge_members.csv"
            )
        
        return True
    
    except Exception as e:
        print(f"  Error: {e}")
        return False

def find_workload_files(database_root: Path, model: str = None, origin: str = None, 
                        file_pattern: str = "*.qasm") -> List[Dict]:
    """
    Find all workload files in the database.
    
    Returns:
        List of dicts with keys: workload_path, model, origin, stem
    """
    workloads_dir = database_root / "Workloads"
    
    if not workloads_dir.exists():
        print(f"Error: Workloads directory not found: {workloads_dir}")
        return []
    
    found = []
    
    # Search pattern
    if model and origin:
        search_path = workloads_dir / model / origin
        workload_files = list(search_path.glob(file_pattern)) if search_path.exists() else []
    elif model:
        workload_files = list((workloads_dir / model).rglob(file_pattern))
    else:
        workload_files = list(workloads_dir.rglob(file_pattern))
    
    for workload_path in workload_files:
        # Extract model and origin from path
        parts = workload_path.relative_to(workloads_dir).parts
        file_model = parts[0]
        file_origin = parts[1] if len(parts) > 1 else "Unknown"
        
        found.append({
            'workload_path': workload_path,
            'model': file_model,
            'origin': file_origin,
            'stem': workload_path.stem
        })
    
    return found

def main():
    parser = argparse.ArgumentParser(
        description='Convert workload files to HDH format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all QASM files in MQTBench
  python convert_workloads_to_hdh.py --database-root ../Database \\
    --origin MQTBench

  # Convert all workloads for Circuits model
  python convert_workloads_to_hdh.py --database-root ../Database \\
    --model Circuits

  # Convert everything
  python convert_workloads_to_hdh.py --database-root ../Database

  # Convert without CSV export (faster)
  python convert_workloads_to_hdh.py --database-root ../Database \\
    --origin MQTBench --no-csv

  # Convert with custom file pattern
  python convert_workloads_to_hdh.py --database-root ../Database \\
    --pattern "*.qasm" --origin MyBench
        """
    )
    
    parser.add_argument('--database-root', type=Path, required=True,
                       help='Root directory of the database')
    
    # Filtering
    parser.add_argument('--model', help='Filter by model (e.g., Circuits)')
    parser.add_argument('--origin', help='Filter by origin (e.g., MQTBench)')
    parser.add_argument('--pattern', default='*.qasm',
                       help='File pattern to match (default: *.qasm)')
    
    # Options
    parser.add_argument('--no-csv', action='store_true',
                       help='Skip CSV export (only create pickle files)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of files to convert (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be converted without doing it')
    
    args = parser.parse_args()
    
    # Find workload files
    print("Finding workload files...")
    workload_files = find_workload_files(
        args.database_root,
        model=args.model,
        origin=args.origin,
        file_pattern=args.pattern
    )
    
    if not workload_files:
        print("No workload files found!")
        print("\nMake sure your workloads are in:")
        print(f"  {args.database_root}/Workloads/<Model>/<Origin>/*.qasm")
        sys.exit(1)
    
    print(f"Found {len(workload_files)} workload files")
    
    # Apply limit if specified
    if args.limit:
        workload_files = workload_files[:args.limit]
        print(f"Limited to {len(workload_files)} files")
    
    if args.dry_run:
        print("\n[DRY RUN] Would convert:")
        for wf in workload_files[:10]:
            print(f"  {wf['model']}/{wf['origin']}/{wf['workload_path'].name}")
        if len(workload_files) > 10:
            print(f"  ... and {len(workload_files) - 10} more")
        sys.exit(0)
    
    # Convert files
    print("\nConverting to HDH...")
    hdhs_root = args.database_root / "HDHs"
    
    success_count = 0
    fail_count = 0
    
    for wf in tqdm(workload_files, desc="Converting", unit="file"):
        # Construct output path
        output_base = hdhs_root / wf['model'] / wf['origin'] / wf['stem']
        
        # Convert based on file type
        if wf['workload_path'].suffix == '.qasm':
            success = convert_qasm_file(
                wf['workload_path'],
                output_base,
                export_csv=not args.no_csv
            )
        else:
            tqdm.write(f"Skipping {wf['workload_path'].name}: Unsupported format")
            fail_count += 1
            continue
        
        if success:
            success_count += 1
        else:
            tqdm.write(f"Failed: {wf['workload_path'].name}")
            fail_count += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Conversion complete:")
    print(f"  Success: {success_count}/{len(workload_files)}")
    print(f"  Failed: {fail_count}/{len(workload_files)}")
    
    if success_count > 0:
        print(f"\nHDH files saved to:")
        print(f"  {hdhs_root}/<Model>/<Origin>/pkl/*.pkl")
        if not args.no_csv:
            print(f"  {hdhs_root}/<Model>/<Origin>/text/*.csv")
        
        print(f"\nNext steps:")
        print(f"  1. Run partitioning tests:")
        print(f"     python run_partition_tests.py --database-root {args.database_root} \\")
        print(f"       --origin {args.origin or 'MQTBench'} --methods greedy_hdh --k-values 2 3 4")
        print(f"  2. Generate leaderboards:")
        print(f"     python generate_leaderboards.py --database-root {args.database_root}")
    
    sys.exit(0 if fail_count == 0 else 1)

if __name__ == '__main__':
    main()
