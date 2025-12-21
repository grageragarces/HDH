"""
Validate results database for consistency and correctness.

Checks:
- Schema compliance (required fields, types)
- No duplicate entries
- Workload files exist
- Bins are valid JSON
- Costs are non-negative
- Capacity constraints make sense

IMPORTANT NOTE: this file cannot take the entire database!
------------------------------------------------------------
If you find a way to make it so it does please open a pull request. 
But otherwise what we recommend is that if you want to check your results (which you *really* should) 
you should call  --database-root to point to a folder such as database that 
only contains the files you may want to test.
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import sys

def validate_schema(row: dict, row_num: int) -> List[str]:
    """Validate a single row against schema."""
    errors = []
    
    # Required fields
    required = ['workload_file', 'model', 'origin', 'n_qubits', 'k_partitions',
                'capacity', 'method_name', 'method_version', 'bins', 'cost',
                'respects_capacity', 'date_run', 'library_version', 'contributor']
    
    for field in required:
        if field not in row or not row[field]:
            errors.append(f"Row {row_num}: Missing required field '{field}'")
    
    # Type validation
    try:
        n_qubits = int(row['n_qubits'])
        if n_qubits < 1:
            errors.append(f"Row {row_num}: n_qubits must be positive (got {n_qubits})")
    except (ValueError, KeyError):
        errors.append(f"Row {row_num}: n_qubits must be integer")
    
    try:
        k_partitions = int(row['k_partitions'])
        if k_partitions < 2:
            errors.append(f"Row {row_num}: k_partitions must be >= 2 (got {k_partitions})")
    except (ValueError, KeyError):
        errors.append(f"Row {row_num}: k_partitions must be integer")
    
    try:
        capacity = int(row['capacity'])
        if capacity < 1:
            errors.append(f"Row {row_num}: capacity must be positive (got {capacity})")
    except (ValueError, KeyError):
        errors.append(f"Row {row_num}: capacity must be integer")
    
    try:
        cost = int(row['cost'])
        if cost < 0:
            errors.append(f"Row {row_num}: cost cannot be negative (got {cost})")
    except (ValueError, KeyError):
        errors.append(f"Row {row_num}: cost must be integer")
    
    # Validate bins JSON
    try:
        bins = json.loads(row['bins'])
        if not isinstance(bins, list):
            errors.append(f"Row {row_num}: bins must be a JSON array")
        else:
            if len(bins) != int(row['k_partitions']):
                errors.append(f"Row {row_num}: bins length ({len(bins)}) doesn't match k_partitions ({row['k_partitions']})")
            
            # Check that bins contain qubit identifiers
            all_qubits = set()
            for i, partition in enumerate(bins):
                if not isinstance(partition, list):
                    errors.append(f"Row {row_num}: bins[{i}] must be a list")
                    continue
                for qubit in partition:
                    if qubit in all_qubits:
                        errors.append(f"Row {row_num}: qubit '{qubit}' appears in multiple partitions")
                    all_qubits.add(qubit)
            
            # Check capacity constraint if respects_capacity is true
            if row.get('respects_capacity', '').lower() == 'true':
                for i, partition in enumerate(bins):
                    if isinstance(partition, list) and len(partition) > int(row['capacity']):
                        errors.append(f"Row {row_num}: partition {i} has {len(partition)} qubits but capacity is {row['capacity']} (respects_capacity=true)")
    
    except json.JSONDecodeError:
        errors.append(f"Row {row_num}: bins must be valid JSON (got: {row.get('bins', '')[:50]}...)")
    except (ValueError, KeyError) as e:
        errors.append(f"Row {row_num}: Error validating bins: {e}")
    
    # Validate respects_capacity
    if row.get('respects_capacity', '').lower() not in ['true', 'false', '']:
        errors.append(f"Row {row_num}: respects_capacity must be 'true' or 'false'")
    
    # Validate method_metadata if present
    if row.get('method_metadata') and row['method_metadata'] != '{}':
        try:
            json.loads(row['method_metadata'])
        except json.JSONDecodeError:
            errors.append(f"Row {row_num}: method_metadata must be valid JSON")
    
    # Validate time_seconds if present
    if row.get('time_seconds') and row['time_seconds']:
        try:
            time_val = float(row['time_seconds'])
            if time_val < 0:
                errors.append(f"Row {row_num}: time_seconds cannot be negative")
        except ValueError:
            errors.append(f"Row {row_num}: time_seconds must be a number")
    
    # Validate memory_mb if present
    if row.get('memory_mb') and row['memory_mb']:
        try:
            mem_val = float(row['memory_mb'])
            if mem_val < 0:
                errors.append(f"Row {row_num}: memory_mb cannot be negative")
        except ValueError:
            errors.append(f"Row {row_num}: memory_mb must be a number")
    
    return errors

def validate_workload_files(results: List[dict], database_root: Path) -> List[str]:
    """Check that workload files actually exist."""
    errors = []
    warnings = []
    workloads_root = database_root / "Workloads"
    
    if not workloads_root.exists():
        warnings.append(f"Workloads directory not found: {workloads_root}")
        return warnings
    
    checked = set()
    for r in results:
        path_key = (r['model'], r['origin'], r['workload_file'])
        if path_key in checked:
            continue
        checked.add(path_key)
        
        workload_path = workloads_root / r['model'] / r['origin'] / r['workload_file']
        if not workload_path.exists():
            errors.append(f"Workload file not found: {workload_path}")
    
    return warnings + errors

def check_duplicates(results: List[dict]) -> List[str]:
    """Check for duplicate entries."""
    errors = []
    seen = defaultdict(list)
    
    for i, r in enumerate(results):
        key = (r['workload_file'], r['k_partitions'], r['method_name'], r['config_hash'])
        seen[key].append(i + 2)  # +2 for 1-indexing and header
    
    for key, row_nums in seen.items():
        if len(row_nums) > 1:
            errors.append(f"Duplicate entries for {key} at rows: {row_nums}")
    
    return errors

def check_consistency(results: List[dict]) -> List[str]:
    """Check for logical consistency across results."""
    errors = []
    warnings = []
    
    # Group by workload to check n_qubits consistency
    by_workload = defaultdict(list)
    for r in results:
        by_workload[r['workload_file']].append(r)
    
    for workload, entries in by_workload.items():
        n_qubits_values = set(e['n_qubits'] for e in entries)
        if len(n_qubits_values) > 1:
            errors.append(f"Inconsistent n_qubits for {workload}: {n_qubits_values}")
        
        models = set(e['model'] for e in entries)
        if len(models) > 1:
            warnings.append(f"Multiple models for {workload}: {models}")
        
        origins = set(e['origin'] for e in entries)
        if len(origins) > 1:
            warnings.append(f"Multiple origins for {workload}: {origins}")
    
    # Check capacity calculations
    for r in results:
        try:
            n_qubits = int(r['n_qubits'])
            k_partitions = int(r['k_partitions'])
            capacity = int(r['capacity'])
            
            min_capacity = (n_qubits + k_partitions - 1) // k_partitions  # Ceiling division
            if capacity < min_capacity:
                warnings.append(f"Row with {r['workload_file']}, k={k_partitions}: capacity {capacity} may be too small for {n_qubits} qubits (min {min_capacity})")
        except (ValueError, KeyError):
            pass  # Will be caught by schema validation
    
    return warnings + errors

def validate_database(database_root: Path, strict: bool = False) -> bool:
    """Main validation function."""
    print(f"Validating database at: {database_root}")
    
    results_csv = database_root / "Partitions" / "results" / "partitions.csv"
    if not results_csv.exists():
        print(f"Error: Results file not found: {results_csv}")
        return False
    
    # Load results
    print("Loading results...")
    results = []
    try:
        with open(results_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            if not fieldnames:
                print("Error: CSV file is empty or malformed")
                return False
            
            for row in reader:
                results.append(row)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False
    
    print(f"Loaded {len(results)} results")
    
    if len(results) == 0:
        print("Warning: No results found in database")
        return True
    
    all_errors = []
    all_warnings = []
    
    # Schema validation
    print("\n1. Validating schema...")
    for i, row in enumerate(results):
        errors = validate_schema(row, i + 2)  # +2 for 1-indexing and header
        all_errors.extend(errors)
    
    if not all_errors:
        print("   ✓ Schema validation passed")
    else:
        print(f"   ✗ Found {len(all_errors)} schema errors")
    
    # Duplicate check
    print("\n2. Checking for duplicates...")
    dup_errors = check_duplicates(results)
    all_errors.extend(dup_errors)
    
    if not dup_errors:
        print("   ✓ No duplicates found")
    else:
        print(f"   ✗ Found {len(dup_errors)} duplicate entries")
    
    # Consistency check
    print("\n3. Checking logical consistency...")
    consistency_issues = check_consistency(results)
    # Separate warnings from errors
    for issue in consistency_issues:
        if issue.startswith("Inconsistent"):
            all_errors.append(issue)
        else:
            all_warnings.append(issue)
    
    if not consistency_issues:
        print("   ✓ Consistency check passed")
    else:
        error_count = sum(1 for i in consistency_issues if i.startswith("Inconsistent"))
        warning_count = len(consistency_issues) - error_count
        if error_count:
            print(f"   ✗ Found {error_count} consistency errors")
        if warning_count:
            print(f"   ⚠ Found {warning_count} consistency warnings")
    
    # Workload file check
    print("\n4. Validating workload files exist...")
    file_issues = validate_workload_files(results, database_root)
    # Separate warnings from errors
    for issue in file_issues:
        if "not found" in issue and "directory" not in issue:
            all_errors.append(issue)
        else:
            all_warnings.append(issue)
    
    error_count = sum(1 for i in file_issues if "not found" in i and "directory" not in i)
    warning_count = len(file_issues) - error_count
    
    if not file_issues:
        print("   ✓ All workload files exist")
    else:
        if error_count:
            print(f"   ✗ Found {error_count} missing workload files")
        if warning_count:
            print(f"   ⚠ Found {warning_count} file-related warnings")
    
    # Summary
    print("\n" + "="*60)
    
    if all_warnings:
        print(f"\n⚠ WARNINGS ({len(all_warnings)}):")
        for warning in all_warnings[:10]:
            print(f"  - {warning}")
        if len(all_warnings) > 10:
            print(f"  ... and {len(all_warnings) - 10} more warnings")
    
    if all_errors:
        print(f"\n✗ VALIDATION FAILED: {len(all_errors)} errors found\n")
        for error in all_errors[:20]:  # Show first 20 errors
            print(f"  - {error}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more errors")
        return False
    else:
        if all_warnings and strict:
            print("\n⚠ VALIDATION PASSED WITH WARNINGS (strict mode)")
            print("Fix warnings before proceeding in strict mode.")
            return False
        else:
            print("✓ VALIDATION PASSED: No errors found")
            if all_warnings:
                print(f"  ({len(all_warnings)} warnings - review recommended)")
            return True

def main():
    parser = argparse.ArgumentParser(description='Validate HDH results database')
    parser.add_argument('--database-root', type=Path, required=True,
                       help='Root directory of the database')
    parser.add_argument('--strict', action='store_true',
                       help='Treat warnings as errors')
    
    args = parser.parse_args()
    
    if not args.database_root.exists():
        print(f"Error: Database root does not exist: {args.database_root}")
        sys.exit(1)
    
    success = validate_database(args.database_root, args.strict)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
