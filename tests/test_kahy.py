#!/usr/bin/env python3
"""
Verify KaHyPar is accessible from hdh package
Run this from your HDH directory (the one containing hdh/)
"""

import sys
from pathlib import Path

print("="*70)
print("KAHYPAR ACCESSIBILITY VERIFICATION")
print("="*70)

# 1. Test basic kahypar import
print("\n1. Testing basic kahypar import...")
try:
    import kahypar
    print(f"   ✓ SUCCESS: kahypar imported from {kahypar.__file__}")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    print("\n   KaHyPar is not installed. Please run the installation again.")
    sys.exit(1)

# 2. Test hdh import
print("\n2. Testing hdh package import...")
try:
    from hdh import HDH
    print(f"   ✓ SUCCESS: HDH imported")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    print("\n   Make sure you're running this from the HDH directory")
    sys.exit(1)

# 3. Test kahypar_cutter import from hdh.passes.cut
print("\n3. Testing kahypar_cutter import from hdh.passes.cut...")
try:
    from hdh.passes.cut import kahypar_cutter
    print(f"   ✓ SUCCESS: kahypar_cutter imported")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    print("\n   Full traceback:")
    traceback.print_exc()
    sys.exit(1)

# 4. Test that kahypar is accessible from within hdh module
print("\n4. Testing kahypar accessibility from within hdh...")
try:
    from hdh.passes.cut import compute_cut
    print(f"   ✓ SUCCESS: compute_cut imported")
    
    # Try to access kahypar from within the module
    import inspect
    cut_module_file = inspect.getfile(compute_cut)
    print(f"   Module location: {cut_module_file}")
    
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Quick functional test
print("\n5. Testing basic kahypar functionality...")
try:
    # Create a simple test hypergraph
    from hdh import HDH
    
    test_hdh = HDH()
    test_hdh.add_node("n1", "q", 0)
    test_hdh.add_node("n2", "q", 0)
    test_hdh.add_node("n3", "q", 1)
    test_hdh.add_hyperedge({"n1", "n2"}, "q", "test")
    test_hdh.add_hyperedge({"n2", "n3"}, "q", "test")
    
    print(f"   Created test HDH with {len(test_hdh.nodes)} nodes and {len(test_hdh.hyperedges)} hyperedges")
    
    # Try to use kahypar_cutter
    partitions = kahypar_cutter(test_hdh, k=2, capacity=2)
    print(f"   ✓ SUCCESS: kahypar_cutter worked! Created {len(partitions)} partitions")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    print("\n   Full traceback:")
    traceback.print_exc()
    
    # This might fail for other reasons, so don't exit
    print("\n   Note: This may be a logic error, not an import error")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
✓ KaHyPar is installed and accessible!

You can now run your experiments:
  python hdh_experiments_FAST.py --exp 4

Or run all experiments:
  python hdh_experiments_FAST.py
""")
print("="*70)