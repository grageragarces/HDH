"""
Alternative QASM to MBQC Pattern Converter
Uses only Qiskit and standard Python libraries - no graphix required.
NOTE: this generator is incredibly inneficient, thus why its currently not used
"""

import os
import sys
import json
from pathlib import Path
import argparse
from qiskit import QuantumCircuit
import numpy as np
from typing import List, Dict, Tuple, Set


class MBQCPattern:
    """Simple MBQC pattern representation."""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.nodes = []  # List of node IDs
        self.edges = []  # List of (node1, node2) edges
        self.measurements = []  # List of measurement commands
        self.corrections = []  # List of correction commands
        self.input_nodes = []
        self.output_nodes = []
        self.node_counter = 0
        
    def add_node(self) -> int:
        """Add a new node and return its ID."""
        node_id = self.node_counter
        self.nodes.append(node_id)
        self.node_counter += 1
        return node_id
    
    def add_edge(self, node1: int, node2: int):
        """Add an edge between two nodes."""
        if (node1, node2) not in self.edges and (node2, node1) not in self.edges:
            self.edges.append((node1, node2))
    
    def add_measurement(self, node: int, angle: float, plane: str = 'XY'):
        """Add a measurement command."""
        self.measurements.append({
            'node': node,
            'angle': angle,
            'plane': plane
        })
    
    def to_dict(self) -> Dict:
        """Convert pattern to dictionary."""
        return {
            'n_qubits': self.n_qubits,
            'n_nodes': len(self.nodes),
            'nodes': self.nodes,
            'edges': self.edges,
            'input_nodes': self.input_nodes,
            'output_nodes': self.output_nodes,
            'measurements': self.measurements,
            'n_measurements': len(self.measurements),
            'n_edges': len(self.edges)
        }


class QASMtoMBQC:
    """Converter from QASM to MBQC patterns."""
    
    def __init__(self, qiskit_circuit: QuantumCircuit):
        self.circuit = qiskit_circuit
        self.pattern = MBQCPattern(qiskit_circuit.num_qubits)
        self.qubit_to_node = {}  # Maps logical qubits to current nodes
        
    def initialize_graph(self):
        """Initialize input nodes for each qubit."""
        for i in range(self.circuit.num_qubits):
            node = self.pattern.add_node()
            self.qubit_to_node[i] = node
            self.pattern.input_nodes.append(node)
    
    def convert_hadamard(self, qubit: int):
        """Convert Hadamard gate to MBQC pattern."""
        # H gate: Create new node, entangle, measure in X basis
        current_node = self.qubit_to_node[qubit]
        new_node = self.pattern.add_node()
        
        self.pattern.add_edge(current_node, new_node)
        self.pattern.add_measurement(current_node, 0, 'XY')
        
        self.qubit_to_node[qubit] = new_node
    
    def convert_x(self, qubit: int):
        """Convert X gate to MBQC pattern."""
        current_node = self.qubit_to_node[qubit]
        new_node = self.pattern.add_node()
        
        self.pattern.add_edge(current_node, new_node)
        self.pattern.add_measurement(current_node, np.pi, 'XY')
        
        self.qubit_to_node[qubit] = new_node
    
    def convert_y(self, qubit: int):
        """Convert Y gate to MBQC pattern."""
        current_node = self.qubit_to_node[qubit]
        new_node = self.pattern.add_node()
        
        self.pattern.add_edge(current_node, new_node)
        self.pattern.add_measurement(current_node, np.pi/2, 'XZ')
        
        self.qubit_to_node[qubit] = new_node
    
    def convert_z(self, qubit: int):
        """Convert Z gate to MBQC pattern."""
        current_node = self.qubit_to_node[qubit]
        new_node = self.pattern.add_node()
        
        self.pattern.add_edge(current_node, new_node)
        self.pattern.add_measurement(current_node, 0, 'YZ')
        
        self.qubit_to_node[qubit] = new_node
    
    def convert_s(self, qubit: int):
        """Convert S gate (phase) to MBQC pattern."""
        # S = RZ(π/2)
        self.convert_rz(qubit, np.pi/2)
    
    def convert_t(self, qubit: int):
        """Convert T gate to MBQC pattern."""
        # T = RZ(π/4)
        self.convert_rz(qubit, np.pi/4)
    
    def convert_rx(self, qubit: int, angle: float):
        """Convert RX rotation to MBQC pattern."""
        current_node = self.qubit_to_node[qubit]
        new_node = self.pattern.add_node()
        
        self.pattern.add_edge(current_node, new_node)
        self.pattern.add_measurement(current_node, -angle/2, 'YZ')
        
        self.qubit_to_node[qubit] = new_node
    
    def convert_ry(self, qubit: int, angle: float):
        """Convert RY rotation to MBQC pattern."""
        current_node = self.qubit_to_node[qubit]
        new_node = self.pattern.add_node()
        
        self.pattern.add_edge(current_node, new_node)
        self.pattern.add_measurement(current_node, -angle/2, 'XZ')
        
        self.qubit_to_node[qubit] = new_node
    
    def convert_rz(self, qubit: int, angle: float):
        """Convert RZ rotation to MBQC pattern."""
        current_node = self.qubit_to_node[qubit]
        new_node = self.pattern.add_node()
        
        self.pattern.add_edge(current_node, new_node)
        self.pattern.add_measurement(current_node, angle/2, 'XY')
        
        self.qubit_to_node[qubit] = new_node
    
    def convert_cnot(self, control: int, target: int):
        """Convert CNOT gate to MBQC pattern."""
        # CNOT requires creating entanglement between control and target
        control_node = self.qubit_to_node[control]
        target_node = self.qubit_to_node[target]
        
        # Create new nodes for both qubits
        new_control = self.pattern.add_node()
        new_target = self.pattern.add_node()
        helper = self.pattern.add_node()
        
        # Add edges to create the CNOT pattern
        self.pattern.add_edge(control_node, helper)
        self.pattern.add_edge(target_node, helper)
        self.pattern.add_edge(helper, new_control)
        self.pattern.add_edge(helper, new_target)
        
        # Measurements
        self.pattern.add_measurement(control_node, 0, 'XY')
        self.pattern.add_measurement(target_node, 0, 'XY')
        self.pattern.add_measurement(helper, 0, 'XY')
        
        # Update qubit mappings
        self.qubit_to_node[control] = new_control
        self.qubit_to_node[target] = new_target
    
    def convert_cz(self, control: int, target: int):
        """Convert CZ gate to MBQC pattern."""
        # CZ can be implemented with CNOT sandwiched by Hadamards
        self.convert_hadamard(target)
        self.convert_cnot(control, target)
        self.convert_hadamard(target)
    
    def convert_swap(self, qubit1: int, qubit2: int):
        """Convert SWAP gate to MBQC pattern."""
        # SWAP = CNOT(1,2) + CNOT(2,1) + CNOT(1,2)
        self.convert_cnot(qubit1, qubit2)
        self.convert_cnot(qubit2, qubit1)
        self.convert_cnot(qubit1, qubit2)
    
    def finalize_pattern(self):
        """Finalize the pattern by setting output nodes."""
        for qubit in range(self.circuit.num_qubits):
            self.pattern.output_nodes.append(self.qubit_to_node[qubit])
    
    def convert(self) -> MBQCPattern:
        """Convert the entire circuit to MBQC pattern."""
        self.initialize_graph()
        
        # Process each gate in the circuit
        for instruction, qargs, cargs in self.circuit.data:
            gate_name = instruction.name.lower()
            qubit_indices = [self.circuit.find_bit(q).index for q in qargs]
            
            try:
                if gate_name == 'h':
                    self.convert_hadamard(qubit_indices[0])
                elif gate_name == 'x':
                    self.convert_x(qubit_indices[0])
                elif gate_name == 'y':
                    self.convert_y(qubit_indices[0])
                elif gate_name == 'z':
                    self.convert_z(qubit_indices[0])
                elif gate_name == 's':
                    self.convert_s(qubit_indices[0])
                elif gate_name == 't':
                    self.convert_t(qubit_indices[0])
                elif gate_name == 'sdg':
                    self.convert_rz(qubit_indices[0], -np.pi/2)
                elif gate_name == 'tdg':
                    self.convert_rz(qubit_indices[0], -np.pi/4)
                elif gate_name in ['cx', 'cnot']:
                    self.convert_cnot(qubit_indices[0], qubit_indices[1])
                elif gate_name == 'cz':
                    self.convert_cz(qubit_indices[0], qubit_indices[1])
                elif gate_name == 'swap':
                    self.convert_swap(qubit_indices[0], qubit_indices[1])
                elif gate_name == 'rx':
                    angle = float(instruction.params[0])
                    self.convert_rx(qubit_indices[0], angle)
                elif gate_name == 'ry':
                    angle = float(instruction.params[0])
                    self.convert_ry(qubit_indices[0], angle)
                elif gate_name == 'rz':
                    angle = float(instruction.params[0])
                    self.convert_rz(qubit_indices[0], angle)
                elif gate_name in ['cp', 'crz', 'cu1']:
                    # Controlled phase gates (cu1 is equivalent to cp/crz)
                    angle = float(instruction.params[0])
                    # Decompose into RZ gates and CNOTs
                    self.convert_rz(qubit_indices[0], angle/2)
                    self.convert_rz(qubit_indices[1], angle/2)
                    self.convert_cnot(qubit_indices[0], qubit_indices[1])
                    self.convert_rz(qubit_indices[1], -angle/2)
                    self.convert_cnot(qubit_indices[0], qubit_indices[1])
                elif gate_name in ['u1', 'p']:
                    # U1 gate and P gate (phase gate) are both just RZ
                    angle = float(instruction.params[0])
                    self.convert_rz(qubit_indices[0], angle)
                elif gate_name == 'u2':
                    # U2(φ,λ) = RZ(λ) · RY(π/2) · RZ(φ)
                    phi = float(instruction.params[0])
                    lam = float(instruction.params[1])
                    self.convert_rz(qubit_indices[0], phi)
                    self.convert_ry(qubit_indices[0], np.pi/2)
                    self.convert_rz(qubit_indices[0], lam)
                elif gate_name == 'u3':
                    # U3(θ,φ,λ) = RZ(λ) · RY(θ) · RZ(φ)
                    theta = float(instruction.params[0])
                    phi = float(instruction.params[1])
                    lam = float(instruction.params[2])
                    self.convert_rz(qubit_indices[0], phi)
                    self.convert_ry(qubit_indices[0], theta)
                    self.convert_rz(qubit_indices[0], lam)
                elif gate_name == 'ccx':
                    # Toffoli gate (controlled-controlled-X)
                    # Simplified decomposition
                    print(f"  Decomposing Toffoli gate")
                    control1, control2, target = qubit_indices[0], qubit_indices[1], qubit_indices[2]
                    # Standard Toffoli decomposition
                    self.convert_hadamard(target)
                    self.convert_cnot(control2, target)
                    self.convert_rz(target, -np.pi/4)
                    self.convert_cnot(control1, target)
                    self.convert_rz(target, np.pi/4)
                    self.convert_cnot(control2, target)
                    self.convert_rz(target, -np.pi/4)
                    self.convert_cnot(control1, target)
                    self.convert_rz(control2, np.pi/4)
                    self.convert_rz(target, np.pi/4)
                    self.convert_hadamard(target)
                    self.convert_cnot(control1, control2)
                    self.convert_rz(control1, np.pi/4)
                    self.convert_rz(control2, -np.pi/4)
                    self.convert_cnot(control1, control2)
                elif gate_name == 'cu3':
                    # Controlled U3 gate
                    theta = float(instruction.params[0])
                    phi = float(instruction.params[1])
                    lam = float(instruction.params[2])
                    control, target = qubit_indices[0], qubit_indices[1]
                    # Simplified decomposition
                    self.convert_rz(target, (lam + phi)/2)
                    self.convert_cnot(control, target)
                    self.convert_rz(target, -(lam + phi)/2)
                    self.convert_ry(target, -theta/2)
                    self.convert_cnot(control, target)
                    self.convert_ry(target, theta/2)
                    self.convert_rz(target, phi)
                elif gate_name == 'sx':
                    # SX gate (sqrt(X)) = RX(π/2)
                    self.convert_rx(qubit_indices[0], np.pi/2)
                elif gate_name == 'sxdg':
                    # SX-dagger gate = RX(-π/2)
                    self.convert_rx(qubit_indices[0], -np.pi/2)
                elif gate_name == 'cy':
                    # Controlled-Y gate
                    control, target = qubit_indices[0], qubit_indices[1]
                    self.convert_s(target)
                    self.convert_cnot(control, target)
                    self.convert_rz(target, -np.pi/2)
                elif gate_name == 'ch':
                    # Controlled-Hadamard gate
                    control, target = qubit_indices[0], qubit_indices[1]
                    self.convert_ry(target, np.pi/4)
                    self.convert_cnot(control, target)
                    self.convert_ry(target, -np.pi/4)
                elif gate_name == 'rzz':
                    # RZZ gate (two-qubit Z rotation)
                    angle = float(instruction.params[0])
                    qubit1, qubit2 = qubit_indices[0], qubit_indices[1]
                    self.convert_cnot(qubit1, qubit2)
                    self.convert_rz(qubit2, angle)
                    self.convert_cnot(qubit1, qubit2)
                elif gate_name in ['measure', 'barrier', 'id', 'i']:
                    # Skip these gates
                    pass
                else:
                    print(f"  Warning: Unsupported gate '{gate_name}' - skipping")
            except Exception as e:
                print(f"  Error processing gate {gate_name}: {e}")
        
        self.finalize_pattern()
        return self.pattern


def convert_qasm_to_mbqc(qasm_file_path: str, output_dir: str = "MBQC", verbose: bool = True) -> bool:
    """
    Convert a QASM file to MBQC pattern.
    
    Args:
        qasm_file_path: Path to the QASM file
        output_dir: Directory to save output files
        verbose: Print detailed progress
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        file_name = Path(qasm_file_path).name
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {file_name}")
            print(f"{'='*60}")
        else:
            print(f"Processing: {file_name}...", end=" ", flush=True)
        
        # Load QASM circuit
        circuit = QuantumCircuit.from_qasm_file(qasm_file_path)
        if verbose:
            print(f"✓ Loaded circuit: {circuit.num_qubits} qubits, {len(circuit.data)} gates")
        
        # Convert to MBQC
        converter = QASMtoMBQC(circuit)
        pattern = converter.convert()
        if verbose:
            print(f"✓ Converted to MBQC: {pattern.n_nodes} nodes, {len(pattern.edges)} edges")
        else:
            print(f"✓ ({pattern.n_nodes} nodes, {len(pattern.edges)} edges)", flush=True)
        
        # Generate output filenames
        base_name = Path(qasm_file_path).stem
        output_base = os.path.join(output_dir, base_name)
        
        # Save pattern as JSON (most important file)
        pattern_dict = pattern.to_dict()
        json_file = f"{output_base}_pattern.json"
        with open(json_file, 'w') as f:
            json.dump(pattern_dict, f, indent=2)
        
        if not verbose:
            print(f"✓ Saved")
        else:
            print(f"✓ Saved pattern: {json_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def process_directory(input_dir: str, output_dir: str = "mbqc_patterns", max_qubits: int = None):
    """
    Process all QASM files in a directory.
    
    Args:
        input_dir: Directory containing QASM files
        output_dir: Directory to save output files
        max_qubits: Maximum number of qubits to process (None for no limit)
    """
    import time
    
    qasm_files = sorted(list(Path(input_dir).glob("*.qasm")))
    
    if not qasm_files:
        print(f"No QASM files found in {input_dir}")
        return
    
    # Filter by qubit count if specified
    if max_qubits is not None:
        filtered_files = []
        skipped = 0
        print(f"Filtering circuits with ≤{max_qubits} qubits...")
        for qasm_file in qasm_files:
            try:
                circuit = QuantumCircuit.from_qasm_file(str(qasm_file))
                if circuit.num_qubits <= max_qubits:
                    filtered_files.append(qasm_file)
                else:
                    skipped += 1
            except Exception as e:
                print(f"  Warning: Could not read {qasm_file.name}: {e}")
                skipped += 1
        
        qasm_files = filtered_files
        print(f"  Selected: {len(qasm_files)} files")
        print(f"  Skipped: {skipped} files (>{max_qubits} qubits or unreadable)\n")
    
    if not qasm_files:
        print(f"No files to process after filtering")
        return
    
    print(f"Processing {len(qasm_files)} QASM files")
    print(f"Output directory: {output_dir}\n")
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, qasm_file in enumerate(qasm_files, 1):
        print(f"[{i}/{len(qasm_files)}] ", end="")
        if convert_qasm_to_mbqc(str(qasm_file), output_dir, verbose=False):
            successful += 1
        else:
            failed += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Successfully processed: {successful}/{len(qasm_files)}")
    print(f"  Failed: {failed}/{len(qasm_files)}")
    print(f"  Time elapsed: {elapsed:.1f} seconds ({elapsed/len(qasm_files):.2f}s per file)")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert QASM files to MBQC patterns (no graphix required)"
    )
    parser.add_argument(
        "input",
        help="QASM file or directory containing QASM files"
    )
    parser.add_argument(
        "-o", "--output",
        default="mbqc_patterns",
        help="Output directory (default: mbqc_patterns)"
    )
    parser.add_argument(
        "-q", "--max-qubits",
        type=int,
        default=None,
        help="Maximum number of qubits to process (default: no limit)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        convert_qasm_to_mbqc(str(input_path), args.output)
    elif input_path.is_dir():
        process_directory(str(input_path), args.output, max_qubits=args.max_qubits)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()