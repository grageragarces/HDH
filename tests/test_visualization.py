import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
from hdh.models.circuit import Circuit
from hdh.visualize import plot_hdh
import tempfile
import os

class TestVisualization:
    """Test visualization functions"""
    
    def test_plot_simple_circuit(self):
        """Test plotting a simple circuit"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("cx", [0, 1])
        hdh = circuit.build_hdh()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_hdh(hdh, save_path=f.name)
            assert os.path.exists(f.name)
            assert os.path.getsize(f.name) > 0
            os.unlink(f.name)
    
    def test_plot_with_measurements(self):
        """Test plotting circuit with measurements"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("measure", [0], [0])
        hdh = circuit.build_hdh()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_hdh(hdh, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)
    
    def test_plot_predicted_nodes(self):
        """Test plotting with predicted/actualized nodes"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("measure", [0], [0])
        circuit.add_instruction("x", [1], bits=[0], cond_flag="p")
        hdh = circuit.build_hdh()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_hdh(hdh, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)