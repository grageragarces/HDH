import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
from hdh.models.circuit import Circuit
from hdh.visualize import plot_hdh

class TestVisualization:
    """Test visualization functions"""
    
    def test_plot_simple_circuit(self, tmp_path):
        """Test plotting a simple circuit"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("cx", [0, 1])
        hdh = circuit.build_hdh()
        
        out = tmp_path / "plot.png"
        plot_hdh(hdh, save_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0
    
    def test_plot_with_measurements(self, tmp_path):
        """Test plotting circuit with measurements"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("measure", [0], [0])
        hdh = circuit.build_hdh()
        
        out = tmp_path / "plot.png"
        plot_hdh(hdh, save_path=str(out))
        assert out.exists()
    
    def test_plot_predicted_nodes(self, tmp_path):
        """Test plotting with predicted/actualized nodes"""
        circuit = Circuit()
        circuit.add_instruction("h", [0])
        circuit.add_instruction("measure", [0], [0])
        circuit.add_instruction("x", [1], bits=[0], cond_flag="p")
        hdh = circuit.build_hdh()
        
        out = tmp_path / "plot.png"
        plot_hdh(hdh, save_path=str(out))
        assert out.exists()
