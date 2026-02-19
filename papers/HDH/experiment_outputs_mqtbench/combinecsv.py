import pandas as pd

# Define the desired output columns
output_columns = [
    'circuit', 'num_qubits', 'num_nodes', 'num_quantum_nodes', 
    'num_classical_nodes', 'num_edges', 'k', 'overhead', 'capacity',
    'optimal_cost', 'heuristic_cost', 'ratio', 'brute_force_time',
    'heuristic_time', 'method', 'optimal_qubit_counts', 
    'heuristic_qubit_counts', 'optimal_classical_counts', 
    'heuristic_classical_counts'
]

# Read the first CSV file
df1 = pd.read_csv('results_node_level_fixed_over10.csv')

# Read the second CSV file
df2 = pd.read_csv('results_node_level_fixed.csv')

# Select only the desired columns from both dataframes
df1_filtered = df1[output_columns]
df2_filtered = df2[output_columns]

# Combine the dataframes
combined_df = pd.concat([df1_filtered, df2_filtered], ignore_index=True)

# Save to results.csv
combined_df.to_csv('results.csv', index=False)
