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
df1 = pd.read_csv('comparison_results_10_qubit-level_weighted.csv')

# Read the second CSV file
df2 = pd.read_csv('comparison_results_timed_weighted.csv')

# Select only the desired columns from both dataframes
df1_filtered = df1[output_columns]
df2_filtered = df2[output_columns]

# Combine the dataframes
combined_df = pd.concat([df1_filtered, df2_filtered], ignore_index=True)

# Save to results.csv
combined_df.to_csv('results.csv', index=False)

print(f"Successfully combined {len(df1)} rows from comparison_results_10_qubit-level_weighted.csv")
print(f"and {len(df2)} rows from comparison_results_timed_weighted.csv")
print(f"Total rows in results.csv: {len(combined_df)}")