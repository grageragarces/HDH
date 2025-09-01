import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("partitions.csv")

# Ensure your DataFrame has columns: model_type, partitioner, partitions, cost

# Step 1: group and sum cost per model + partitioner
agg = df.groupby(["model_type", "partitioner"], as_index=False)["cost"].sum()

# Step 2: pivot for easy plotting (models on rows, partitioners as columns)
pivot = agg.pivot(index="model_type", columns="partitioner", values="cost")

# Step 3: plot as grouped bar chart
ax = pivot.plot(kind="bar", figsize=(10,6))

# Step 4: formatting
plt.title("Summed Costs per Model and Partitioner")
plt.xlabel("Model Type")
plt.ylabel("Total Cost")
plt.legend(title="Partitioner")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_cost_comparison.png")
