import pandas as pd

from pathlib import Path


mean = 309.0
result_path = Path("notebooks") / "results" / "GTSRB_cnn_A2_norm_l2_eps_36.csv"
print(result_path.name)
df = pd.read_csv(result_path)
df["explanation_size"] = 1024 - df["freed_features"]
print("Mean explanation size: ", df["explanation_size"].mean())

df["size_difference"] = mean - df["explanation_size"]
d = df["size_difference"]

print("Mean difference from our mean: ", d.mean())
print("Mean difference (Ours - FAME):", d.mean())
print("Sample variance of differences:", d.var(ddof=1))
print("Sample Standard Deviation of differences:", d.std(ddof=1))
print("Minimum difference:", d.min())
print("Maximum difference:", d.max())