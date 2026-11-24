import pandas as pd

from pathlib import Path

eps = 0.36
mean_sizes_our = {
    34: 309.0,
    35: 346.0,
    36: 369.0
}

int_eps = int(eps*100)
mean = mean_sizes_our[int_eps]
result_path = Path("notebooks") / "results" / f"GTSRB_cnn_A2_norm_l2_eps_{int_eps}.csv"
print(result_path.name)
df = pd.read_csv(result_path)
df["explanation_size"] = 1024 - df["freed_features"]
print("Mean explanation size: ", df["explanation_size"].mean())

df["size_difference"] = mean - df["explanation_size"]
d = df["size_difference"]

print("Mean difference (Ours - FAME):", d.mean())
print("Sample variance of differences:", d.var(ddof=1))
print("Sample Standard Deviation of differences:", d.std(ddof=1))
print("Minimum difference:", d.min())
print("Maximum difference:", d.max())