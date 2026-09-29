import pandas as pd


filepath = "notebooks/results/CIFAR10_cnn_A2_norm_l2_eps_08.csv"

df = pd.read_csv(filepath)

print(len(df))

perturbed_size = df['freed_features']
exp_size = 1024 - perturbed_size
mean_exp_size = exp_size.mean()
max_exp_size = exp_size.max()
min_exp_size = exp_size.min()

print("mean explanation size: ", mean_exp_size)
print("max explanation size: ", max_exp_size)
print("min explanation size: ", min_exp_size)

times = df['greedy_time']
mean_time = times.mean()
max_time = times.max()
min_time = times.min()

print("mean time: ", mean_time)
print("max time: ", max_time)
print("min time: ", min_time)