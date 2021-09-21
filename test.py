import matplotlib.pyplot as plt
import pandas as pd
import os

input_csv = "test.csv"
arr = pd.read_csv(os.path.join("data", input_csv)).values
arr = arr[:, :2]

plt.figure()
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.scatter(arr[:, 0], arr[:, 1])
plt.savefig("log_scale.png")
plt.show()
plt.close()