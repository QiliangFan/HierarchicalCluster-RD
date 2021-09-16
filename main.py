import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import OrderedDict


def compute_distance(data: torch.Tensor, clusters: dict):
    num_sample = data.shape[0]
    distance_matrix = torch.zeros((num_sample, num_sample))
    for i in range(num_sample):
        for j in range(i + 1, num_sample):
            if data[i][0] <= data[j][0] and data[i][1] <= data[j][1]:
                distance_matrix[i][j] = distance_matrix[j][i] = ((torch.abs(data[i][0] - data[j][0]))).sum()
    return distance_matrix


def plot_cluster(clusters: dict, fig_idx: int):
    plt.figure()
    if not os.path.exists("output"):
        os.mkdir("output")
    trim_clusters = [tuple(k for k in v) for v in clusters.values()]
    for i, clusters in enumerate(trim_clusters):
        coords = np.asarray(clusters)
        plt.scatter(coords[:, 0], coords[:, 1], color=list(mpl.colors.cnames.values())[i+10])

    plt.xlabel("CPU")
    plt.ylabel("MEM")
    plt.savefig(os.path.join("output", f"{fig_idx}.png"))
    plt.close()
    return len(trim_clusters)

def main():
    arr = pd.read_csv(os.path.join("data", input_csv))
    arr = torch.from_numpy(arr.values)
    sample: torch.Tensor = arr[:,:2]
    count: torch.Tensor = arr[:, 2]
    count_map = {}
    for (cpu, mem), num in zip(sample, count):
        count_map[(float(cpu), float(mem))] = int(num)
    raw_sample = sample.clone()
    # print(sample.shape)
    clusters = OrderedDict({i: {(float(raw_sample[i][0]), float(raw_sample[i][1]))} for i in range(raw_sample.shape[0])})
    fig_idx = 0

    num_cluster = plot_cluster(clusters, fig_idx)
    fig_idx += 1

    while len(sample) > NUM_CLUSTER:
        distance_matrix = compute_distance(sample, clusters)
        max_idx = torch.where(
            distance_matrix == torch.where(distance_matrix > 0, distance_matrix, distance_matrix.max() + 1).min())
        # print(max_idx)
        a, b = int(max_idx[0][0]), int(max_idx[1][0])
        if sample[a][0] <= sample[b][0] and sample[a][1] <= sample[b][1]:
            remove_idx = a
        else:
            remove_idx = b
        clusters[a] = clusters[b] = clusters[a] | clusters[b]

        sample = sample[torch.arange(sample.shape[0]) != remove_idx]
        del clusters[remove_idx]
        clusters = OrderedDict({i: v for i, v in enumerate(clusters.values())})
        plot_cluster(clusters, fig_idx)
        # print(len(sample))
        fig_idx += 1
    result: list = sample[sample[:, 0] > 0].tolist()
    division = {}
    for idx in range(sample.shape[0]):
        division[(float(sample[idx][0]), float(sample[idx][1]))] = sum(count_map[(cpu, mem)] for cpu, mem in clusters[int(idx)])
    result.sort()
    print(result)
    _division = division.copy()

    for k in division:
        division[k] /= sum(_division.values())
    print(division)


if __name__ == "__main__":
    input_csv = "test.csv"

    NUM_CLUSTER = 5
    main()
