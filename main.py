from typing import Set, List

import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from cluster import Cluster


def plot_cluster(clusters: Set[Cluster], fig_idx: int):
    plt.figure()
    if not os.path.exists("output"):
        os.mkdir("output")
    for i, cluster in enumerate(clusters):
        coords = np.asarray(list(cluster.items))
        plt.scatter(coords[:, 0], coords[:, 1], color=list(mpl.colors.cnames.values())[i+10])

    plt.xlabel("CPU")
    plt.ylabel("MEM")
    plt.savefig(os.path.join("output", f"{fig_idx}.png"))
    plt.close()
    return len(clusters)

def main():
    arr = pd.read_csv(os.path.join("data", input_csv))
    arr = torch.from_numpy(arr.values)
    sample: torch.Tensor = arr[:,:2]
    count: torch.Tensor = arr[:, 2]

    # print(sample.shape)
    clusters: set = set()
    for (cpu, mem), num in zip(sample, count):  # 初始时每种类型一个簇
        clusters.add(Cluster(float(cpu), float(mem), num))

    fig_idx = 0

    plot_cluster(clusters, fig_idx)
    fig_idx += 1

    while len(clusters) > NUM_CLUSTER:
        cluster_list: List[Cluster] = list(clusters)
        min_distance = 999999999999999
        a_idx = -1
        b_idx = -1
        for i in range(len(cluster_list)):
            for j in range(i+1, len(cluster_list)):
                cluster_a: Cluster = cluster_list[i]
                tag_a = cluster_a.tag
                cluster_b: Cluster = cluster_list[j]
                tag_b = cluster_b.tag
                distance = cluster_a.intra_distance(cluster_b)
                if distance < min_distance and (tag_a <= tag_b or tag_b <= tag_a):
                    a_idx = i
                    b_idx = j
                    min_distance = distance
        clusters.clear()
        cluster_list[a_idx].merge(cluster_list[b_idx])
        clusters.add(cluster_list[a_idx])
        for k in range(len(cluster_list)):
            if k != a_idx and k != b_idx:
                clusters.add(cluster_list[k])

        plot_cluster(clusters, fig_idx)
        print(len(clusters))
        fig_idx += 1
    result: list = sample[sample[:, 0] > 0].tolist()
    division = {}
    for cluster in clusters:
        tag = cluster.tag
        division[(tag[0], tag[1])] = float(cluster.num)
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
