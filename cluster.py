import numpy as np

def sigmoid(arr: np.ndarray):
    return 1 / np.exp(1 + np.exp(-1 * arr))

class Cluster:

    def __init__(self, cpu, mem, num: int):
        self.items = {(cpu, mem)}
        self.num = num

    def merge(self, cluster: 'Cluster'):
        self.items.update(cluster.items)
        self.num += cluster.num

    def intra_distance(self, cluster: 'Cluster'):
        arr1 = np.asarray(list(self.items))[:, None]
        arr2 = np.asarray(list(cluster.items))
        # distance = np.abs(arr1 - arr2).sum(axis=-1) / np.sqrt(arr1 * arr2).sum(axis=-1)
        distance = np.abs(np.log2(arr1) - np.log2(arr2)).sum(axis=-1) / sigmoid((arr1 * arr2).sum(axis=-1)) ** 1.2
        return np.mean(distance) * (self.num + cluster.num)

    @property
    def tag(self):
        _res = list(sorted(self.items))
        return _res[-1]