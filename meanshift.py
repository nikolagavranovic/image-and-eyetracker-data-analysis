import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math

from dataloader import Dataloader


class MeanShift:
    def __init__(self, radius, data):
        self.radius = radius
        self.data = data.astype(float)

    def fit_gauss(self, sd, max_iter=500):
        centroids = np.array(self.data)

        optimized = False
        iter = 0
        # while cluster change and iteration maximum is not exceeded
        while not optimized and iter < max_iter:
            new_centroids = []
            # iterating through centeroids and finding new average (mean center) for each
            for i in range(len(centroids)):
                centroid_old = centroids[i]
                # formula is sum(x[i]*exp(-1/2*||centroid - xi||^2/h^2))/sum(exp(-1/2*||centroid - xi||^2/h^2))
                sum_nom = 0  # sum in nominator
                sum_den = 0  # sum in denominator

                for el in self.data:
                    # apply formula for gaussian meanshift
                    dist = np.linalg.norm(el - centroid_old)
                    sum_nom += el * math.exp(-1 / 2 * (dist / sd) ** 2)
                    sum_den += math.exp(-1 / 2 * (dist / sd) ** 2)

                centroid_new = sum_nom / sum_den
                new_centroids.append(centroid_new)

            prev_centroids = np.array(centroids)
            centroids = np.array(new_centroids)

            optimized = True
            # in case something changed, go on
            if len(centroids) == len(prev_centroids):
                # if difference between current and previous centroids are less than threshold, than it has converged
                if np.max(np.linalg.norm(prev_centroids - centroids, axis=1)) > 0.1:
                    optimized = False
            else:
                optimized = False  # in case lengths are different, there is change

            iter += 1

        # rounding centroids and making uniques set
        centroids = np.round(centroids)
        uniques = np.unique(centroids, axis=0)
        # making clusterizing dictionary for each unqiue centroid
        clusterized = {}
        for i in range(len(uniques)):
            clusterized[i] = []
        # for all ccentroids find index of unique centroid (this represents cluster)
        # index of centroid is the same as index of data
        for i in range(len(centroids)):
            for j in range(len(uniques)):
                if np.all(centroids[i] == uniques[j]):
                    ind = j
                    break
            clusterized[ind].append(self.data[i])

        # convert list to numpy array
        for i in range(len(uniques)):
            clusterized[i] = np.array(clusterized[i])

        self.uniques = uniques
        self.clusterized = clusterized

    def gaussian(self, x, xi, sd):
        """Helper function for plotting distribution"""
        return (
            1
            / (2 * math.pi * sd**2)
            * math.exp(
                -1 / 2 * (((x[0] - xi[0]) / sd) ** 2 + ((x[1] - xi[1]) / sd) ** 2)
            )
        )

    def plot_gaussian_distribution(self, height, width, sd):
        # complexity of this is cubic and it will execute very slowly, so it should be used just for plotting and analizing data
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(shape=(height, width))

        Z = np.zeros(shape=(len(y), len(x)))
        for i in range(x.size):
            for j in range(y.size):
                for d in self.data:
                    Z[j, i] += self.gaussian((x[i], y[j]), d, sd)
            print(i)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        plt.show()

    def plot_clusters(self):
        fig, ax = plt.subplots(figsize = (7, 9))

        colors = np.random.rand(
            len(self.uniques), 3
        )  # making n random colors for graph

        # for each cluster and it's data
        for k, v in self.clusterized.items():
            # if there is any data for cluster (and there must be on original dataset used for fitting)
            if len(v) != 0:
                x, y = v[:, 0], v[:, 1]
                ax.scatter(x, y, color=colors[k])
            else:
                print(v)

        # star indicates centroid
        ax.scatter(self.uniques[:, 0], self.uniques[:, 1], marker="*", color=colors)
        # ax.invert_yaxis()  # invert y axis (image representation)
        plt.show()

