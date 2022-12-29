from dataloader import Dataloader
from heatmap import Heatmap
from meanshift import MeanShift
import numpy as np

if __name__ == "__main__":
    dl = Dataloader("Heineken.jpg", "Heineken.csv")
    dl.filter_nonvalid_data()
    dl.filter_isolated_pixels(30, 1)
    img, df, x, y = dl.get_data()

    # creating a heatmap
    hm = Heatmap(ksize=(31, 31), sd=20)
    hm.make_heatmap(img, x, y)
    # optionaly, show image with heatmap
    # hm.show_image_with_heatmap()
    hm.save_image_with_heatmap(path="Haineken_custom_heatmap.jpg")

    data = np.array([[el[0], el[1]] for el in zip(x, y)])
    ms = MeanShift(50, data)
    ms.fit_gauss(22)
    # ms.clusterize_data(data, ms.centroids, 22)
    ms.plot_clusters()
