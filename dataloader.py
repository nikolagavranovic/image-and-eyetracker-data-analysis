import cv2
import pandas as pd
from scipy.spatial import distance

class Dataloader():
    """ Loads image and csv file and contains functions for filtering eyetracking data.
    """

    def __init__(self, img_path, csv_path):
        # loading image
        self.img = cv2.imread(img_path)
        self.height, self.width = self.img.shape[0], self.img.shape[1]


        # loading csv into DataFrame
        self.df = pd.read_csv(csv_path, sep=";", names=["timestamp", "x", "y"])

    def filter_nonvalid_data(self):
        # filtering all values that are not in screen (image)
        # the upper left corner has coordinates (0, 0), so negative values are out of screen
        self.df = self.df[
            (self.df["x"] > 0)
            & (self.df["y"] > 0)
            & (self.df["x"] < self.width)   
            & (self.df["y"] < self.height)
        ]

    def get_data(self):
        """
        Getter.
        Returns:
            tuple: Dataframe, array of x coordinates and array of y coordinates
        """
        return self.img, self.df, self.df["x"].to_numpy(), self.df["y"].to_numpy()
    
    def count_neighbours(self, pixel, x, y, radius):
        # ounts neighbours within a radius
        counter = 0
        for i in range(len(x)):
            if distance.euclidean((pixel), (x[i], y[i])) < radius:
                counter += 1

        return counter

    def filter_isolated_pixels(self, radius, n):
        """ Removes all data that have less than n neighbours in radius.

        Args:
            radius (float): Radius
            n (int): Number of neighbours.
        """
        x = self.df["x"].to_numpy()
        y = self.df["y"].to_numpy()
        inx_to_drop = []

        for i in range(len(x)):
            if (
                self.count_neighbours((x[i], y[i]), x, y, radius) <= n
            ):  # every dot will have himself counted as neighbour, so condition is <=
                inx_to_drop.append(i)
        self.df.drop(self.df.index[inx_to_drop], inplace=True)


