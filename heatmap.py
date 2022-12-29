import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
from dataloader import Dataloader


class Heatmap:
    def __init__(self, ksize, sd):
        self.ksize = ksize
        self.sd = sd

    def make_heatmap(self, img, x, y):
        height, width = img.shape[0], img.shape[1]
        heatmap = np.zeros(
            shape=(height, width)
        )  # make the heatmap of the same dimensions as input image
        # heatmap[self.y, self.x] = 255  # set coordinates to white

        for x, y in zip(x, y):
            heatmap = cv2.circle(
                heatmap, (x, y), 7, 255, -1
            )  # draw filled circle on heatmap

        # kernel = np.ones((3, 3))  # TODO: coordinates should be as big as the time someone watched them
        # heatmap = cv2.dilate(heatmap, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)   # enlarge coordinates to 5x5 (maybe too big)

        #  Smoothing image with Gaussian function
        for i in range(2):
            heatmap = cv2.GaussianBlur(
                src=heatmap.astype(np.uint8),
                ksize=self.ksize,
                sigmaX=self.sd,
                sigmaY=self.sd,
            )
            # setting all small values (noise) to 0
            heatmap[heatmap < 10] = 0

            heatmap = (255 * (heatmap / np.amax(heatmap))).astype(
                np.uint8
            )  # firstly scale to 0-1 and then scale to 0-255

        heatmap_color = cv2.applyColorMap(
            heatmap, cv2.COLORMAP_JET
        )  # convert to colormap for heatmap plotting

        # neutralizing all colors where black (0) is in original heatmap (for this colormap only blue should be neutralized
        # but in case of other...)
        heatmap_color[:, :, 0] = np.multiply(heatmap_color[:, :, 0], heatmap >= 1)
        heatmap_color[:, :, 1] = np.multiply(heatmap_color[:, :, 1], heatmap >= 1)
        heatmap_color[:, :, 2] = np.multiply(heatmap_color[:, :, 2], heatmap >= 1)

        # making the imposed image
        self.super_imposed_img = cv2.addWeighted(heatmap_color, 0.5, img, 0.5, 0)

        # making a mask where 0 will be on every color and every pixel where heatmap doesnot exist
        heatmap_mask = np.zeros(shape=(height, width, 3))
        heatmap_mask[:, :, [0, 1, 2]] = np.expand_dims(heatmap, axis=2)

        self.super_imposed_img[heatmap_mask == 0] = img[
            heatmap_mask == 0
        ]  # selecting frames that are not covered by heatmap to original
        return heatmap_color, self.super_imposed_img

    def insert_legend(self, img):
        legend_h = 50
        legend_w = 256
        colors = np.ones(shape=(legend_h, legend_w)) * 255
        for i in range(32):
            colors[i, :] = np.arange(256)

        colors = colors.astype(np.uint8)
        colors = cv2.applyColorMap(colors, cv2.COLORMAP_JET)
        colors[32:, :] = 255  # reseting to white

        # putting text to white background
        cv2.putText(
            colors,
            "least dense",
            (0, legend_h - 2),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.55,
            color=(0, 0, 0),
        )
        cv2.putText(
            colors,
            "most dense",
            (175, legend_h - 2),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.55,
            color=(0, 0, 0),
        )

        offset = 5  # offset relative to lower left corner
        img[-legend_h - offset : -offset, offset : offset + legend_w] = colors
        return img

    def show_image_with_heatmap(self):
        cv2.imshow("heatmap", self.insert_legend(self.super_imposed_img))
        cv2.waitKey(0)

    def save_image_with_heatmap(self, path="image.jpg"):
        cv2.imwrite(path, self.insert_legend(self.super_imposed_img))


