import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


class LaneLines:

    def __init__(self, img):
        self.img = img

    def grayscale(self):
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

    def canny(self, low_threshold, high_threshold):
        return cv2.Canny(self.img, low_threshold, high_threshold)
