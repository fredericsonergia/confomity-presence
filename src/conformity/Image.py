import numpy as np
import cv2
import matplotlib.pyplot as plt

class Image:
    def __init__(self, imagePath):
        self.resolution = (500,500)
        self.basic = cv2.imread(imagePath)
        self.resized = cv2.resize(self.basic, self.resolution, interpolation = cv2.INTER_AREA)
        self.original = cv2.cvtColor(self.resized, cv2.COLOR_BGR2RGB)
    
    def gray_scale(self):
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        return gray

    #Defining functions that gives edges
    def edges_detection(self):
        gray = self.gray_scale()
        edges = cv2.Canny(gray,100,200,apertureSize = 3)
        return edges

    #Defining function that returns dilated binary image from edges
    def dilate(self,iteration):
        kernel = np.ones((5,5),np.uint8)
        edges = self.edges_detection()
        dilation = cv2.dilate(edges,kernel,iterations = 10)
        return dilation

    #Defining resized format getter
    def get_resized(self):
        return self.resized
    
    #Defining resolution getter
    def get_resolution(self):
        return self.resolution

    #Defining original getter
    def get_original(self):
        return self.original

if __name__ == '__main__':
    #print("okay, brother")
    my_image = Image("im1-rotate.png")
    #print(my_image.original.shape)