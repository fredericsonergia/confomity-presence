from Image import Image
from helpers import my_arg_max
from scipy import ndimage
import numpy as np

class Protection:
    """ class defining the protection object on a photograph"""

    def __init__(self,imagePath:str)->None:
        """ constructor of class Protection"""
        self.__image = Image(imagePath)
        self.__axis_from_edges = None
        self.__right_direction = None

    def get_right_direction(self):
        if self.__right_direction == None:
            angle = None
            counts = 0
            ordinate = None
            for i in range(-20,21):
                rotated_i = ndimage.rotate(self.__image.edges_detection(), i, reshape=False)
                list_of_ones = ones_per_line(rotated_i)
                argmax = my_arg_max(list_of_ones)
                if argmax[1] > counts:
                    counts = argmax[1]
                    ordinate = argmax[0]
                    angle = i
            rotated = ndimage.rotate(self.__image.edges_detection(), i, reshape=False)
            for i in range(rotated.shape[0]):
                for j in range(rotated.shape[1]):
                    if rotated[i][j] > 0:
                        rotated[i][j] = 255
            list_of_ones = ones_per_line(rotated)
            argmax = my_arg_max(list_of_ones)
            self.__right_direction = {"counts":argmax[1], "ordinate": ordinate, "angle": angle}
        return self.__right_direction


    def get_axis_from_edges(self)->(tuple,tuple):
        """
            @parameters: the class itself
            @returns: two points delimiting the horizontal axis approximating the protection. Got from dilated edges binary image
        """
        if self.__axis_from_edges == None:
            obj = self.get_right_direction()
            angle = obj["angle"]
            ordinate = obj["ordinate"]
            angle = angle*np.pi/180
            slope = np.tan(angle)
            rotation_center = (self.__image.get_resolution()[1]/2,self.__image.get_resolution()[0]/2)
            basic_point = (0,ordinate)
            converted_point = ((basic_point[0]-rotation_center[0])*np.cos(-angle)+(basic_point[1]-rotation_center[1])*np.sin(-angle)+rotation_center[0], -(basic_point[0]-rotation_center[0])*np.sin(-angle)+(basic_point[1]-rotation_center[1])*np.cos(-angle)+rotation_center[1])
            origin_ordinate = converted_point[1]-slope*converted_point[0]
            start_point = (0,origin_ordinate)
            end_point = (self.__image.get_resolution()[1], slope * self.__image.get_resolution()[1]+origin_ordinate)
            self.__axis_from_edges = (start_point, end_point)
        return self.__axis_from_edges


    def check_protection(self)-> bool:
        right_dir = self.get_right_direction()
        min_of_255_on_a_line = int(self.__image.resolution[1]/5)
        #print("the min",min_of_255_on_a_line)
        #print("the counts",right_dir["counts"]/255)
        return right_dir["counts"]/255 > min_of_255_on_a_line

def ones_per_line(np_arr)->list:
    """
        @parameters: an array drescribing a binary image (in our case: edges or dilated edges)
        @returns: a list of number of ones per line of the image
    """
    number_of_ones = []
    counter = 0
    for element in np_arr:
        if counter <= np_arr.shape[0]/2:
            #print("what it looks like", element[300])
            number_of_ones.append(sum(element))
            counter += 1
        else:
            break
    return number_of_ones

if __name__ == '__main__':
    #print("okay, brother")
    my_protection = Protection("im1-rotate.png")
    #print("protection is checked", my_protection.check_protection())
    #print(my_protection.get_axis_from_edges())
