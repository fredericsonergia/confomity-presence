from Image import Image
from helpers import my_arg_max

class Protection:
    """ class defining the protection object on a photograph"""

    def __init__(self,imagePath:str)->None:
        """ constructor of class Protection"""
        self.__image = Image(imagePath)
        self.__axis_from_edges = None
        self.__axis_from_dilated = None

    def get_axis_from_edges(self)->(tuple,tuple):
        """
            @parameters: the class itself
            @returns: two points delimiting the horizontal axis approximating the protection. Got from dilated edges binary image
        """
        if self.__axis_from_edges == None:
            list_of_ones = ones_per_line(self.__image.edges_detection())
            argmax = my_arg_max(list_of_ones)
            my_dots = ((0,argmax[0]),(self.__image.get_resolution()[0],argmax[0]))
            self.__axis_from_edges = my_dots
        return self.__axis_from_edges

    def get_axis_from_dilated(self)->(tuple, tuple):
        """
            @parameters: the class itself
            @returns: two points delimiting the horizontal axis approximating the protection. Gotten from dilated edges binary image
        """
        if self.__axis_from_dilated == None:
            list_of_ones = ones_per_line(self.__image.dilate(1))
            argmax = my_arg_max(list_of_ones)
            my_dots = ((0,argmax[0]),(self.__image.get_resolution()[0], argmax[0]))
            self.__axis_from_dilated = my_dots
        return self.__axis_from_dilated

def ones_per_line(np_arr)->list:
    """
        @parameters: an array drescribing a binary image (in our case: edges or dilated edges)
        @returns: a list of number of ones per line of the image
    """
    number_of_ones = []
    counter = 0
    for element in np_arr:
        if counter <= np_arr.shape[0]/2:
            number_of_ones.append(sum(element))
            counter += 1
        else:
            break
    return number_of_ones

if __name__ == '__main__':
    print("okay, brother")
    my_protection = Protection("im1-rotate.png")
    print(my_protection.get_axis_from_edges())
    print(my_protection.get_axis_from_dilated())
