from Ruler import Ruler
from Protection import Protection
from helpers import slope_ordinate, find_intersection, approximate_text_by_point, my_arg_max

class Conformity:
    "Class representing the conformity of a construction site"

    def __init__(self, imagePath:str)->None:
        """Constructor of class Conformity """
        self.__ruler = Ruler(imagePath)
        self.__protection = Protection(imagePath)
        self.__intersection = None
        self.__distance = None
        self.__conformity = None
        self.__max_digit_read = None
        self.__conformity_distance = 12.0

    def get_intersection(self)->(float,float):
        """
            @parameters: object itself
            @returns: point representing the intersection of ruler and protection of the object
        """
        if self.__intersection == None:
            ruler_start_point, ruler_end_point = self.__ruler.get_axis()
            print("the axis is: ", self.__ruler.get_axis())
            print("ruler point coordinates ",ruler_start_point, ruler_end_point)
            protection_start_point, protection_end_point = self.__protection.get_axis_from_edges()
            print("protection point coordinates ",protection_start_point, protection_end_point)
            ruler_slope, ruler_ordinate = slope_ordinate(ruler_start_point, ruler_end_point)
            protection_slope, protection_ordinate = slope_ordinate(protection_start_point, protection_end_point)
            print("ruler slope and ordinate ",ruler_slope, ruler_ordinate)
            print("protection_slope_ordinate", protection_slope, protection_ordinate)
            self.__intersection = find_intersection(ruler_slope, ruler_ordinate, protection_slope, protection_ordinate)
        return self.__intersection

    def get_max_digit_read(self)->(float,float):
        """
            @parameters: object itself
            @returns: the maximum of the digits on the ruler under the protection axis. Alongside with its ordinate.
        """
        if self.__max_digit_read == None:
            ruler_digits = self.__ruler.get_digits()
            ruler_digits = [(element[0],approximate_text_by_point(element)[1]) for element in ruler_digits]
            print("0: ",ruler_digits)
            intersection = self.get_intersection()
            ruler_digits = [element for element in ruler_digits if element[1] >= intersection[1]]
            ruler_digits_ordinates = [element[1] for element in ruler_digits]
            ruler_digits_value = [int(element[0]) for element in ruler_digits]
            index,value = my_arg_max(ruler_digits_value)
            self.__max_digit_read = (value,ruler_digits_ordinates[index])
        return self.__max_digit_read

    def get_distance(self)->float:
        """
            @parameters: object itself
            @returns: the distance between the protection and the chimney
        """
        if self.__distance == None:
            pixel_cm_scale = self.__ruler.get_pixel_centimeter_scale()
            intersection = self.get_intersection()
            max_digit_read = self.get_max_digit_read()
            pix_delta_from_inter_to_max_digit = max_digit_read[1]-intersection[1]
            cm_delta_from_inter_to_max_digit=pixel_cm_scale[1]*pix_delta_from_inter_to_max_digit/pixel_cm_scale[0]
            self.__distance = cm_delta_from_inter_to_max_digit+max_digit_read[0]
        return self.__distance

    def get_conformity(self)->bool:
        """
            @parameters: object itself
            @returns: True if conformity site is conform i.e distance > self.__minimum_distance. False otherwise.
        """
        if self.__conformity == None:
            self.__conformity =  self.get_distance() > self.__conformity_distance
        return self.__conformity

    def get_conformity_distance(self)->float:
        """
            @parameters: object itself
            @returns: return the current conformity distance
        """
        return self.__conformity_distance

    def set_conformity_distance(self,new_distance:float)->None:
        """
            @parameters: object itself, and the new conformity distance
            @returns: Nothing
        """
        self.__conformity_distance = new_distance

    def get_ruler(self)-> Ruler:
        """
            @parameters: object itself
            @returns: ruler object coresponding to ruler on the construction site
        """
        return self.__ruler

    def get_protection(self) -> Protection:
        """
            @parameters: object itself
            @returns: protection object representing the protection on the construction site
        """
        return self.__protection

#if __name__ == '__main__':
#    print("okay, brother")
#    my_conformity = Conformity("eaf2-rotate.png")
#    print("intersection",my_conformity.get_intersection())
#    print("max digit read",my_conformity.get_max_digit_read())
#    print("work sclae", my_conformity.get_ruler().get_pixel_centimeter_scale())
#    print("digits", my_conformity.get_ruler().get_digits())
#    print("distance ", my_conformity.get_distance()) 
    