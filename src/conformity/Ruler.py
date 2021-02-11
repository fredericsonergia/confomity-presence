import numpy as np
from src.conformity.Image import Image
import keras_ocr
from src.conformity.helpers import slope_ordinate, extend_line, approximate_text_by_point, double_of_box_width, distance_to_line

class Ruler:
    """ class defining Ruler object on a photograph"""

    def __init__(self, model_pipeline, imagePath:str)->None:
        """constructor of class Ruler"""
        self.image = Image(imagePath)
        self.digits = None
        self.__axis = None
        self.__pixel_centimeter_scale = None
        self.__model_pipeline = model_pipeline

    def get_digits(self)->list:
        """
            @parameters: the class itself
            @returns: the digits detected on the ruler. The boxes around the digits
        """
        if self.digits == None:
            images = [self.image.get_resized()]
            #pipeline = pipeline = keras_ocr.pipeline.Pipeline()
            prediction_groups = self.__model_pipeline.recognize(images)
            to_return =[]
            #print('digits_brut', prediction_groups)
            if len(prediction_groups[0]) >0:
                for my_tuple in prediction_groups[0]:
                    try:
                        int(my_tuple[0])
                        to_return.append(my_tuple)
                    except:
                        continue
                accepted_distance = double_of_box_width(to_return[0])/2
                first_point = approximate_text_by_point(to_return[0])
                second_point = approximate_text_by_point(to_return[1])
                slope,ordinate = slope_ordinate(first_point,second_point)
                to_return = []
                for my_tuple in prediction_groups[0]:
                    try:
                        int(my_tuple[0])
                        to_return.append(my_tuple)
                    except:
                        box_point = approximate_text_by_point(my_tuple)
                        distance_to_axis = distance_to_line(slope,ordinate,box_point)
                        if distance_to_axis < accepted_distance:
                            if my_tuple[0] in ['o', 'O']:
                                to_return.append((str(0),my_tuple[1]))
                            #elif my_tuple[0] in ['s','S']:
                                #to_return.append((str(5),my_tuple[1]))
                            else:
                                if len(my_tuple[0])==2:
                                    try:
                                        int(my_tuple[0][0])
                                        unit = my_tuple[0][1]
                                        if unit in ['o', 'O']:
                                            to_return.append((my_tuple[0][0]+'0',my_tuple[1]))
                                        elif unit in ['s','S']:
                                            to_return.append((my_tuple[0][0]+'5',my_tuple[1]))
                                        else:
                                            continue
                                    except:
                                        if my_tuple[0][0]=='l':
                                            unit = my_tuple[0][1]
                                            if unit in ['o', 'O']:
                                                to_return.append(('10',my_tuple[1]))
                                            elif unit in ['s','S']:
                                                to_return.append(('15',my_tuple[1]))
                                        else:
                                            continue
                                else:
                                    continue
                        else:
                            continue
            self.digits = to_return
        return self.digits

    def get_axis(self)->tuple:
        """
            @parameters: the class itself
            @returns: Two dots delimiting the axis of the object
        """
        if self.__axis == None:
            predicted_integers = self.get_digits()
            if len(predicted_integers) >1:
                first_text = predicted_integers[0]
                second_text = predicted_integers[1]
                #print("text taken for axis", first_text, second_text)
                first_point = approximate_text_by_point(first_text)
                second_point = approximate_text_by_point(second_text)
                slope, ordinate = slope_ordinate(first_point, second_point)
                #print("first points from rulers taken before extent", first_point, second_point)
                start_point, end_point = extend_line(slope,ordinate,0,self.image.get_resolution()[0])
                self.__axis= start_point,end_point
        return self.__axis

    def get_pixel_centimeter_scale(self) -> tuple:
        """
            @parameters: the object itself
            @returns: the scale pexel->centimeter coresponding to the object
        """
        if self.__pixel_centimeter_scale == None:
            digits = self.get_digits()
            digits = [(int(element[0]),approximate_text_by_point(element)[1]) for element in digits]
            differences = []
            prev = digits[0]
            for element in digits[1:]:
                current = element
                differences.append((abs(current[0]-prev[0]), abs(current[1]-prev[1])))
                prev = current
            #print("different scales", differences)
            means = np.mean(differences, axis=0)
            self.__pixel_centimeter_scale = (means[1], means[0])
        return self.__pixel_centimeter_scale

    def check_inclinaison_conformity(self, tolerance) ->bool:
        """
        @parameters: the object itself and a tolorance value on distance change
        @returns: True if inclinaison with respect to horizontal plane. False otherwise
        """
        digits = self.get_digits()
        digits = [(int(element[0]),approximate_text_by_point(element)[1]) for element in digits]
        differences = []
        prev = digits[0]
        for element in digits[1:]:
            current = element
            differences.append((abs(current[0]-prev[0]), abs(current[1]-prev[1])))
            prev = current
        ratios = [element[1]/element[0] for element in differences if element[0]!=0]
        ratios_of_ratios=[]
        prev = ratios[0]
        for element in ratios[1:]:
            if prev > 0:
                ratios_of_ratios.append(element/prev)
            prev = element
        check_bools = [1-tolerance<=val for val in ratios_of_ratios]
        return all(check_bools)

    def check_digits_readability(self) -> bool:
        """
        @parameters: the object itself
        @returns: True if it is considered that digits are well read. False otherwise
        """
        digits = self.get_digits()
        return len (digits) >= 3


if __name__ == '__main__':
    #print("okay, brother")
    my_ruler = Ruler("im1-rotate.png")
    #print(my_ruler.get_digits())
    #print("are enough digits read? ", my_ruler.check_digits_readability())
    #print("is ok for inclinaison? ",my_ruler.check_inclinaison_conformity(15/100))
    
        

