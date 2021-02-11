try:
    from src.conformity.Ruler import Ruler
except:
    from .Ruler import Ruler

try:
    from src.conformity.Protection import Protection
except:
    from .Protection import Protection

try:
    from src.conformity.Image import Image
except:
    from .Image import Image

try:
    from src.conformity.helpers import (
        slope_ordinate,
        find_intersection,
        approximate_text_by_point,
        my_arg_max,
        angle_betw_2_vects,
    )
except:
    from .helpers import (
        slope_ordinate,
        find_intersection,
        approximate_text_by_point,
        my_arg_max,
        angle_betw_2_vects,
    )
import math
import cv2
import os


class Conformity:
    "Class representing the conformity of a construction site"

    def __init__(self, model_pipeline, imagePath: str, tolerance=2 / 100) -> None:
        """Constructor of class Conformity """
        self.__image = Image(imagePath)
        self.__ruler = Ruler(model_pipeline, imagePath)
        self.__protection = Protection(imagePath)
        self.__intersection = None
        self.__distance = None
        self.__conformity = None
        self.__max_digit_read = None
        self.__conformity_distance = 12.0
        self.__image_path = imagePath
        self.__ruler_axis_color = (0, 255, 0)
        self.__protection_axis_color = (0, 0, 255)
        self.__illustration_thickness = 3
        self.__tolerance = tolerance
        self.__orthogonality = None

    def get_intersection(self) -> (float, float):
        """
            @parameters: object itself
            @returns: point representing the intersection of ruler and protection of the object
        """
        if self.__intersection == None:
            ruler_start_point, ruler_end_point = self.__ruler.get_axis()
            # print("the axis is: ", self.__ruler.get_axis())
            # print("ruler point coordinates ", ruler_start_point, ruler_end_point)
            (
                protection_start_point,
                protection_end_point,
            ) = self.__protection.get_axis_from_edges()
            # print(
            # "protection point coordinates ",
            # protection_start_point,
            # protection_end_point,
            # )
            ruler_slope, ruler_ordinate = slope_ordinate(
                ruler_start_point, ruler_end_point
            )
            protection_slope, protection_ordinate = slope_ordinate(
                protection_start_point, protection_end_point
            )
            # print("ruler slope and ordinate ", ruler_slope, ruler_ordinate)
            # print("protection_slope_ordinate", protection_slope, protection_ordinate)
            self.__intersection = find_intersection(
                ruler_slope, ruler_ordinate, protection_slope, protection_ordinate
            )
        return self.__intersection

    def check_orthogonality(self) -> bool:
        """
        @parameters: object itself
        @returns: True if orthogonality checked between the ruler and the protection. False otherwise
        """
        ruler_start_point, ruler_end_point = self.__ruler.get_axis()
        (
            protection_start_point,
            protection_end_point,
        ) = self.__protection.get_axis_from_edges()
        ruler_vector_start_end = (
            ruler_end_point[0] - ruler_start_point[0],
            ruler_end_point[1] - ruler_start_point[1],
        )
        protection_vector_end_start = (
            protection_start_point[0] - protection_end_point[0],
            protection_start_point[1] - protection_end_point[1],
        )
        # ruler_vector_start_end=(0,1)
        # protection_vector_end_start=(-1,0)
        intersection_angle = angle_betw_2_vects(
            protection_vector_end_start, ruler_vector_start_end
        )
        absolute_error_on_angle = math.atan(math.sqrt(2 * self.__tolerance))
        value1 = math.pi / 2 - absolute_error_on_angle <= intersection_angle
        value2 = intersection_angle <= math.pi / 2 + absolute_error_on_angle
        return value1 and value2

    def get_max_digit_read(self) -> (float, float):
        """
            @parameters: object itself
            @returns: the maximum of the digits on the ruler under the protection axis. Alongside with its ordinate.
        """
        if self.__max_digit_read == None:
            ruler_digits = self.__ruler.get_digits()
            ruler_digits = [
                (element[0], approximate_text_by_point(element)[1])
                for element in ruler_digits
            ]
            # print("0: ", ruler_digits)
            intersection = self.get_intersection()
            ruler_digits = [
                element for element in ruler_digits if element[1] >= intersection[1]
            ]
            ruler_digits_ordinates = [element[1] for element in ruler_digits]
            ruler_digits_value = [int(element[0]) for element in ruler_digits]
            index, value = my_arg_max(ruler_digits_value)
            self.__max_digit_read = (value, ruler_digits_ordinates[index])
        return self.__max_digit_read

    def get_distance(self) -> float:
        """
            @parameters: object itself
            @returns: the distance between the protection and the chimney
        """
        if self.__distance == None:
            pixel_cm_scale = self.__ruler.get_pixel_centimeter_scale()
            intersection = self.get_intersection()
            max_digit_read = self.get_max_digit_read()
            pix_delta_from_inter_to_max_digit = max_digit_read[1] - intersection[1]
            cm_delta_from_inter_to_max_digit = (
                pixel_cm_scale[1]
                * pix_delta_from_inter_to_max_digit
                / pixel_cm_scale[0]
            )
            self.__distance = cm_delta_from_inter_to_max_digit + max_digit_read[0]
        return self.__distance

    def get_conformity(self) -> dict:
        """
            @parameters: object itself
            @returns: dict with type : error | valid, with either message or distance
        """
        if self.__conformity == None:
            if not self.__protection.check_protection():
                self.__conformity = {
                    "type": "error",
                    "message": "Svp, reprenez une photo. Nous n'arrivons pas à bien détecter la protection présente sur l'image.",
                }
            elif not self.__ruler.check_digits_readability():
                self.__conformity = {
                    "type": "error",
                    "message": "Svp, reprenez une photo. Nous n'arrivons pas à détecter de réglette ou lire les chiffres sur la réglette.",
                }
            elif not self.__ruler.check_inclinaison_conformity(15 / 100):
                self.__conformity = {
                    "type": "error",
                    "message": "Svp, reprenez une photo. Votre réglettre ne semble pas parallèle au sol ou à l'appareil photo.",
                }
            elif not self.check_orthogonality():
                self.__conformity = {
                    "type": "error",
                    "message": "Votre réglette n'est pas assez perpendiculaire avec la protection.",
                }
            else:
                self.__conformity = {"type": "valid", "distance": self.get_distance(), "intersection":self.get_intersection(), "ruleur_axis":self.__ruler.get_axis(), "protection_axis":self.__protection.get_axis_from_edges()}
        return self.__conformity

    def get_conformity_distance(self) -> float:
        """
            @parameters: object itself
            @returns: return the current conformity distance
        """
        return self.__conformity_distance

    def set_conformity_distance(self, new_distance: float) -> None:
        """
            @parameters: object itself, and the new conformity distance
            @returns: Nothing
        """
        self.__conformity_distance = new_distance

    def get_ruler(self) -> Ruler:
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

    def get_image_path(self) -> str:
        """
            @parameters: object itself
            @returns: a string corresponding to the path of the image
        """
        return self.__image_path

    def get_illustration(self, output_image_path):
        """
        @parameters: object itself, the output path output_image_path
        @returns: Nothing. An image in bytes is created and then saved at given location
        """
        ruler_axis = self.__ruler.get_axis()
        protection_axis = self.__protection.get_axis_from_edges()
        ruler_start_point, ruler_end_point = ruler_axis
        protection_start_point, protection_end_point = protection_axis
        image = self.__image.get_original()
        cv2.line(
            image,
            (int(ruler_start_point[0]), int(ruler_start_point[1])),
            (int(ruler_end_point[0]), int(ruler_end_point[1])),
            self.__ruler_axis_color,
            self.__illustration_thickness,
        )
        cv2.line(
            image,
            (int(protection_start_point[0]), int(protection_start_point[1])),
            (int(protection_end_point[0]), int(protection_end_point[1])),
            self.__protection_axis_color,
            self.__illustration_thickness,
        )
        cv2.imwrite(output_image_path, image)
        return

    def get_illustration_thickness(self) -> int:
        """
            @parameters: object itself
            @returns: current thickness used for axis illustration
        """
        return self.__illustration_thickness

    def set_illustration_thickness(self, thickness: int) -> None:
        """
            @parameters: object itself and a thickness value
            @returns: Nothing
        """
        self.__illustration_thickness = thickness

    def get_ruler_axis_color(self) -> (int, int, int):
        """
            @parameters: object itself
            @returns: a triplet corresponding to the BGR code of the color for ruler axis drawing
        """
        return self.__ruler_axis_color

    def set_ruler_axis_color(self, color: (int, int, int)) -> None:
        """
            @parameters: object itself and a triplet corresponding to a BGR code
            @returns: Nothing
        """
        self.__ruler_axis_color = color

    def get_protection_axis_color(self) -> (int, int, int):
        """
            @parameters: object itself
            @returns: a triplet corresponding to the BGR code of the color for protection axis drawing
        """
        return self.__protection_axis_color

    def set_protection_axis_color(self, color: (int, int, int)) -> None:
        """
            @parameters: object itself and a triplet corresponding to a BGR code
            @returns: Nothing
        """
        self.__protection_axis_color = color


# if __name__ == '__main__':
# print("okay, brother")
# my_conformity = Conformity("im1-rotate.png")
#    print("intersection",my_conformity.get_intersection())
#    print("max digit read",my_conformity.get_max_digit_read())
#    print("work sclae", my_conformity.get_ruler().get_pixel_centimeter_scale())
#    print("digits", my_conformity.get_ruler().get_digits())
#    print("conform?",my_conformity.check_orthogonality())
#    print("distance ", my_conformity.get_distance())
# print(my_conformity.get_conformity())

