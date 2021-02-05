import numpy as np
import math
from PIL import Image
import io
from base64 import encodebytes

def my_arg_max(my_list):
  """
    @parameters: a list my_list
    @returns: the index of the maximum element in the list and that element itself. If many, then the first tuple is returned.
  """
  index = 0
  current = my_list[0]
  for i in range(len(my_list)):
    if my_list[i] > current :
      index = i
      current = my_list[i]
  return index, current

def find_intersection(slope1,ordinate1,slope2,ordinate2):
  """
    @parameters: two lines parameters slope1, ordinate1 for the first and slope2, ordinate2 for the second
    @returns: return the ĉoordinates of the point at the intersection of the two lines.
  """
  x = (ordinate2-ordinate1)/(slope1-slope2)
  y = (slope1*ordinate2-ordinate1*slope2)/(slope1-slope2)
  return (x,y)

def slope_ordinate(start_point:tuple, end_point:tuple)->tuple:
  """
    @parameters: two points start_point and end_point
    @returns: the couple slope,ordinate of the line formed by the two points
  """
  slope = (start_point[1]-end_point[1])/(start_point[0]-end_point[0])
  ordinate = start_point[1] - slope * start_point[0]
  return slope,ordinate

def extend_line(slope:float, ordinate:float,m:int,n:int)->(tuple,tuple):
  """
    @parameters: parameters slope and ordinate defining a line. Two ordinates m and n
    @returns: a couple of points located on the line defined by slope and ordinate and located each at ordinate m and n
  """
  m_x = (m-ordinate)/slope
  n_x = (n-ordinate)/slope
  return (m_x,m),(n_x,n)

def approximate_text_by_point(detected_text)->tuple:
    """
        @parameters: a detected text by keras_ocr
        @returns: a dot approximating the position of that text on the image
    """
    text_box = detected_text[1]
    point = tuple(sum(text_box)/4)
    return point

def double_of_box_width(detected_text) -> float:
  """
    @parameters: a detected text by keras ocr
    @returns: the double of the width of the rectangular box locating the detected text
  """
  text_box = detected_text[1]
  width = abs(text_box[0][0]-text_box[1][0])
  return 2*width

def distance_to_line(slope,ordinate,point) -> float:
  """
    @parameters: parameters of a line: slope and ordinate. A point point
    @returns: the distance separating point from line slope,ordinate
  """
  distance = abs(slope*point[0]-point[1]+ordinate)/np.sqrt(slope*slope+1)
  return distance

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode="r")  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format="PNG")  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode("ascii")  # encode as base64
    return encoded_img

def angle_betw_2_vects(vect1:tuple, vect2:tuple)->float:
  """
    @parameters: two tuples of size 2 each. They represents plane vectors
    @returns: a float representing the inner angle of the two vectors. in radians
  """
  vect1_normalized = (vect1[0]/math.sqrt(vect1[0]*vect1[0]+vect1[1]*vect1[1]),vect1[1]/math.sqrt(vect1[0]*vect1[0]+vect1[1]*vect1[1]))
  vect2_normalized = (vect2[0]/math.sqrt(vect2[0]*vect2[0]+vect2[1]*vect2[1]),vect2[1]/math.sqrt(vect2[0]*vect2[0]+vect2[1]*vect2[1]))
  print("normalized vector",vect1_normalized)
  print("normalized vector",vect2_normalized)
  scalar_product = vect1_normalized[0]*vect2_normalized[0]+vect1_normalized[1]*vect2_normalized[1]
  return math.acos(scalar_product)


    