import cv2


def draw_label(path, start, end, color=(255, 0, 0)):
    image = cv2.imread(path)
    window_name = "Image"
    color = (255, 0, 0)
    thickness = 2
    image = cv2.rectangle(image, start, end, color, thickness)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_points(path, points):
    image = cv2.imread(path)
    window_name = "Image"
    color = (255, 0, 0)
    thickness = 2
    for point in points:
        image = cv2.circle(image, point, 0, color, thickness)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_annotates_image(
    shapes, path="/Users/matthieu/Documents/Project3/image_generator/rectangle.jpg",
):
    chimney = shapes[0]["points"]
    protection = shapes[1]["points"]
    image = cv2.imread(path)
    window_name = "Image"
    thickness = 2
    start = (chimney[0][0], 270 - chimney[0][1])
    end = (chimney[2][0], 270 - chimney[2][1])
    image = cv2.rectangle(image, start, end, (255, 0, 0), thickness)
    start = (protection[0][0], 270 - protection[0][1])
    end = (protection[2][0], 270 - protection[2][1])
    image = cv2.rectangle(image, start, end, (0, 255, 0), thickness)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
