import xml.dom.minidom
import cv2
import os


def create_visualisation(
    image_path="./Images/", annotation_path="./Annotations/",
):
    colors = {
        "cheminee": (246, 151, 1),
        "no_eaf": (61, 38, 215),
        "eaf": (145, 184, 101),
    }
    files_name = os.listdir(image_path)
    for filename_ in files_name:
        if filename_ == ".DS_Store":
            continue
        filename, extension = os.path.splitext(filename_)
        img_path = image_path + filename + ".jpg"
        xml_path = annotation_path + filename + ".xml"
        print(img_path)
        img = cv2.imread(img_path)
        if img is None:
            pass
        dom = xml.dom.minidom.parse(xml_path)
        root = dom.documentElement
        objects = dom.getElementsByTagName("object")
        print(objects)
        for obj in objects:
            bndbox = obj.getElementsByTagName("bndbox")[0]
            xmin = bndbox.getElementsByTagName("xmin")[0]
            ymin = bndbox.getElementsByTagName("ymin")[0]
            xmax = bndbox.getElementsByTagName("xmax")[0]
            ymax = bndbox.getElementsByTagName("ymax")[0]
            xmin_data = xmin.childNodes[0].data
            ymin_data = ymin.childNodes[0].data
            xmax_data = xmax.childNodes[0].data
            ymax_data = ymax.childNodes[0].data
            label = obj.getElementsByTagName("name")[0].childNodes[0].data
            thickness = 2
            fontScale = 1
            color = colors[label]
            cv2.putText(
                img,
                label,
                (int(xmax_data), int(ymin_data)),
                cv2.FONT_HERSHEY_SIMPLEX,
                int(fontScale),
                color,
                thickness,
            )
            cv2.rectangle(
                img,
                (int(xmin_data), int(ymin_data)),
                (int(xmax_data), int(ymax_data)),
                color,
                5,
            )
        flag = 0
        flag = cv2.imwrite("./Visualisations/{}.jpg".format(filename), img)
        if flag:
            print(filename, "done")
    print("all done ====================================")
