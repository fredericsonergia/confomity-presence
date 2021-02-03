from Save_xml import Save_Xml


def create_annotation_file(shapes, filename, image_path):
    path = "Annotations/" + filename + ".xml"
    saver = Save_Xml()
    saver.savePascalVocFormat(path, shapes, image_path)


if __name__ == "__main__":
    shapes = (
        {
            "points": [(178, 105), (178, 270), (282, 270), (282, 105)],
            "label": "eaf",
            "difficult": 0,
        },
        {
            "points": [(104, 5), (104, 137), (318, 137), (318, 5)],
            "label": "cheminee",
            "difficult": 0,
        },
    )
    filename = "test1"
    image_path = "./Images/rectangle.jpg"
    create_annotation_file(shapes, filename, image_path)
