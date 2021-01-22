import os
from .Pascal_writer import PascalVocWriter
from .utils import convertPoints2BndBox
from PIL import Image


class Save_Xml:
    def __init__(self):
        self.filePath = None
        self.imageData = None

    def savePascalVocFormat(
        self,
        filename,
        shapes,
        imagePath,
        imageData=None,
        lineColor=None,
        fillColor=None,
        databaseSrc=None,
    ):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        image = Image.open(imagePath)
        width, height = image.size
        imageShape = [height, width, 3]
        writer = PascalVocWriter(
            imgFolderName, imgFileName, imageShape, localImgPath=imagePath
        )

        for shape in shapes:
            points = shape["points"]
            label = shape["label"]
            # Add Chris
            difficult = int(shape["difficult"])
            bndbox = convertPoints2BndBox(points)
            writer.addBndBox(
                bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, difficult
            )

        writer.save(targetFile=filename)
        print("Saved")
        return
