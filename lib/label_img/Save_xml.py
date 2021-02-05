import os
from Pascal_writer import PascalVocWriter
from label_utils import convertPoints2BndBox
import cv2


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
        image = cv2.imread(imagePath)
        imageShape = image.shape
        writer = PascalVocWriter(
            imgFolderName, imgFileName, imageShape, localImgPath=imagePath
        )

        for shape in shapes:
            points = shape["points"]
            label = shape["label"]
            difficult = int(shape["difficult"])
            bndbox = convertPoints2BndBox(points)
            writer.addBndBox(
                bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, difficult
            )

        writer.save(targetFile=filename)
        print("Saved")
        return
