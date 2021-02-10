from fastapi import FastAPI, File, UploadFile, Form, Depends
from pydantic import BaseModel, BaseSettings
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

import shutil
import io
from base64 import encodebytes
from PIL import Image
import settings
import os
import matplotlib

matplotlib.use("agg")
import sys

sys.path.append("../Detector")
sys.path.append("../conformity")

from Detector import ModelBasedDetector
from Conformity import Conformity

sys.path.append("/utils")

MODEL_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/models/"
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/uploads/"
OUTPUT_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/outputs/"


class Settings(BaseSettings):
    MODEL_FOLDER: str = MODEL_FOLDER
    UPLOAD_FOLDER: str = UPLOAD_FOLDER
    OUTPUT_FOLDER: str = OUTPUT_FOLDER
    SERVER_NAME: str = settings.FLASK_SERVER_NAME
    MAX_CONTENT_LENGTH: int = 8 * 1024 * 1024


settings = Settings()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_response_image(image_path):
    pil_img = Image.open(image_path, mode="r")  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format="PNG")  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode("ascii")  # encode as base64
    return encoded_img


"""
TYPES
"""


class requestForm:
    def __init__(self, file: UploadFile = Form(...)):
        self.file = file


class presenceResponse(BaseModel):
    score: float
    image: str
    box: list
    prediction: bool

    class Config:
        schema_extra = {
            "example": {
                "score": "score de confiance entre 0 et 1",
                "image": "image encodé en bytes",
                "box": "list qui définisse le rectangle de justification [xmin,ymin,xmax,ymax]",
                "prediction": "True si presence de protection, False sinon",
            }
        }


class conformityResponse(BaseModel):
    type: str
    image: Optional[str]
    distance: Optional[float]
    message: Optional[str]

    class Config:
        schema_extra = {
            "example": {
                "type": "error si le model n'arrive pas à traiter l'image, valid sinon",
                "image": "image encodé en bytes",
                "distance": "si type est valid : distance mesurée en cm, None sinon",
                "message": "si type est error : message à renvoyer à l'utilisateur pour expliquer l'erreur",
            }
        }


"""
ROUTES
"""


@app.post("/presence", response_model=presenceResponse)
def presence(request: requestForm = Depends()):
    detector = ModelBasedDetector.from_finetuned(
        "models/fake400_7style+real_best.params", thresh=0.32499
    )
    uploaded_file = request.file
    filename = uploaded_file.filename
    if allowed_file(filename):
        with open(os.path.join(settings.UPLOAD_FOLDER, filename), "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
        score, prediction, box_coord = detector.predict(
            os.path.join(settings.UPLOAD_FOLDER, filename), settings.OUTPUT_FOLDER,
        )
        output_image_path = os.path.join(settings.OUTPUT_FOLDER, filename)
    encoded_img = get_response_image(output_image_path)
    return {
        "score": score,
        "image": encoded_img,
        "box": box_coord,
        "prediction": prediction,
    }


@app.post("/conformity", response_model=conformityResponse)
def conformity(request: requestForm = Depends()):
    uploaded_file = request.file
    filename = uploaded_file.filename
    if allowed_file(filename):
        with open(os.path.join(settings.UPLOAD_FOLDER, filename), "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
        conformity = Conformity(os.path.join(settings.UPLOAD_FOLDER, filename))
        result = conformity.get_conformity()
        output_image_path = os.path.join(settings.OUTPUT_FOLDER, filename)
        if result["type"] == "valid":
            conformity.get_illustration(output_image_path)
            encoded_img = get_response_image(output_image_path)
            result["image"] = encoded_img
        return result
    else:
        return {
            "type": "error",
            "message": f"Les extensions autorisées sont {', '.join(ALLOWED_EXTENSIONS)}.",
        }
