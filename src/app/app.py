from flask import Flask, request, Response, jsonify, flash, redirect
from flask_cors import cross_origin, CORS
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

app = Flask(__name__)  # Create a Flask WSGI application
cors = CORS(app, resources={r"/predict": {"origins": "http://localhost:8888"}})
MODEL_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/models/"
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/uploads/"
OUTPUT_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/outputs/"

ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def configure_app(flask_app):
    flask_app.config["CORS_HEADERS"] = "Content-Type"
    flask_app.config["MODEL_FOLDER"] = MODEL_FOLDER
    flask_app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    flask_app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
    flask_app.config["SERVER_NAME"] = settings.FLASK_SERVER_NAME
    flask_app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


@app.route("/predict_presence", methods=["POST"])
def presence():
    detector = ModelBasedDetector.from_finetuned(
        "models/ssd_512_best.params", thresh=0.2
    )
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    uploaded_file = request.files["file"]
    filename = uploaded_file.filename
    if not filename:
        return jsonify({"msg": "Votre fichier n'a pas de nom."}), 400
    if allowed_file(filename):
        uploaded_file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        score, prediction = detector.predict(
            os.path.join(app.config["UPLOAD_FOLDER"], filename),
            app.config["OUTPUT_FOLDER"],
        )
        output_image_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    else:
        return (
            jsonify(
                {
                    "msg": f"Les extensions autorisées sont {', '.join(ALLOWED_EXTENSIONS)}."
                }
            ),
            400,
        )
    encoded_img = get_response_image(output_image_path)
    return Response(
        str({"score": score, "image": encoded_img, "prediction": str(prediction)})
    )


@app.route("/predict_conformity", methods=["POST"])
def conformity():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    uploaded_file = request.files["file"]
    filename = uploaded_file.filename
    if not filename:
        return jsonify({"msg": "Votre fichier n'a pas de nom."}), 400
    if allowed_file(filename):
        uploaded_file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        conformity = Conformity(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        output_image_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
        conformity.get_illustration(output_image_path)
        encoded_img = get_response_image(output_image_path)
        prediction = conformity.get_distance()
        return Response(str({"image": encoded_img, "prediction": prediction}))
    else:
        return (
            jsonify(
                {
                    "msg": f"Les extensions autorisées sont {', '.join(ALLOWED_EXTENSIONS)}."
                }
            ),
            400,
        )


def get_response_image(image_path):
    pil_img = Image.open(image_path, mode="r")  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format="PNG")  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode("ascii")  # encode as base64
    return encoded_img


if __name__ == "__main__":
    configure_app(app)
    app.run(debug=True)
