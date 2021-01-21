from flask import Flask,request, Response, jsonify
import io
from base64 import encodebytes
from PIL import Image
import settings
import os 
import matplotlib
matplotlib.use('agg')
import sys
sys.path.append('../Detector')
from Detector import ModelBasedDetector
sys.path.append('/utils')
from get_results import process_output_img


app = Flask(__name__)                  #  Create a Flask WSGI application
MODEL_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/models/'
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
OUTPUT_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/outputs/'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def configure_app(flask_app):
    flask_app.config['MODEL_FOLDER'] = MODEL_FOLDER
    flask_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    flask_app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
    flask_app.config['SERVER_NAME'] = settings.FLASK_SERVER_NAME
    flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
    flask_app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE
    flask_app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER
    flask_app.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP


@app.route('/predict', methods=['POST'])
def index():
    detector = ModelBasedDetector.from_finetuned('models/ssd_512_best.params', thresh=0.2)
    uploaded_file = request.files.get('file')
    filename = uploaded_file.filename
    if not filename:
        return jsonify({'msg': "Votre fichier n'a pas de nom."}), 400
    if allowed_file(filename):
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        score, prediction = detector.predict(os.path.join(app.config['UPLOAD_FOLDER'], filename), app.config['OUTPUT_FOLDER'])
        output_image_path = process_output_img(os.path.join(app.config['OUTPUT_FOLDER'], filename))
    else:
        return jsonify({'msg': f"Les extensions autorisées sont {', '.join(ALLOWED_EXTENSIONS)}."}), 400
    encoded_img = get_response_image(output_image_path)
    return Response(str({'score': score, 'image':encoded_img, 'prediction':prediction}))


def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

if __name__ == '__main__':
    configure_app(app)
    app.run(debug=True)