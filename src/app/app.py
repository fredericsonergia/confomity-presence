from flask import Flask, Blueprint
from app.api.restplus import api
from app import settings

app = Flask(__name__)                  #  Create a Flask WSGI application


def configure_app(flask_app):
    flask_app.config['SERVER_NAME'] = settings.FLASK_SERVER_NAME
    flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
    flask_app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE
    flask_app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER
    flask_app.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP

def initialize_app(flask_app):
    configure_app(flask_app)

    blueprint = Blueprint('eaf', __name__, url_prefix='/eaf')
    api.init_app(blueprint)
    flask_app.register_blueprint(blueprint)

if __name__ == '__main__':
    app.run(debug=True)