import logging
from flask import request
from flask_restplus import Resource
from app.api.restplus import api


ns = api.namespace('eaf/detection', description='detection of eaf')

@ns.route('/')
class EafPrediction(Resource):

    def get(self):
        """Returns list of blog categories."""

    @api.response(201, 'Category successfully created.')
    def post(self):
        """Creates a new blog category."""
        create_category(request.json)
        return None, 201