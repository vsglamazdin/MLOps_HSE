from flask import Flask, jsonify, json
from flask_restx import Api, Resource, reqparse
from api_modules.model import *


app = Flask(__name__)
api = Api(app)


def get_response(answer: list):
    """
    Возвращает конструкцию
    :param answer: Ответ
    :return: Возвращает конструкцию
    """
    json_ans = {
        'result': answer,
        'error': [],
        'success': True
    }
    response = jsonify(json_ans)
    # response.status_code = 200
    return response


@api.route('/available_model_types')
class AvailableModelTypes(Resource):
    """
    Доступные для обучения модели
    """
    @staticmethod
    def get():
        return get_response(available_model_types())


fit_parser = reqparse.RequestParser()
fit_parser.add_argument('model_name', help='Type of model to fit')
fit_parser.add_argument('params', help='Params to fit the model')
fit_parser.add_argument('X', action='append', help='Features')
fit_parser.add_argument('y', action='append', help='Target')

delete_parser = reqparse.RequestParser()
delete_parser.add_argument('model_name', help='Name of fitted model to delete')


@api.route('/fitted_models')
class FittedModels(Resource):
    """
    Работа с обученными моделями:
        - обучение;
        - получение списка обученных;
        - удаление обученной.
    """
    @api.expect(fit_parser)
    def post(self):
        req_args = fit_parser.parse_args()
        print(req_args.params)
        params = json.loads(req_args.params)
        return get_response(
            fit(req_args.model_name, params,
                req_args.X, req_args.y)
        )

    @staticmethod
    def get():
        return get_response(available_fitted_models())

    @api.expect(delete_parser)
    def delete(self):
        req_args = delete_parser.parse_args()
        return get_response(delete(req_args.model_name))


predict_parser = reqparse.RequestParser()
predict_parser.add_argument('model_name', help='Name of fitted model')
predict_parser.add_argument('X', action='append', help='Features')


@api.route('/prediction')
class Prediction(Resource):
    """
    Получение предсказания
    """
    @api.expect(predict_parser)
    def post(self):
        req_args = predict_parser.parse_args()
        return get_response(predict(req_args.model_name, req_args.X))
