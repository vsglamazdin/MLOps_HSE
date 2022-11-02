from flask import Flask, jsonify, json
from flask_restx import Api, Resource, reqparse
from api_modules.model import *


app = Flask(__name__)
api = Api(app)


def get_response(answer: dict):
    """
    Формирует response
    :param answer: Ответ со статусом
    :return: Возвращает response
    """
    if answer['code'] in [200, 201]:
        json_ans = {
            'result': answer['result'],
            'success': True,
            'error': []
        }
    else:
        json_ans = {
            'result': [],
            'success': False,
            'error': answer['result']
        }
    response = jsonify(json_ans)
    response.status_code = answer['code']
    return response


@api.route(
    '/available_model_types',
    methods=['GET'],
    doc={'description': 'You can get available model types for training'}
)
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


@api.route(
    '/fitted_models',
    methods=['POST', 'GET', 'DELETE'],
    doc={'description': 'You can fit model, get or delete fitted models'}
)
class FittedModels(Resource):
    """
    Работа с обученными моделями:
        - обучение;
        - получение списка обученных;
        - удаление обученной.
    """
    @api.doc(
        params={
            'model_name': 'Type of model to fit',
            'params': 'Hyperparameters to fit the model',
            'X': 'Features',
            'y': 'Target'
        }
    )
    @api.expect(fit_parser)
    def post(self):
        req_args = fit_parser.parse_args()
        return get_response(
            fit(req_args.model_name, json.loads(req_args.params),
                req_args.X, req_args.y)
        )

    @staticmethod
    def get():
        return get_response(available_fitted_models())

    @api.doc(params={'model_name': 'Name of fitted model to delete'})
    @api.expect(delete_parser)
    def delete(self):
        req_args = delete_parser.parse_args()
        return get_response(delete(req_args.model_name))


predict_parser = reqparse.RequestParser()
predict_parser.add_argument('model_name', help='Name of fitted model')
predict_parser.add_argument('X', action='append', help='Features')


@api.route(
    '/prediction',
    methods=['POST'],
    doc={'description': 'You can make prediction with fitted model'}
)
class Prediction(Resource):
    """
    Получение предсказания
    """
    @api.doc(
        params={
            'model_name': 'Name of fitted model',
            'X': 'Features'
        }
    )
    @api.expect(predict_parser)
    def post(self):
        req_args = predict_parser.parse_args()
        return get_response(predict(req_args.model_name, req_args.X))
