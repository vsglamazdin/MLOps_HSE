import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import joblib

fitted_models_pathname = 'fitted_models'
if not os.path.exists(fitted_models_pathname):
    os.makedirs(fitted_models_pathname)


def eval_data(data: list) -> list:
    """
    Преобразует элементы списка из строк в нужное
    :param data: Список, элементы которого надо преобразовать
    :return: Список с преобразованными элементами
    """
    return [eval(elem) for elem in data]


def available_model_types() -> dict:
    """
    Возвращает список доступных для обучения моделей
    :return: Список доступных для обучения моделей
    """
    return {
        'result': ['Ridge', 'RandomForestRegressor', 'CatBoostRegressor'],
        'code': 200
    }


def fit(model_name: str, params: dict, X: list, y: list) -> dict:
    """
    Обучает модель
    :param model_name: Тип модели
    :param params: Словарь с параметрами для обучения
    :param X: Список признаков
    :param y: Список таргетов
    :return: Комментарий о проделанной работе
    """
    if model_name in available_model_types():
        try:
            model = eval(model_name)(**params)
            model.fit(np.array(eval_data(X)), np.array(eval_data(y)))
        except TypeError:
            return {
                'result': f'Model {model_name} got an unexpected params',
                'code': 400
            }
        except Exception as e:
            return {
                'result': e.__str__(),
                'code': 400
            }
        filename_params = ', '.join([
            f'{param}={params[param]}' for param in params.keys()]
        )
        filename = f'{model_name}({filename_params})'
        joblib.dump(model, f'./{fitted_models_pathname}/{filename}.pkl')
        return {
            'result': f'''Model successfully fitted and saved 
            with name {filename}''',
            'code': 201
        }
    else:
        return {
            'result': f'Model {model_name} is not available to fit',
            'code': 404
        }


def available_fitted_models() -> dict:
    """
    Возвращает список обученных моделей
    :return: Список обученных моделей
    """
    return {
        'result': [m[:-4] for m in os.listdir(f'./{fitted_models_pathname}/')],
        'code': 200
    }


def delete(model_name: str) -> dict:
    """
    Удаляет обученную модель
    :param model_name: Имя обученной модели
    :return: Комментарий о проделанной работе
    """
    filename = f'./{fitted_models_pathname}/{model_name}.pkl'
    if os.path.exists(filename):
        os.remove(filename)
        return {
            'result': f'Model {model_name} successfully deleted',
            'code': 200
        }
    else:
        return {
            'result': f'Model {model_name} is not in fitted',
            'code': 404
        }


def predict(model_name: str, X: list) -> dict:
    """
    Предсказание моделью
    :param model_name: Имя обученной модели
    :param X: Список признаков
    :return: Предикт или комментарий
    """
    try:
        model = joblib.load(f'./{fitted_models_pathname}/{model_name}.pkl')
        prediction = model.predict(np.array(eval_data(X))).tolist()
    except FileNotFoundError:
        return {
            'result': f'Model {model_name} is not fitted',
            'code': 404
        }
    except Exception as e:
        return {
            'result': e.__str__(),
            'code': 400
        }
    return {
        'result': prediction,
        'code': 200
    }
