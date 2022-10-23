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
    Преобразует элемента списка из строк в нужное
    :param data: Список, элементы которого надо преобразовать
    :return: Список с преобразованными элементами
    """
    return [eval(elem) for elem in data]

def available_model_types() -> list:
    """
    Возвращает список доступных для обучения моделей
    :return: Список доступных для обучения моделей
    """
    return ['Ridge', 'RandomForestRegressor', 'CatBoostRegressor']

def fit(model_name: str, params: dict, X: list, y: list) -> list:
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
        except TypeError:
            return [f'Model {model_name} got an unexpected params']
        model.fit(np.array(eval_data(X)), np.array(eval_data(y)))
        filename_params = ', '.join([
            f'{param}={params[param]}' for param in params.keys()]
        )
        filename = f'{model_name}({filename_params})'
        joblib.dump(model, f'./{fitted_models_pathname}/{filename}.pkl')
        return [f'Model successfully fitted and saved with name {filename}']
    else:
        return [f'Model {model_name} is not available to fit']


def available_fitted_models() -> list:
    """
    Возвращает список обученных моделей
    :return: Список обученных моделей
    """
    return [m[:-4] for m in os.listdir(f'./{fitted_models_pathname}/')]

def delete(model_name: str) -> list:
    """
    Удаляет обученную модель
    :param model_name: Имя обученной модели
    :return: Комментарий о проделанной работе
    """
    filename = f'./{fitted_models_pathname}/{model_name}.pkl'
    if os.path.exists(filename):
        os.remove(filename)
        return [f'Model {model_name} successfully deleted']
    else:
        return [f'Model {model_name} is not in fitted']

def predict(model_name: str, X: list) -> list:
    """
    Предсказание моделью
    :param model_name: Имя обученной модели
    :param X: Список признаков
    :return: Предикт или комментарий
    """
    try:
        model = joblib.load(f'./{fitted_models_pathname}/{model_name}.pkl')
    except FileNotFoundError:
        return [f'Model {model_name} is not fitted']
    return model.predict(np.array(eval_data(X))).tolist()
