# Домашнее задание 1 - API

## Как запустить API:
1. Поставить зависимости с помощью `poetry`;
2. Запустить в командной строке `python main.py`

## Примеры команд для работы с API:
### 1. Список доступных для обучения классов моделей:
    curl -X GET http://127.0.0.1:5000/available_model_types

### 2. Список обученных моделей, готовых к предсказанию:
    curl -X GET http://127.0.0.1:5000/fitted_models

### 3. Обучить модель:
    curl -X POST http://127.0.0.1:5000/fitted_models -H 'Content-type: application/json' -d '{"model_name": "Ridge", "params": "{\"alpha\": 10}", "X": [[23,34,2],[12,34,98],[1,5,-2]], "y": [34,98,72]}'

    curl -X POST http://127.0.0.1:5000/fitted_models -H 'Content-type: application/json' -d '{"model_name": "RandomForestRegressor", "params": "{\"n_estimators\": 30, \"max_features\": \"log2\"}", "X": [[23,34,2],[12,34,98],[1,5,-2]], "y": [34,98,72]}'

    curl -X POST http://127.0.0.1:5000/fitted_models -H 'Content-type: application/json' -d '{"model_name": "CatBoostRegressor", "params": "{\"iterations\": 50, \"bootstrap_type\": \"Bayesian\"}", "X": [[23,34,2],[12,34,98],[1,5,-2]], "y": [34,98,72]}'

### 4. Удалить обученную модель:
    curl -X DELETE -H "Content-type: application/json" http://127.0.0.1:5000/fitted_models -d '{"model_name": "Ridge(alpha=10)"}'

### 5. Получить предсказание:
    curl -X POST http://127.0.0.1:5000/prediction -H 'Content-type: application/json' -d '{"model_name": "Ridge(alpha=10)", "X": [[23,34,2],[12,34,98],[1,5,-2]]}'

    curl -X POST http://127.0.0.1:5000/prediction -H 'Content-type: application/json' -d '{"model_name": "RandomForestRegressor(n_estimators=30, max_features=log2)", "X": [[23,34,2],[12,34,98],[1,5,-2]]}'

    curl -X POST http://127.0.0.1:5000/prediction -H 'Content-type: application/json' -d '{"model_name": "CatBoostRegressor(iterations=50, bootstrap_type=Bayesian)", "X": [[23,34,2],[12,34,98],[1,5,-2]]}'