import pandas as pd
import numpy as np
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from flask import request, jsonify

# Функуция перевода уникальных названий
def UniqMarka(xx):
    unique_numbers = list(set(xx[:, 4]))

    uniq = {unique_numbers[0]: 1}
    tmp = 1
    for x in unique_numbers[1:]:
        tmp = tmp + 1
        uniq[x] = tmp

    for x in range(0, len(xx)):
        n = xx[x, 4]
        xx[x, 4] = uniq[n]

    return xx

# Основная функция
def prediction(dataFile, age=50, gender=1, exp=10, region=48, marka=2, year=2006, engine=140, kbm=0.85):
    try:
        if request.method == 'POST':
            file = request.files['dataFile']
            dataset = pd.read_csv(file, ',')
            xx = dataset.iloc[:, :].values
            # Переводим слова в цифры
            xx = UniqMarka(xx)
            # Меняем тип столбца
            xx[:, 0] = xx[:, 0].astype('float').astype('int32')
            # Разделяем зависимые и независимые переменные
            X = xx[:, :-1]
            y = xx[:, 8]

            # Разделяем на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

            # Стандартизации всех признаков, чтобы они имели примерно одинаковый масштаб.
            scaler = StandardScaler()
            scaler.fit(X_train)

            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            y_train = y_train.astype('float')
            y_test = y_test.astype('float')

            # Поиск оптимального числа соседних точек К
            knn2 = KNeighborsClassifier()
            param_grid = {'n_neighbors': np.arange(1, 25)}
            knn_gscv = GridSearchCV(knn2, param_grid, cv=7)
            knn_gscv.fit(X_train, y_train)
            # print(knn_gscv.best_params_)
            # print(knn_gscv.best_score_)

            classifier = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
            # Обучение модели
            classifier.fit(X_train, y_train)
            # Предсказание на тестовых данных
            y_pred = classifier.predict(X_test)
            # Точность модели
            print('acc: ', accuracy_score(y_pred, y_test))
            # Предсказание с полученными из функции данными
            y_pred = classifier.predict([[ age, gender, exp, region, marka, year, engine, kbm ]])
            # Ответ
            return jsonify({
                'code': 200,
                'prediction':y_pred[0],
                'score':knn_gscv.best_score_,
                'message': 'Успешно'
            }), 200
    except Exception:
        return jsonify({
            'code': 400,
            'message': 'На сервисе произошла ошибка'
        }), 400



