from catboost import CatBoostRegressor
import numpy as np
from scipy.special import softmax
import datetime

cat_features = [2]
model = CatBoostRegressor(iterations=3000)
model.load_model('model', format='cbm')

days_in_months = {'01': 31, '02': 30, '03': 28, '04': 30, '05': 31, '06': 30, '07': 31, '08': 30, '09': 31, '10': 30,
                  '11': 31, '12': 30}


def predict(money: int, start: datetime.datetime, end: datetime.datetime, product, products):

    if product != 'all':
        product = int(product)

    profits = []
    year1, month1, day1 = start.year, start.month, start.day
    year2, month2, day2 = end.year, end.month, end.day

    for month in range(1, month2 - month1 + 1):
        month = str(month) if month > 9 else '0' + str(month)
        for day in range(1, int(days_in_months[month]) + 1):
            if product == 'all':
                preds = max([model.predict([int(month), day, product]) for product in products])
            else:
                preds = [model.predict([int(month), day, product])]

            profits.append(preds)
    profits = np.array(profits)
    profits /= profits.max()
    profits = softmax(profits)

    return profits / profits.max()
