from catboost import CatBoostRegressor
cat_features = [2]
model = CatBoostRegressor(iterations=3000)
model.load_model('model', format='cbm')