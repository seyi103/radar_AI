import numpy as np
import pandas as pd
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.metrics import mean_squared_error
import optuna
import torch
from tqdm.auto import tqdm
from pytorch_tabnet.tab_model import TabNetRegressor

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
import catboost as cb
from catboost import CatBoostRegressor

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler

SEED = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정

#data read
train_df = pd.read_csv('D:/dacon_data/train.csv')
test_x = pd.read_csv('D:/dacon_data/test.csv').drop(columns=['ID'])
submit = pd.read_csv('D:/dacon_data/sample_submission.csv')

train = train_df.copy()
test = test_x.copy()

#데이터 처리
features = ['X_01', 'X_02', 'X_03', 'X_05', 'X_06', 'X_07', 'X_08',
       'X_09', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17',
       'X_18', 'X_19', 'X_20', 'X_21', 'X_22', 'X_24', 'X_25', 'X_26',
       'X_27', 'X_28', 'X_29', 'X_30', 'X_31', 'X_32', 'X_33', 'X_34', 'X_35',
       'X_36', 'X_37', 'X_38', 'X_39', 'X_40', 'X_41', 'X_42', 'X_43', 'X_44',
       'X_45', 'X_46', 'X_49', 'X_50', 'X_51', 'X_52', 'X_53',
       'X_54', 'X_55', 'X_56']

target_list = ['Y_01', 'Y_02', 'Y_03', 'Y_04', 'Y_05', 'Y_06',
       'Y_07', 'Y_08', 'Y_09', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14']

for i in tqdm(range(len(target_list))):
    globals()['train_{}'.format(target_list[i])] = train[['X_01', 'X_02', 'X_03', 'X_04', 'X_05', 'X_06', 'X_07', 'X_08',
       'X_09', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17',
       'X_18', 'X_19', 'X_20', 'X_21', 'X_22', 'X_23', 'X_24', 'X_25', 'X_26',
       'X_27', 'X_28', 'X_29', 'X_30', 'X_31', 'X_32', 'X_33', 'X_34', 'X_35',
       'X_36', 'X_37', 'X_38', 'X_39', 'X_40', 'X_41', 'X_42', 'X_43', 'X_44',
       'X_45', 'X_46', 'X_47', 'X_48', 'X_49', 'X_50', 'X_51', 'X_52', 'X_53',
       'X_54', 'X_55', 'X_56', f'{target_list[i]}']]
    locals()['train_{}'.format(target_list[i])].drop(columns=['X_48', 'X_47', 'X_23', 'X_04'], inplace=True)
    feature = locals()['train_{}'.format(target_list[i])][features]
    target_t = locals()['train_{}'.format(target_list[i])][[target_list[i]]]

train_x = train.filter(regex='X') # Input : X Featrue
train_y = train.filter(regex='Y') # Output : Y Feature
test_x = test.filter(regex = 'X')
X_test = test[features]

train_x = train_x.drop(['X_48', 'X_47', 'X_23', 'X_04'], axis = 1, inplace=False)
test_x = test_x.drop(['X_48', 'X_47', 'X_23', 'X_04'], axis = 1, inplace=False)
x_train_all = train.filter(regex='X') # Input : X Featrue
Y_train_all = train.filter(regex='Y') # Output : Y Feature

#스코어 확인
def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(0,14): # ignore 'ID'
        rmse = mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:15])
    return score

#GB
# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(learning_rate=0.03, max_depth= 8, n_estimators=200, random_state= 10)

#XGB
xgboost = xgb.XGBRegressor(gamma = 1, learning_rate = 0.05, n_estimators=200, max_depth=8, min_child_weight=7, device = 'gpu')

#LGBM
n_splits = 10
predictions = []
lgnrmses = []
kfold = KFold(n_splits=n_splits, random_state=SEED, shuffle=True)
for i, (train_idx, val_idx) in enumerate(kfold.split(train_x)):
    preds = []
    y_vals = []
    predictions_ = []
    for j in range(1, 15):
        if j < 10:
            train_y_ = train_y[f'Y_0{j}']
        else:
            train_y_ = train_y[f'Y_{j}']
        X_train, y_train = train_x.iloc[train_idx], train_y_.iloc[train_idx]
        X_val, y_val = train_x.iloc[val_idx], train_y_.iloc[val_idx]

        print(f'fit {train_y.columns[j - 1]}')
        model = lgb.LGBMRegressor(num_leaves=50, learning_rate=0.03, n_estimators=200, max_depth= 8,
                                                       min_child_weight= 1, min_split_gain= 0.3, n_jobs=-1, random_state = SEED)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        print(f'predict {train_y.columns[j - 1]}')
        pred = model.predict(X_val)
        prediction = model.predict(test_x)
        # print(prediction)
        predictions_.append(prediction)
        # print(predictions_)
        preds.append(pred)
        y_vals.append(y_val)
    predictions.append(predictions_)
    print(predictions)
    lgnrmse = lg_nrmse(np.array(y_vals).T, np.array(preds).T)
    lgnrmses.append(lgnrmse)
    print(f'Fold {i} / lg_nrmse : {lgnrmse}')


np.mean(lgnrmse)
lgbm_prediction = np.mean(np.array(predictions), axis=0).T

lightgbm_model = joblib.load('./lgbm.pkl')
lgbm_all_prediction = lightgbm_model.predict(test)

#svr
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

#lasso
alpha = 0.0001
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8,
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05,
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35,
                          alpha * 1.4],
                max_iter = 50000, cv = 10, n_jobs=-1)
lasso = make_pipeline(RobustScaler(),lasso)

#ridge
alpha = 0.006
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], cv = 10)
ridge = make_pipeline(RobustScaler(),ridge)

'''# # iter = 100, 18 seconds, 3000+GPU = 2 mins

cb_dtrain = cb.Pool(data= X_train, label= y_train)

clf = CatBoostRegressor(
    iterations=3000,
    task_type="GPU",
    bagging_temperature = 0.2
)'''

#rf
rf_run = RandomForestRegressor(random_state=0, max_depth=8, min_samples_leaf=18, min_samples_split=8,n_estimators=200)


#10번 fold한 catboost 모델
n_splits = 10
predictions = []
lgnrmses = []
kfold = KFold(n_splits=n_splits, random_state=SEED, shuffle=True)
for i, (train_idx, val_idx) in enumerate(kfold.split(train_x)):
    preds = []
    y_vals = []
    predictions_ = []
    for j in range(1, 15):
        if j < 10:
            train_y_ = train_y[f'Y_0{j}']
        else:
            train_y_ = train_y[f'Y_{j}']
        X_train, y_train = train_x.iloc[train_idx], train_y_.iloc[train_idx]
        X_val, y_val = train_x.iloc[val_idx], train_y_.iloc[val_idx]

        print(f'fit {train_y.columns[j - 1]}')
        model = CatBoostRegressor(random_state=SEED)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        joblib.dump(model, './cat.pkl')
        print(f'predict {train_y.columns[j - 1]}')
        pred = model.predict(X_val)
        prediction = model.predict(test_x)
        # print(prediction)
        predictions_.append(prediction)
        # print(predictions_)
        preds.append(pred)
        y_vals.append(y_val)
    predictions.append(predictions_)
    print(predictions)
    lgnrmse = lg_nrmse(np.array(y_vals).T, np.array(preds).T)
    lgnrmses.append(lgnrmse)
    print(f'Fold {i} / lg_nrmse : {lgnrmse}')

np.mean(lgnrmse)
cat_prediction = np.mean(np.array(predictions), axis=0).T

cat_model = joblib.load('./cat.pkl')
cat_all_prediction = cat_model.predict(test)

'''#전체 데이터 학습
gbr_full_data = MultiOutputRegressor(gbr).fit(train_x, train_y)
xgboost_full_data = MultiOutputRegressor(xgboost).fit(train_x, train_y)
#lightgbm_full_data = LGBM.fit(train, train_y)
svr_full_data = MultiOutputRegressor(svr).fit(train_x, train_y)
lasso_full_data = MultiOutputRegressor(lasso).fit(train_x, train_y)
ridge_full_data = MultiOutputRegressor(ridge).fit(train_x, train_y)
#cat_full_data = MultiOutputRegressor(clf).fit(train, train_y)
rf_full_data = MultiOutputRegressor(rf_run).fit(train_x, train_y)
print("done.")

#모델 저장
joblib.dump(gbr_full_data, './gbr.pkl')
joblib.dump(xgboost_full_data, './xgb.pkl')
#joblib.dump(lightgbm_full_data, './lgbm.pkl')
joblib.dump(svr_full_data, './svr.pkl')
joblib.dump(lasso_full_data, './lasso.pkl')
joblib.dump(ridge_full_data, './ridge.pkl')
#joblib.dump(cat_full_data, './cat.pkl')
joblib.dump(rf_full_data, './rf.pkl')
print("모델 저장 종료")'''


#모델 불러오기
svr_model = joblib.load('./svr.pkl')
lasso_model = joblib.load('./lasso.pkl')
ridge_model = joblib.load('./ridge.pkl')
gbr_model = joblib.load('./gbr.pkl')
xgboost_model = joblib.load('./xgb.pkl')
#cat_model = joblib.load('./cat.pkl')
#lightgbm_model = joblib.load('./lgbm.pkl')
rf_model = joblib.load('./rf.pkl')

# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(X):
    return ((0.02 * svr_model.predict(X)) + \
            (0.02 * lasso_model.predict(X)) + \
            (0.02 * ridge_model.predict(X)) + \
            (0.04 * gbr_model.predict(X)) + \
            (0.2 * xgboost_model.predict(X)) + \
            (0.4 * cat_prediction) + \
            (0.3 * lgbm_prediction))


start_time = time.time()
print('시작')
pred = blended_predictions(test_x)
print('종료')
end_time = time.time()
print('걸린 시간: ', (end_time - start_time))

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = pred[:,idx-1]
print('Done.')

submit.to_csv('./submission.csv', index=False)

