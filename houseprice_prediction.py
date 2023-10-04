import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

#data load
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv', index_col='Id')
data = train
print(train.shape, test.shape, submission.shape)
data.head()

#집 가격에 따른 그래프
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(14,6)
sns.distplot(data['SalePrice'], fit=norm, ax=ax1)
sns.distplot(np.log(data['SalePrice']+1), fit=norm, ax=ax2)

#요인과 집 가격에 따른 상관관계 분석
corr = data.corr()
top_corr = data[corr.nlargest(40,'SalePrice')['SalePrice'].index].corr()
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(20,15)
sns.heatmap(top_corr, annot=True, ax=ax1)

sns.regplot(x = 'GrLivArea', y = 'SalePrice', data = data)

#해당 조건의 데이터는 오류를 불러올 수 있으므로, 제거
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
Ytrain = train['SalePrice']
train = train[list(test)]
all_data = pd.concat((train, test), axis=0)
print(all_data.shape)
#그래프의 균형을 잡기 위해서 +1
Ytrain = np.log(Ytrain+1)

cols = list(all_data)
for col in list(all_data):
    if (all_data[col].isnull().sum()) == 0:
        cols.remove(col)
    else:
        pass
print(len(cols))

#빈 데이터 여부 확인
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea','LotFrontage'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    
print(f"Total count of missing values in all_data : {all_data.isnull().sum().sum()}")

#층과 집 가격 간의 관계 확인
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(14, 10)
sns.regplot(x='TotalBsmtSF', y='SalePrice', ax=ax1, data=data, color='blue')
sns.regplot(x='1stFlrSF', y='SalePrice', ax=ax2, data=data, color='green')
sns.regplot(x='2ndFlrSF', y='SalePrice', ax=ax3, data=data, color='red')
sns.regplot(x=data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF'], y='SalePrice', ax=ax4, data=data, color='purple')

#그래프 확인 후, 묶을 건 묶기
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['No2ndFlr'] = (all_data['2ndFlrSF'] == 0)
all_data['NoBsmt'] = (all_data['TotalBsmtSF'] == 0)

#욕조 등 추가 시설과 집 가격 간의 관계 확인
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(14,10)
sns.barplot(x = data['BsmtFullBath'], y = data['SalePrice'], ax=ax1)
sns.barplot(x = data['FullBath'], y = data['SalePrice'], ax=ax2)
sns.barplot(x = data['BsmtHalfBath'], y = data['SalePrice'], ax=ax3)
sns.barplot(x = data['HalfBath'], y = data['SalePrice'], ax=ax4)

figure, (ax5) = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(14,6)
sns.barplot(x = data['BsmtFullBath'] + data['FullBath'] + (data['BsmtHalfBath']/2) + (data['HalfBath']/2), y = data['SalePrice'], ax=ax5)

#그래프 확인 후, 데이터 묶어주기
all_data['TotalBath']=all_data['BsmtFullBath'] + all_data['FullBath'] + (all_data['BsmtHalfBath']/2) + (all_data['HalfBath']/2)

#완공 시기와 집 가격 간의 가격 확인
figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(18,8)
sns.regplot(x = data['YearBuilt'], y = data['SalePrice'], ax=ax1, color = 'red')
sns.regplot(x = data['YearRemodAdd'], y = data['SalePrice'], ax=ax2, color = 'blue')
sns.regplot(x = (data['YearBuilt']+data['YearRemodAdd'])/2, y = data['SalePrice'], ax=ax3, color = 'green')

#그래프 확인 후, 데이터 묶어주기
all_data['YrBltAndRemod'] = all_data['YearBuilt']+all_data['YearRemodAdd']
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)

#지하실의 추가 시설에 따른 딕셔너리 추가
Basement = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtUnfSF', 'TotalBsmtSF']
Bsmt = all_data[Basement]

Bsmt = Bsmt.replace(to_replace = 'Po', value = 1)
Bsmt = Bsmt.replace(to_replace = 'Fa', value = 2)
Bsmt = Bsmt.replace(to_replace = 'TA', value = 3)
Bsmt = Bsmt.replace(to_replace = 'Gd', value = 4)
Bsmt = Bsmt.replace(to_replace = 'Ex', value = 5)
Bsmt = Bsmt.replace(to_replace = 'None', value = 0)

Bsmt = Bsmt.replace(to_replace = 'No', value = 1)
Bsmt = Bsmt.replace(to_replace = 'Mn', value = 2)
Bsmt = Bsmt.replace(to_replace = 'Av', value = 3)
Bsmt = Bsmt.replace(to_replace = 'Gd', value = 4)

Bsmt = Bsmt.replace(to_replace = 'Unf', value = 1)
Bsmt = Bsmt.replace(to_replace = 'LwQ', value = 2)
Bsmt = Bsmt.replace(to_replace = 'Rec', value = 3)
Bsmt = Bsmt.replace(to_replace = 'BLQ', value = 4)
Bsmt = Bsmt.replace(to_replace = 'ALQ', value = 5)
Bsmt = Bsmt.replace(to_replace = 'GLQ', value = 6)

Bsmt['BsmtScore'] = Bsmt['BsmtQual'] * Bsmt['BsmtCond'] * Bsmt['TotalBsmtSF']
all_data['BsmtScore'] = Bsmt['BsmtScore']

Bsmt['BsmtFin'] = (Bsmt['BsmtFinSF1'] * Bsmt['BsmtFinType1']) + (Bsmt['BsmtFinSF2'] * Bsmt['BsmtFinType2'])
all_data['BsmtFinScore'] = Bsmt['BsmtFin']
all_data['BsmtDNF'] = (all_data['BsmtFinScore'] == 0)

lot=['LotFrontage', 'LotArea','LotConfig','LotShape']
Lot=all_data[lot]

Lot['LotScore'] = np.log((Lot['LotFrontage'] * Lot['LotArea'])+1)

all_data['LotScore'] = Lot['LotScore']

#차고에 따른 딕셔너리 추가
garage = ['GarageArea','GarageCars','GarageCond','GarageFinish','GarageQual','GarageType','GarageYrBlt']
Garage = all_data[garage]
all_data['NoGarage'] = (all_data['GarageArea'] == 0)

Garage = Garage.replace(to_replace = 'Po', value = 1)
Garage = Garage.replace(to_replace = 'Fa', value = 2)
Garage = Garage.replace(to_replace = 'TA', value = 3)
Garage = Garage.replace(to_replace = 'Gd', value = 4)
Garage = Garage.replace(to_replace = 'Ex', value = 5)
Garage = Garage.replace(to_replace = 'None', value = 0)

Garage = Garage.replace(to_replace = 'Unf', value = 1)
Garage = Garage.replace(to_replace = 'RFn', value = 2)
Garage = Garage.replace(to_replace = 'Fin', value = 3)

Garage = Garage.replace(to_replace = 'CarPort', value = 1)
Garage = Garage.replace(to_replace = 'Basment', value = 4)
Garage = Garage.replace(to_replace = 'Detchd', value = 2)
Garage = Garage.replace(to_replace = '2Types', value = 3)
Garage = Garage.replace(to_replace = 'Basement', value = 5)
Garage = Garage.replace(to_replace = 'Attchd', value = 6)
Garage = Garage.replace(to_replace = 'BuiltIn', value = 7)
Garage['GarageScore'] = (Garage['GarageArea']) * (Garage['GarageCars']) * (Garage['GarageFinish']) * (Garage['GarageQual']) * (Garage['GarageType'])
all_data['GarageScore'] = Garage['GarageScore']

#다른 시설과는 관련이 없다고 생각하여 제거
all_data = all_data.drop(columns = ['Street','Utilities','Condition2','RoofMatl','Heating'])
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(14,6)
sns.regplot(x = 'PoolArea', y = 'SalePrice', ax = ax1, data = data)
sns.barplot(x = 'PoolQC', y = 'SalePrice', ax = ax2, data = data)

all_data = all_data.drop(columns=['PoolArea','PoolQC'])
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(14,6)
sns.regplot(x = 'MiscVal', y = 'SalePrice', ax = ax1, data = data)
sns.barplot(x = 'MiscFeature', y = 'SalePrice', ax = ax2, data = data)

all_data = all_data.drop(columns = ['MiscVal','MiscFeature'])
figure, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3)
figure.set_size_inches(14,6)

sns.regplot(x = 'LowQualFinSF', y = 'SalePrice', ax = ax1, data = data)
sns.regplot(x = 'OpenPorchSF', y = 'SalePrice', ax = ax2, data = data)
sns.regplot(x = 'WoodDeckSF', y = 'SalePrice', ax = ax3, data = data)

all_data['NoLowQual'] = (all_data['LowQualFinSF'] == 0)
all_data['NoOpenPorch'] = (all_data['OpenPorchSF'] == 0)
all_data['NoWoodDeck'] = (all_data['WoodDeckSF'] == 0)
non_numeric = all_data.select_dtypes(np.object)

def onehot(col_list):
    global all_data
    while len(col_list) != 0:
        col = col_list.pop(0)
        data_encoded = pd.get_dummies(all_data[col], prefix = col)
        all_data = pd.merge(all_data, data_encoded, on='Id')
        all_data = all_data.drop(columns=col)
    print(all_data.shape)
    
onehot(list(non_numeric))

numeric = all_data.select_dtypes(np.number)

def log_transform(col_list):
    transformed_col = []
    while len(col_list) != 0:
        col = col_list.pop(0)
        if all_data[col].skew() > 0.5:
            all_data[col] = np.log(all_data[col]+1)
            transformed_col.append(col)
        else:
            pass
    print(f"{len(transformed_col)} features had been tranformed")
    print(all_data.shape)

log_transform(list(numeric))

print(train.shape, test.shape)
Xtrain = all_data[:len(train)]
Xtest = all_data[len(train):]
print(Xtrain.shape, Xtest.shape)

#학습 설정
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
import time
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

model_Lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.000327, random_state = 18))
model_ENet = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.00052, l1_ratio = 0.70654, random_state = 18))

model_Lasso.fit(Xtrain, Ytrain)
Lasso_predictions = model_Lasso.predict(Xtest)
train_Lasso = model_Lasso.predict(Xtrain)

model_ENet.fit(Xtrain, Ytrain)
ENet_predictions = model_ENet.predict(Xtest)
train_ENet = model_ENet.predict(Xtrain)

log_train_predictions = (train_Lasso + train_ENet)/2
train_score = np.sqrt(mean_squared_error(Ytrain, log_train_predictions))
print(f"Scoring with train data : {train_score}")

log_predictions = (Lasso_predictions + ENet_predictions) / 2
predictions = np.exp(log_predictions)-1
submission['SalePrice'] = predictions
submission.to_csv('Result.csv')
