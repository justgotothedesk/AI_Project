import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn import set_config

#data load
df_train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
df_submission = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')

df_train.head()
df_train.info()
df_train.columns

#각 항목 별, value 값 확인
df_train['HomePlanet'].unique()
df_train['CryoSleep'].unique()
df_train['Transported'].unique()

row1,column1 = df_train.shape
print('Train Rows: {} & Columns: {} '.format(row1,column1))
row2,column2 = df_test.shape
print('Test Rows: {} & Columns: {} '.format(row2,column2))

df_test.head()
df_test.columns
df_test.isnull().sum()
df_train.isnull().sum()
df_test['Cabin']
df_train.info()

#Cabin 값을 분류하여 새로운 항목 생성
df_train[['Deck','Num','Side']] = df_train.Cabin.str.split('/',expand = True)
df_test[['Deck','Num','Side']] = df_test.Cabin.str.split('/',expand = True)

#편의 시설 비용 합치기
df_train['total_spent'] = df_train['RoomService']+ df_train['FoodCourt']+ df_train['ShoppingMall']+ df_train['Spa']+ df_train['VRDeck']
df_test['total_spent'] = df_test['RoomService']+df_test['FoodCourt']+df_test['ShoppingMall']+df_test['Spa']+df_test['VRDeck']

#승객 나이
df_train['AgeGroup'] = 0
for i in range(6):
    df_train.loc[(df_train.Age >= 10*i) & (df_train.Age < 10*(i+1)),'AgeGroup'] = i

df_train.head()

df_test['AgeGroup'] = 0
for i in range(6):
    df_test.loc[(df_test.Age >= 10*i) & (df_test.Age < 10*(i+1)),'AgeGroup'] = i

#Deck 별, 결과 확인
sns.countplot(x = df_train.Deck, hue = df_train.Transported);

X = df_train.drop('Transported',axis=1)
y = df_train['Transported']

X['Num'] = pd.to_numeric(X['Num'])
#이름은 결과에 영향을 끼치지 않는다고 판단하여 제거
X = X.drop(['PassengerId','Name'], axis = 1)

cat_cols = X.select_dtypes('object').columns.to_list()
cat_cols
num_cols = X.select_dtypes(exclude='object').columns.to_list()
num_cols

#전처리 설정
numeric_preprocessor = Pipeline(steps = [
    ('imputer',SimpleImputer(strategy = 'mean')),
    ('scaling',StandardScaler()),
])
categorical_preprocessor = Pipeline(steps = [
    ('encoder',OneHotEncoder(handle_unknown = 'ignore')),
    ('imputer',SimpleImputer(strategy = 'constant')),
])
numeric_preprocessor
categorical_preprocessor

#학습 설정
preprocessor = ColumnTransformer([
    ('categorical', categorical_preprocessor,cat_cols),
    ('numeric', numeric_preprocessor,num_cols)
])
Pipe = Pipeline(steps = [
    ('preprocessor',preprocessor),
    ('model',GradientBoostingClassifier())
])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 101)
Pipe.fit(X_train, y_train)

pred = Pipe.predict(X_val)
accuracy_score(y_val.values,pred)

#parameter 설정 및 도출
param_grid = {'model__n_estimators' : [500,1000], 'model__learning_rate' : [0.1,0.2], 'model__verbose' : [1], 'model__max_depth' : [2,3]}
gcv = GridSearchCV(Pipe, param_grid = param_grid, cv = 5, scoring = "roc_auc")
gcv.fit(X,y)

params = gcv.best_params_

Hyper_Pipe = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier(n_estimators = 500, max_depth = 3, random_state = 1)),
])
Hyper_Pipe.fit(X_train, y_train)
y_pred = Hyper_Pipe.predict(df_test)

sub = pd.DataFrame({'Transported' : y_pred.astype(bool)}, index = df_test['PassengerId'])
sub.head()
sub.to_csv('submission.csv')
