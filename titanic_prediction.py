import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import re

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV

import catboost
from catboost import CatBoostClassifier
from catboost import Pool

#data load
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

#data 전치리
df.head()
df.shape
df.isna().sum()

#PassengerID는 관계가 없다고 판단하여 제거
df = df.drop('PassengerId', axis = 1)
df['Count_Cabins'] = np.nan
df.loc[df['Cabin'].notna(),'Count_Cabins'] = df.loc[df['Cabin'].notna(),'Cabin'].apply(lambda x: len(x.split()))
df['Cabin'] = df['Cabin'].str[0]
df['Name'] = df['Name'].apply(lambda x: x.split(", ")[1].split(".")[0])
df = df.drop('Ticket', axis = 1)
df['Count_Cabins'] = df['Count_Cabins'].astype('Int8')
df.dtypes
df = df.rename(columns = {'Name': 'Status',
                          'Cabin': 'Cabin Type'})
df.head()

#성별 인원 수 확인
Status = df.groupby('Status').agg({'Age' : 'mean'})
Status['Age'] = Status['Age'].astype('int')
Status = Status.loc[df[df['Age'].isna()]['Status'].value_counts().index]
Status

lst = df[df['Age'].isna()]['Status'].value_counts().index

#출항 장소에 따른 데이터 분류
for l in lst:
    df.loc[(df['Age'].isna())&(df['Status'] == l),'Age'] = \
    df.loc[(df['Age'].isna())& (df['Status'] == l),'Age'].fillna(Status.loc[l][0])
df.loc[df['Embarked'].isna(),'Embarked'] = df.loc[df['Embarked'].isna(),'Embarked'].fillna('S')
df.head()
Pclass = df.groupby('Pclass')['Cabin Type'].apply(lambda x: x.mode()[0]).to_frame()
Pclass

#탑승 좌석에 따른 데이터 분류
for i in Pclass.index:
    df.loc[(df['Cabin Type'].isna())&(df['Pclass'] == i),'Cabin Type'] = \
    df.loc[(df['Cabin Type'].isna())&(df['Pclass'] == i),'Cabin Type'].fillna(Pclass.loc[i][0])
df.loc[df['Count_Cabins'].isna(),'Count_Cabins'] = \
df.loc[df['Count_Cabins'].isna(),'Count_Cabins'].fillna(1)
df.isna().sum()
df_test.head()

column_remove = ['PassengerId','Ticket']
column_transformer = ColumnTransformer(transformers = [('drop_columns', 'drop', column_remove)], remainder= 'passthrough')
def create_Count_Cabins(dataframe):
    dataframe['Count_Cabins'] = np.nan
    dataframe.loc[dataframe['Cabin'].notna(),'Count_Cabins'] = \
    dataframe.loc[dataframe['Cabin'].notna(),'Cabin'].apply(lambda x: len(x.split()))
    dataframe['Count_Cabins'] = dataframe['Count_Cabins'].astype('Int8')

new_column_creator = FunctionTransformer(create_Count_Cabins(df_test), validate=False)
def replace_values(dataframe):
    dataframe['Cabin'] = dataframe['Cabin'].str[0]
    dataframe['Name'] = dataframe['Name'].apply(lambda x: x.split(", ")[1].split(".")[0])
value_replace = FunctionTransformer(replace_values(df_test),validate=False)
pipeline = Pipeline([
    ('column_transformer', column_transformer),
    ('new_column', new_column_creator),
    ('value_replacer', value_replace)
])

processed_data = pipeline.fit_transform(df_test)
processed_df = pd.DataFrame(processed_data)
df.columns[1:]

processed_df.columns = df.columns[1:]
processed_df.head()
Pclass = processed_df.groupby('Pclass')['Cabin Type'].apply(lambda x: x.mode()[0]).to_frame()
Pclass

for i in Pclass.index:
    processed_df.loc[(processed_df['Cabin Type'].isna())&(processed_df['Pclass'] == i),'Cabin Type'] = \
    processed_df.loc[(processed_df['Cabin Type'].isna())&(processed_df['Pclass'] == i),'Cabin Type']\
    .fillna(Pclass.loc[i][0])
processed_df.loc[processed_df['Count_Cabins'].isna(),'Count_Cabins'] = \
processed_df.loc[processed_df['Count_Cabins'].isna(),'Count_Cabins'].fillna(1)
fare_value = round(processed_df['Fare'].median(),2)

processed_df.loc[processed_df['Fare'].isna(),'Fare'] = \
processed_df.loc[processed_df['Fare'].isna(),'Fare'].fillna(fare_value)
lst = processed_df[processed_df['Age'].isna()]['Status'].value_counts().index

Status = processed_df.groupby('Status').agg({'Age' : 'mean'})
Status[Status['Age'].isna()] = Status[Status['Age'].isna()].fillna(processed_df['Age'].median())
Status['Age'] = Status['Age'].astype('int')

Status

for l in lst:
    processed_df.loc[(processed_df['Age'].isna())&(processed_df['Status'] == l),'Age'] = \
    processed_df.loc[(processed_df['Age'].isna())& (processed_df['Status'] == l),'Age'].fillna(Status.loc[l][0])
processed_df[['Pclass','SibSp','Parch','Count_Cabins']] = \
processed_df[['Pclass','SibSp','Parch','Count_Cabins']].astype('int')

processed_df[['Age','Fare']] = \
processed_df[['Age','Fare']].astype('float')
processed_df.dtypes
processed_df.head()

#학습 설정
X_train = df.drop('Survived', axis = 1)
y_train = df['Survived']
X_test = processed_df
cat_features = [0,1,2,7,8]
pool_train = Pool(data = X_train, label = y_train, cat_features = cat_features)
print('Train Pool')
print(pool_train.get_feature_names())
print(pool_train.shape)

model = CatBoostClassifier(iterations = 20,
                           learning_rate = 0.21,
                           l2_leaf_reg = 6,
                           depth = 5,
                           eval_metric = 'Accuracy'
                           )

model.fit(pool_train,
          verbose= 1,
          plot = True)

prediction = model.predict(X_test)
prediction

PassengerId = df_test['PassengerId']
Submition = {
        'PassengerId': PassengerId,
        'Survived': prediction
       }

Submition = pd.DataFrame(Submition)
Submition

Submition.to_csv('CatBoost.csv', index = False)
