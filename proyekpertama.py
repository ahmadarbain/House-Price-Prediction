# -*- coding: utf-8 -*-
"""proyekPertama.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rAvB9fPK43tvWtTNK1XWnk6cIHK01ITx

# Data Loading

## Import Library
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

from google.colab import drive

"""## Connect to google drive"""

drive.mount('/content/drive')

"""## Unzip file data """

!unzip  '/content/drive/MyDrive/Dataset/archive.zip' -d '/content/drive/MyDrive/Dataset/'

"""## Load dataset

overview 5 Top list dataset
"""

url = '/content/drive/MyDrive/Dataset/kc_house_data.csv'

df = pd.read_csv(url)
df.head()

"""# Exploratory Data Analysis

## Melihat informasi pada dataset
"""

df.info()

"""## Melihat gambaran dataset"""

df.describe()

"""## Data Wrangling

### Menghapus kolom id yang tidak memiliki korelasi dengan kolom price
"""

df.drop(['id'],axis=1,inplace=True)

"""### Melihat gambaran dataset setelah kolom id di hapus"""

df.describe()

"""## merubah format date """

df.date = pd.to_datetime(df.date, infer_datetime_format=True)
df.head()

df['month'] = df["date"].dt.month
df['day'] = df['date'].dt.day

"""## mencari keterkaitan date denggan price"""

plt.figure(figsize = (10, 5))
sns.lineplot(x = df['date'], y = df['price'])

"""###Mencari relasi date month dengan price"""

plt.figure(figsize = (10, 5))
sns.boxplot(x = df['month'], y = df['price'])

"""### Mencari keterkaian antara date day dengan price"""

plt.figure(figsize = (10, 5))
sns.boxplot(x = df['day'], y=  df['price'])

"""### Menghapus kolom date, day, dan month karena tidak ditemukannya keterkaitan"""

df.drop(columns = ['date', 'day', 'month'], inplace = True)

"""## Melihat informasi pada dataset """

df.info()

"""## Melihat visualisasi outliner"""

f = plt.figure(figsize=(16, 6))

f.add_subplot(1,3,1)
sns.boxplot(df['bedrooms']) 

f.add_subplot(1,3,2)
sns.boxplot(df['bathrooms'])

f.add_subplot(1,3,3)
sns.boxplot(df['view'])

"""## Mengatasi outlier menggunakan metode IQR"""

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR=Q3-Q1
df=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
 
# Cek ukuran dataset setelah kita drop outliers
df.shape

"""## Mengecek kolum pada dataset"""

df.columns

"""## Mencari sebaran data keterkaitan setiap kolom dengan kolom price"""

plt.figure(figsize=(10,8))
sns.pairplot(data=df, x_vars=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15'], y_vars=['price'], size=5, aspect=0.75)

"""## Mencari Korelasi seluruh fitur dengan fitur price"""

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

"""## Menghapus kolom yang tidak memiliki keterkaitan dengan price"""

df.drop(columns = ['sqft_lot', 'waterfront', 'view', 'condition', 'yr_built', 
                   'zipcode', 'long', 'sqft_lot15', 'yr_renovated'], inplace = True)

"""# Data Preparation

## Split Test menjadi data Training dan Test 80% 20%
"""

from sklearn.model_selection import train_test_split
 
X = df.drop(["price"],axis =1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

df.head()

"""# Model Development

## Melakukan Standarisasi
"""

from sklearn.preprocessing import StandardScaler
 
numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'grade',
       'sqft_above', 'sqft_basement', 'lat', 'sqft_living15']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

"""## Develop 3 model Alogaritam"""

models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'Boosting'])

"""### Melakukan Development menggunakan alogaritma KNN"""

from sklearn.neighbors import KNeighborsRegressor
 
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_train)

"""### Melakukan Development menggunakan alogaritma Random Forest"""

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
 
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""### Melakukan Development menggunakan alogaritma Boosting"""

from sklearn.ensemble import AdaBoostRegressor
 
boosting = AdaBoostRegressor(n_estimators=50, learning_rate=0.05, random_state=55)                             
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""## Standarisai pada data test

Melakukan standarisai pada data test dengan menggunakan transform scaler
"""

X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

"""## Evalusai model menggunakan MSE"""

mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
 
mse

"""### Visualisasi Alogaritma 

KNN, Random Forest, dan Boosting
"""

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

"""# Evaluasi Model

## Melakukan prediksi pada data test

membandingkan ketiga algoritma pada data test dengan menentukan prediksi terhadap data test
"""

prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
 
pd.DataFrame(pred_dict)