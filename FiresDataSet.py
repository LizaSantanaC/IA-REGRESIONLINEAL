import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

dataset = pd.read_csv('forestfires.csv')
X= dataset.iloc[:, 0:12].values
y= dataset.iloc[:, 12].values
print(dataset)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

LinearRegression(copy_X= True, fit_intercept= True, n_jobs= None, normalize= False)
y_pred= model.predict(X_test)

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

print('MSE= ', mse(y_pred, y_test))
print('MAE= ', mae(y_pred, y_test))
print('R2_SCORE= ', r2_score(y_pred, y_test))

MSE= 17008.717227119996
MAE= 31.696653019693787
R2_Score= -328.217991752422

for dirname, _, filenames in os.walk('/home/liza/PycharmProjects/RegresionLineal-ForestFiresDataSet'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/home/liza/PycharmProjects/RegresionLineal-ForestFiresDataSet/forestfires.csv', encoding='latin1')
df.tail()

queim_sum_ano = df.groupby(['year'], as_index=False).sum()

piores_anos = queim_sum_ano[queim_sum_ano['number']>queim_sum_ano['number'].mean() + queim_sum_ano['number'].std()]
print('Worst years: ')
for i in range(len(piores_anos)):
    print(piores_anos['year'].values[i])
plt.figure(figsize=[12,7])
plt.xlim([1998, 2017])
plt.title('Registered number of fires per year (sum of all entries)')
sns.lineplot(x='year', y='number',data=queim_sum_ano);

df['number'].describe()