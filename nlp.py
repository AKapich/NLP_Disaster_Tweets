import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from get_xy import get_xy
from data_mining import extract_data

df = pd.read_csv('train.csv')

calc_df = extract_data(df)

train, test = np.split(calc_df.sample(frac=1), [int(len(df) * 0.7)])
X_train, y_train = get_xy(train)
X_test, y_test = get_xy(test)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))