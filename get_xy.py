# def get_xy(df):
#     return df[df.columns[:-1]].values, df[df.columns[-1]].values


import numpy as np
from sklearn.preprocessing import StandardScaler


def get_xy(df):
  X = df[df.columns[:-2]].values
  y = df[df.columns[-1]].values
  scaler = StandardScaler()
  # actually makes things worse
  # url_col = df['has_url'].values
  
  # X = np.hstack((scaler.fit_transform(X), np.reshape(url_col, (-1,1))))

  return X, y