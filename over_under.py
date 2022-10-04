import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("df_proc.csv", usecols=[
       'HS', 'AS', 'HST', 'AST', 'HC', 'AC','over',
        'CumGD', 'HPG', 'APG', 'PointsDiff', 'pgh', 'pga'])
print(df.head())

first_column = df.pop('over')

# insert column using insert(position,column_name,
# first_column) function
df.insert(0, 'over', first_column)
X = df.iloc[0:4180, 1:].values
y = df.iloc[0:4180, 0].values
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=16, activation='relu'))
model.add(tf.keras.layers.Dense(units=8, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode='min')
model.fit(X_train, y_train, validation_split=0.1, batch_size=5, epochs=200, verbose=2,
          callbacks=[early_stop])
y_pred = model.predict(X_test)
print(y_test)
print(np.round(abs(y_pred)))
print(classification_report(y_test, np.round(abs(y_pred))))
print(confusion_matrix(y_test, np.round(abs(y_pred))))











