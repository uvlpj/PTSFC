#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
def compute_return(y, r_type="log", h=1):
    
    # exclude first h observations
    y2 = y[h:]
    # exclude last h observations
    y1 = y[:-h]
    
    if r_type == "log":
        ret = np.concatenate(([np.nan]*h, 100 * (np.log(y2) - np.log(y1))))
    else:
        ret = np.concatenate(([np.nan]*h, 100 * (y2-y1)/y1))
        
    return ret


#%%
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Abagbe3_^GDAXI.csv')
#data = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Abgab2^GDAXI.csv')
#data = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Abgabe_29_11_23^GDAXI.csv')
#data = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission5/^GDAXI Kopie.csv')
#data = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission6/^GDAXI.csv')
data = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission8/^GDAXI.csv')
#%%
#hist2 = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/22_11_23^GDAXI.csv')
#hist2 = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission5/^GDAXI.csv')
#hist2 = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission6/^GDAXI Kopie.csv')
hist2 = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission8/^GDAXI Kopie 3.csv')
#%%

#%%
for i in range(5):
    data["ret"+str(i+1)] = compute_return(data["Close"].values, h=i+1)

#%%
for i in range(5):
    hist2["ret"+str(i+1)] = compute_return(hist2["Close"].values, h=i+1)    

#%%
'''Hat noch den letzen Datenpunkt vom Mittwoch welcher in hist2 nicht vorhanden ist'''
last_row = hist2.tail(1).copy()

#%%
frames = [data, last_row]
hist = pd.concat(frames)

#%%
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as pl

# %%

#%%
hist = hist.dropna()

# %%
del hist['Open']
del hist['High']
del hist['Low']
del hist['Volume']
del hist['Adj Close']
# %%

# %%
hist['Rt'] = 100 * (np.log(hist['Close']) - np.log(hist['Close'].shift(1)))
#%%
hist = hist.dropna()
#%%
hist
# %%
hist['Date'] = pd.to_datetime(hist['Date'],infer_datetime_format=True)
hist = hist.set_index(['Date'])
# %%
hist
#%%

hist['price_up'] = np.where(hist['Close'] > hist['Close'].shift(1), 1, 0)

# Variable für Renditenänderung erstellen (1: gestiegen, 0: gleich geblieben oder gefallen)
hist['ret1_up'] = np.where(hist['ret1'] > 0, 1, 0)
hist['ret2_up'] = np.where(hist['ret2'] > 0, 1, 0)
hist['ret3_up'] = np.where(hist['ret3'] > 0, 1, 0)
hist['ret4_up'] = np.where(hist['ret4'] > 0, 1, 0)
hist['ret5_up'] = np.where(hist['ret5'] > 0, 1, 0)
#%%
'''
-----
Binärer Prozess für die Aktivität
Does the price move or not
-----
'''

#%%
'''
----
Binärer Prozess für die Richtung der Preisbewegung
-----
'''

#%%

#%%
hist = hist.dropna()
#%%
hist
# %%
train_size = int(len(hist) * 0.8)
train, test = hist[:train_size], hist[train_size:]
# %%
train_size
# %%
train
# %%
test
# %%
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from scipy import stats
#%%

def QuantileLoss(perc, delta=1e-4):
    perc = np.array(perc).reshape(-1)
    perc.sort()
    perc = perc.reshape(1, -1)
    def _qloss(y, pred):
        I = tf.cast(y <= pred, tf.float32)
        d = K.abs(y - pred)
        correction = I * (1 - perc) + (1 - I) * perc
        # huber loss
        huber_loss = K.sum(correction * tf.where(d <= delta, 0.5 * d ** 2 / delta, d - 0.5 * delta), -1)
        # order loss
        q_order_loss = K.sum(K.maximum(0.0, pred[:, :-1] - pred[:, 1:] + 1e-6), -1)
        return huber_loss + q_order_loss
    return _qloss

#%%
np.random.seed(42)
from tensorflow.random import set_seed
set_seed(42)
#%%
'''
---------
=> Deep Quantile Regression in Keras
For h = 1 
Quantile Predictions for the log returns ret1
---------
'''
X = sm.add_constant(hist['Rt'].abs().iloc[:-1]) 
Y = hist['ret1'].shift(-1).iloc[:-1]  # Rt+1:t+1
X = X.values
Y = Y.values
perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])
model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate = 2e-3), loss=QuantileLoss(perc_points))
model.fit(X, Y, epochs=100, verbose=0)
# %%
# Assuming 'quantiles' is a 1-dimensional array or list
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

recent_Rt = hist['Rt'].iloc[-1]
new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
new_data = new_data.values

# Vorhersagen für die Quantile machen
predictions = model.predict(new_data)

# Ergebnisse in einem DataFrame speichern
pred_quantiles = pd.DataFrame(index=quantiles, columns=['predicted_ret1'])
pred_quantiles['predicted_ret1'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'

# Ergebnisse anzeigen
print(pred_quantiles)

#%%
''' 
--------------------------------------------------
Deep Quantile Regression With ROLLING WINDOW ANSATZ
Grid Search
--------------------------------------------------

-----
Model1 => ret1
-----
'''
window_sizes = [10, 20, 30]
step_sizes = [5, 10, 15]
X1 = sm.add_constant(hist[['Rt']].abs().iloc[:-1])
Y1 = hist['ret1'].shift(-1).iloc[:-1]  # Rt+1:t+1
for window_size in window_sizes:
    for step_size in step_sizes:
        # Trainingsdaten und Validierungsdaten erstellen
        X1_train, X1_val, y1_train, y1_val = train_test_split(X1, Y1, test_size=0.2, shuffle=False, random_state = 42)

        # Modellstruktur
        model1 = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(5)
        ])

        model1.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

        # Trainings- und Validierungsverluste überwachen
        train_losses, val_losses = [], []
        for i in range(0, len(X1_train) - window_size, step_size):
            X_window = X1_train.iloc[i:i+window_size]
            y_window = y1_train.iloc[i:i+window_size]

            # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
            X_train, X_val, y_train, y_val = train_test_split(X_window, y_window, test_size=0.2, shuffle=False, random_state = 42)

            history = model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

            # Erfasse den Trainings- und Validierungsverlust
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Ausgabe der Ergebnisse für die aktuelle Hyperparameter-Kombination
        print(f"Window Size: {window_size}, Step Size: {step_size}")
        print(f"Final Training Loss: {train_losses[-1]}, Final Validation Loss: {val_losses[-1]}")
        print("=" * 50)
#%%
'''
------
Model 1
=> return 1
=> mit dem Rolling Window Ansatz für das Training des Models
------
'''
X1 = sm.add_constant(hist[['Rt']].abs().iloc[:-1])
Y1 = hist['ret1'].shift(-1).iloc[:-1]  # Rt+1:t+1
X1 = X1.values
Y1 = Y1.values

perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]

model_1 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)])

model_1.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate = 2e-3), loss=QuantileLoss(perc_points))

window_size = 20
step_size = 10

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.2, shuffle=False, random_state = 42)
X1_test, X1_val, y1_test, y1_val = train_test_split(X1_test, y1_test, test_size=0.5, shuffle=False, random_state = 42)

# Trainiere das Modell mit dem Rolling Window-Ansatz
for i in range(0, len(X1_train) - window_size, step_size):
    X_window = X1_train[i:i+window_size]
    y_window = y1_train[i:i+window_size]
    
    model_1.fit(X_window, y_window, epochs=10, verbose=0)


test_predictions1 = model_1.predict(X1_test)

val_predictions1 = model_1.predict(X1_val)

# Erstelle Vorhersagen für das neueste Datenfenster
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret1'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])

# Mache Vorhersagen für die Quantile
predictions = model_1.predict(new_data)

# Ergebnisse in einem DataFrame speichern
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pred_quant__1 = pd.DataFrame(index=quantiles, columns=['predicted_ret1'])
pred_quant__1['predicted_ret1'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'
print(pred_quant__1)

#%%

# %%
'''
---------
=> Deep Quantile Regression in Keras
For h = 2 
Quantile Predictions for the log returns ret2
---------
'''
X = sm.add_constant(hist['Rt'].abs().iloc[:-1]) 
Y = hist['ret2'].shift(-1).iloc[:-1]  # Rt+1:t+1
X = X.values
Y = Y.values
perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]
model2 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])
model2.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate = 2e-3), loss=QuantileLoss(perc_points))
model2.fit(X, Y, epochs=100, verbose=0)
# %%
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
recent_Rt = hist['Rt'].iloc[-1]
new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
new_data = new_data.values

# Vorhersagen für die Quantile machen
predictions = model2.predict(new_data)

# Ergebnisse in einem DataFrame speichern
pred_quantiles2 = pd.DataFrame(index=quantiles, columns=['predicted_ret2'])

pred_quantiles2['predicted_ret2'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'
#%%
# Ergebnisse anzeigen
print(pred_quantiles2)

#%%

''' 
--------------------------------------------------
Deep Quantile Regression With ROLLING WINDOW ANSATZ
Grid Search
--------------------------------------------------

-----
Model2 => ret2
-----
'''
window_sizes = [10, 20, 30]
step_sizes = [5, 10, 15]
X2 = sm.add_constant(hist[['Rt']].abs().iloc[:-1])
Y2 = hist['ret2'].shift(-1).iloc[:-1]  # Rt+1:t+1
X2 = X2.values
Y2 = Y2.values
for window_size in window_sizes:
    for step_size in step_sizes:
        # Trainingsdaten und Validierungsdaten erstellen
        X2_train, X2_val, y2_train, y2_val = train_test_split(X2, Y2, test_size=0.2, shuffle=False, random_state = 42)

        # Modellstruktur
        model2 = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(5)
        ])

        model2.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate = 2e-3), loss=QuantileLoss(perc_points))

        # Trainings- und Validierungsverluste überwachen
        train_losses, val_losses = [], []
        for i in range(0, len(X1_train) - window_size, step_size):
            X_window = X2_train.iloc[i:i+window_size]
            y_window = y2_train.iloc[i:i+window_size]

            # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
            X_train, X_val, y_train, y_val = train_test_split(X_window, y_window, test_size=0.2, shuffle=False, random_state = 42)

            history = model2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

            # Erfasse den Trainings- und Validierungsverlust
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Ausgabe der Ergebnisse für die aktuelle Hyperparameter-Kombination
        print(f"Window Size: {window_size}, Step Size: {step_size}")
        print(f"Final Training Loss: {train_losses[-1]}, Final Validation Loss: {val_losses[-1]}")
        print("=" * 50)

#%%
'''
------
Model2 with optimal window size and Step size after Grid search
------
'''
X2 = sm.add_constant(hist[['Rt']].abs().iloc[:-1])
Y2 = hist['ret2'].shift(-1).iloc[:-1]  # Rt+1:t+1
X2 = X2.values
Y2 = Y2.values

perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]

model_2 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)])

model_2.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate = 2e-3), loss=QuantileLoss(perc_points))

window_size = 20
step_size = 10

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size=0.2, shuffle=False, random_state = 42)
X2_test, X2_val, y2_test, y2_val = train_test_split(X2_test, y2_test, test_size=0.5, shuffle=False, random_state = 42)

# Trainiere das Modell mit dem Rolling Window-Ansatz
for i in range(0, len(X2_train) - window_size, step_size):
    X_window = X2_train[i:i+window_size]
    y_window = y2_train[i:i+window_size]
    
    model_2.fit(X_window, y_window, epochs=10, verbose=0)


test_predictions2 = model_2.predict(X1_test)

val_predictions2 = model_2.predict(X1_val)

# Erstelle Vorhersagen für das neueste Datenfenster
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret2'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])

# Mache Vorhersagen für die Quantile
predictions = model_2.predict(new_data)

# Ergebnisse in einem DataFrame speichern
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pred_quant__2 = pd.DataFrame(index=quantiles, columns=['predicted_ret2'])
pred_quant__2['predicted_ret2'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'
print(pred_quant__2)



# %%
'''
---------
=> Deep Quantile Regression in Keras
For h = 3
Quantile Predictions for the log returns ret3
---------
'''
X = sm.add_constant(hist['Rt'].abs().iloc[:-1]) 
Y = hist['ret3'].shift(-1).iloc[:-1]  # Rt+1:t+1
X = X.values
Y = Y.values
perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]
model3 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])
model3.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate = 2e-3), loss=QuantileLoss(perc_points))
model3.fit(X, Y, epochs=100, verbose=0)
# %%
recent_Rt = hist['Rt'].iloc[-1]
new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])

# Vorhersagen für die Quantile machen
predictions = model3.predict(new_data)

# Ergebnisse in einem DataFrame speichern
pred_quantiles3 = pd.DataFrame(index=quantiles, columns=['predicted_ret3'])

pred_quantiles3['predicted_ret3'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'
#%%
print(pred_quantiles3)
#%%

''' 
--------------------------------------------------
Deep Quantile Regression With ROLLING WINDOW ANSATZ
Grid Search
--------------------------------------------------

-----
Model3 => ret3
-----
'''
window_sizes = [10, 20, 30]
step_sizes = [5, 10, 15]
X3 = sm.add_constant(hist[['Rt']].abs().iloc[:-1])
Y3 = hist['ret3'].shift(-1).iloc[:-1]  # Rt+1:t+1
for window_size in window_sizes:
    for step_size in step_sizes:
        # Trainingsdaten und Validierungsdaten erstellen
        X3_train, X3_val, y3_train, y3_val = train_test_split(X3, Y3, test_size=0.2, shuffle=False, random_state = 42)

        # Modellstruktur
        model3 = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(5)
        ])

        model3.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

        # Trainings- und Validierungsverluste überwachen
        train_losses, val_losses = [], []
        for i in range(0, len(X1_train) - window_size, step_size):
            X_window = X3_train.iloc[i:i+window_size]
            y_window = y3_train.iloc[i:i+window_size]

            # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
            X_train, X_val, y_train, y_val = train_test_split(X_window, y_window, test_size=0.2, shuffle=False, random_state = 42)

            history = model2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

            # Erfasse den Trainings- und Validierungsverlust
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Ausgabe der Ergebnisse für die aktuelle Hyperparameter-Kombination
        print(f"Window Size: {window_size}, Step Size: {step_size}")
        print(f"Final Training Loss: {train_losses[-1]}, Final Validation Loss: {val_losses[-1]}")
        print("=" * 50)


#%%
'''
------
Model3 with optimal window size and Step size after Grid search
------
'''
X3 = sm.add_constant(hist[['Rt']].abs().iloc[:-1])
Y3 = hist['ret3'].shift(-1).iloc[:-1]  # Rt+1:t+1
X3 = X3.values
Y3 = Y3.values

perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]

model_3 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)])

model_3.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate = 2e-3), loss=QuantileLoss(perc_points))

window_size = 10
step_size = 15

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, Y3, test_size=0.2, shuffle=False, random_state = 42)
X3_test, X3_val, y3_test, y3_val = train_test_split(X3_test, y3_test, test_size=0.5, shuffle=False, random_state = 42)

# Trainiere das Modell mit dem Rolling Window-Ansatz
for i in range(0, len(X3_train) - window_size, step_size):
    X_window = X3_train[i:i+window_size]
    y_window = y3_train[i:i+window_size]
    
    model_3.fit(X_window, y_window, epochs=10, verbose=0)


test_predictions2 = model_3.predict(X3_test)

val_predictions2 = model_3.predict(X3_val)

# Erstelle Vorhersagen für das neueste Datenfenster
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret3'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])

# Mache Vorhersagen für die Quantile
predictions = model_3.predict(new_data)

# Ergebnisse in einem DataFrame speichern
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pred_quant__3 = pd.DataFrame(index=quantiles, columns=['predicted_ret3'])
pred_quant__3['predicted_ret3'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'
print(pred_quant__3)



# %%
'''
---------
=> Deep Quantile Regression in Keras
For h = 4
Quantile Predictions for the log returns ret4
---------
'''
X = sm.add_constant(hist['Rt'].abs().iloc[:-1]) 
Y = hist['ret4'].shift(-1).iloc[:-1]  # Rt+1:t+1
X = X.values
Y = Y.values
perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]
model4 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])
model4.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate = 2e-3), loss=QuantileLoss(perc_points))
model4.fit(X, Y, epochs=100, verbose=0)
# %%
recent_Rt = hist['Rt'].iloc[-1]
new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])

# Vorhersagen für die Quantile machen
predictions = model4.predict(new_data)

# Ergebnisse in einem DataFrame speichern
pred_quantiles4 = pd.DataFrame(index=quantiles, columns=['predicted_ret4'])

pred_quantiles4['predicted_ret4'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'
#%%
print(pred_quantiles4)


#%%

''' 
--------------------------------------------------
Deep Quantile Regression With ROLLING WINDOW ANSATZ
Grid Search
--------------------------------------------------

-----
Model4 => ret4
-----
'''
window_sizes = [10, 20, 30]
step_sizes = [5, 10, 15]
X4 = sm.add_constant(hist[['Rt']].abs().iloc[:-1])
Y4 = hist['ret4'].shift(-1).iloc[:-1]  # Rt+1:t+1
for window_size in window_sizes:
    for step_size in step_sizes:
        # Trainingsdaten und Validierungsdaten erstellen
        X4_train, X4_val, y4_train, y4_val = train_test_split(X4, Y4, test_size=0.2, shuffle=False, random_state = 42)

        # Modellstruktur
        model4 = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(5)
        ])

        model4.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

        # Trainings- und Validierungsverluste überwachen
        train_losses, val_losses = [], []
        for i in range(0, len(X1_train) - window_size, step_size):
            X_window = X4_train.iloc[i:i+window_size]
            y_window = y4_train.iloc[i:i+window_size]

            # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
            X_train, X_val, y_train, y_val = train_test_split(X_window, y_window, test_size=0.2, shuffle=False, random_state = 42)

            history = model4.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

            # Erfasse den Trainings- und Validierungsverlust
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Ausgabe der Ergebnisse für die aktuelle Hyperparameter-Kombination
        print(f"Window Size: {window_size}, Step Size: {step_size}")
        print(f"Final Training Loss: {train_losses[-1]}, Final Validation Loss: {val_losses[-1]}")
        print("=" * 50)

#%%

'''
------
Model4 with optimal window size and Step size after Grid search
------
'''
X4 = sm.add_constant(hist[['Rt']].abs().iloc[:-1])
Y4 = hist['ret4'].shift(-1).iloc[:-1]  # Rt+1:t+1
X4 = X4.values
Y4 = Y4.values


perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]

model_4 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)])

model_4.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate = 2e-3), loss=QuantileLoss(perc_points))

window_size = 10
step_size = 10

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, Y4, test_size=0.2, shuffle=False, random_state = 42)
X4_test, X4_val, y4_test, y4_val = train_test_split(X4_test, y4_test, test_size=0.5, shuffle=False, random_state = 42)

# Trainiere das Modell mit dem Rolling Window-Ansatz
for i in range(0, len(X4_train) - window_size, step_size):
    X_window = X4_train[i:i+window_size]
    y_window = y4_train[i:i+window_size]
    
    model_4.fit(X_window, y_window, epochs=10, verbose=0)


test_predictions4 = model_4.predict(X4_test)

val_predictions4 = model_4.predict(X4_val)

# Erstelle Vorhersagen für das neueste Datenfenster
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret4'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])

# Mache Vorhersagen für die Quantile
predictions = model_4.predict(new_data)

# Ergebnisse in einem DataFrame speichern
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pred_quant__4 = pd.DataFrame(index=quantiles, columns=['predicted_ret4'])
pred_quant__4['predicted_ret4'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'
print(pred_quant__4)









# %%
'''
---------
=> Deep Quantile Regression in Keras
For h = 5
Quantile Predictions for the log returns ret4
---------
'''
X = sm.add_constant(hist['Rt'].abs().iloc[:-1]) 
Y = hist['ret5'].shift(-1).iloc[:-1]  # Rt+1:t+1
X = X.values
Y = Y.values
perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]
model5 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])
model5.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate = 2e-3), loss=QuantileLoss(perc_points))
model5.fit(X, Y, epochs=100, verbose=0)
# %%
recent_Rt = hist['Rt'].iloc[-1]
new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])

# Vorhersagen für die Quantile machen
predictions = model5.predict(new_data)

# Ergebnisse in einem DataFrame speichern
pred_quantiles5 = pd.DataFrame(index=quantiles, columns=['predicted_ret5'])

pred_quantiles5['predicted_ret5'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'
#%%
print(pred_quantiles5)


#%%

''' 
--------------------------------------------------
Deep Quantile Regression With ROLLING WINDOW ANSATZ
Grid Search
--------------------------------------------------

-----
Model5 => ret5
-----
'''
window_sizes = [10, 20, 30]
step_sizes = [5, 10, 15]
X5 = sm.add_constant(hist[['Rt']].abs().iloc[:-1])
Y5 = hist['ret5'].shift(-1).iloc[:-1]  # Rt+1:t+1
for window_size in window_sizes:
    for step_size in step_sizes:
        # Trainingsdaten und Validierungsdaten erstellen
        X5_train, X5_val, y5_train, y5_val = train_test_split(X5, Y5, test_size=0.2, shuffle=False, random_state = 42)

        # Modellstruktur
        model5 = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(5)
        ])

        model5.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

        # Trainings- und Validierungsverluste überwachen
        train_losses, val_losses = [], []
        for i in range(0, len(X1_train) - window_size, step_size):
            X_window = X5_train.iloc[i:i+window_size]
            y_window = y5_train.iloc[i:i+window_size]

            # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
            X_train, X_val, y_train, y_val = train_test_split(X_window, y_window, test_size=0.2, shuffle=False, random_state = 42)

            history = model5.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

            # Erfasse den Trainings- und Validierungsverlust
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Ausgabe der Ergebnisse für die aktuelle Hyperparameter-Kombination
        print(f"Window Size: {window_size}, Step Size: {step_size}")
        print(f"Final Training Loss: {train_losses[-1]}, Final Validation Loss: {val_losses[-1]}")
        print("=" * 50)

#%%
        

'''
------
Model5 with optimal window size and Step size after Grid search
------
'''

X5 = sm.add_constant(hist[['Rt']].abs().iloc[:-1])
Y5 = hist['ret5'].shift(-1).iloc[:-1]  # Rt+1:t+1
X5 = X5.values
Y5 = Y5.values

perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]

model_5 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)])

model_5.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate = 2e-3), loss=QuantileLoss(perc_points))

window_size = 10
step_size = 15

X5_train, X5_test, y5_train, y5_test = train_test_split(X5, Y5, test_size=0.2, shuffle=False, random_state = 42)
X5_test, X5_val, y5_test, y5_val = train_test_split(X5_test, y5_test, test_size=0.5, shuffle=False, random_state = 42)

# Trainiere das Modell mit dem Rolling Window-Ansatz
for i in range(0, len(X5_train) - window_size, step_size):
    X_window = X5_train[i:i+window_size]
    y_window = y5_train[i:i+window_size]
    
    model_5.fit(X_window, y_window, epochs=10, verbose=0)


test_predictions5 = model_5.predict(X5_test)

val_predictions5 = model_5.predict(X5_val)

# Erstelle Vorhersagen für das neueste Datenfenster
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret5'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])

# Mache Vorhersagen für die Quantile
predictions = model_5.predict(new_data)

# Ergebnisse in einem DataFrame speichern
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pred_quant__5 = pd.DataFrame(index=quantiles, columns=['predicted_ret5'])
pred_quant__5['predicted_ret5'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'
print(pred_quant__5)


#%%

'''
Predcitions of the Deep Quantile Regression without additional variables
=> optimized window size and step size 
=> after Rolling window ansatz
'''
p1 = pred_quant__1.values
p2 = pred_quant__2.values
p3 = pred_quant__3.values
p4 = pred_quant__4.values
p5 = pred_quant__5.values
#%%
p1 = p1.reshape(-1,1)
p2 = p2.reshape(-1,1)
p3 = p3.reshape(-1,1)
p4 = p4.reshape(-1,1)
p5 = p5.reshape(-1,1)
#%%
forecasts_deep_qaunt_without = np.column_stack((p1, p2, p3, p4, p5))
forecasts_deep_qaunt_without = forecasts_deep_qaunt_without.T

#%%
import datetime as dt
#%%
forecastdate = dt.datetime(2024, 1, 17, 00, 00)
#%%
df_sub_dax_DeepQuant_without = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "DAX",
    "horizon": [str(i) + " day" for i in (1,2,5,6,7)],
    "q0.025": forecasts_deep_qaunt_without [:,0],
    "q0.25": forecasts_deep_qaunt_without [:,1],
    "q0.5": forecasts_deep_qaunt_without [:,2],
    "q0.75": forecasts_deep_qaunt_without [:,3],
    "q0.975": forecasts_deep_qaunt_without [:,4]})
df_sub_dax_DeepQuant_without

#%%
PATH = "/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/PTSFC Submission JahrMonatTag_Rachel"

df_sub_dax_DeepQuant_without.to_csv(PATH+ "DeepQauntile_DAX_Abgabe8.csv", index=False)

# %%
pred1 = pred_quantiles.values
pred2 = pred_quantiles2['predicted_ret2'].values
pred3 = pred_quantiles3['predicted_ret3'].values
pred4 = pred_quantiles4['predicted_ret4'].values
pred5 = pred_quantiles5['predicted_ret5'].values
# %%
pred1 = pred1.reshape(-1,1)
pred2 = pred2.reshape(-1,1)
pred3 = pred3.reshape(-1,1)
pred4 = pred4.reshape(-1,1)
pred5 = pred5.reshape(-1,1)

#%%
forecasts = np.column_stack((pred1, pred2, pred3, pred4, pred5))
#%%
forecasts_sub = forecasts.T
# %%
forecasts_sub
# %%
from datetime import datetime
#%%
forecastdate = datetime(2023, 12, 20, 00, 00)
# %%

df_sub_dax_DeepQuant = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "DAX",
    "horizon": [str(i) + " day" for i in (1,2,5,6,7)],
    "q0.025": forecasts_sub[:,0],
    "q0.25": forecasts_sub[:,1],
    "q0.5": forecasts_sub[:,2],
    "q0.75": forecasts_sub[:,3],
    "q0.975": forecasts_sub[:,4]})
df_sub_dax_DeepQuant
#%%
# %%

'''
--------------------------------------------------------
DEEP QUANTILE REGRESSION WITH MORE EXPLANATORY VARAIBELS
(no rollwing window ansatz)
-------------------------------------------------------
'''
#%%
'''
---------
=> Deep Quantile Regression in Keras
For h = 1
Quantile Predictions for the log returns ret4
---------
'''

X = sm.add_constant(hist[['Rt', 'price_up', 'ret1_up']].abs().iloc[:-1])
Y = hist['ret1'].shift(-1).iloc[:-1]  # Rt+1:t+1
perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]
model1 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])
model1.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))
model1.fit(X, Y, epochs=100, verbose=0)
# %%
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret1'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret1_up': recent_return_up}, index=[0])

# Vorhersagen für die Quantile machen
predictions1 = model1.predict(new_data)

# Ergebnisse in einem DataFrame speichern
pred_quant1 = pd.DataFrame(index=quantiles, columns=['predicted_ret1'])

pred_quant1['predicted_ret1'] = predictions1[0]  # Index 0 für die Vorhersagen für 'ret1'
#%%
print(pred_quant1)
#%%

'''
---------
=> Deep Quantile Regression in Keras
For h = 2
Quantile Predictions for the log returns ret4
---------
'''

X = sm.add_constant(hist[['Rt', 'price_up', 'ret2_up']].abs().iloc[:-1])
Y = hist['ret2'].shift(-1).iloc[:-1]  # Rt+1:t+1
perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]
model2 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])
model2.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))
model2.fit(X, Y, epochs=100, verbose=0)
# %%
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret2'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret2_up': recent_return_up}, index=[0])

# Vorhersagen für die Quantile machen
predictions2 = model2.predict(new_data)

# Ergebnisse in einem DataFrame speichern
pred_quant2 = pd.DataFrame(index=quantiles, columns=['predicted_ret2'])

pred_quant2['predicted_ret2'] = predictions2[0]  # Index 0 für die Vorhersagen für 'ret1'
#%%
print(pred_quant2)
#%%

'''
---------
=> Deep Quantile Regression in Keras
For h = 3
Quantile Predictions for the log returns ret4
---------
'''

X = sm.add_constant(hist[['Rt', 'price_up', 'ret3_up']].abs().iloc[:-1])
Y = hist['ret3'].shift(-1).iloc[:-1]  # Rt+1:t+1
perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]
model3 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])
model3.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))
model3.fit(X, Y, epochs=100, verbose=0)
# %%
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret3'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret3_up': recent_return_up}, index=[0])

# Vorhersagen für die Quantile machen
predictions3 = model3.predict(new_data)

# Ergebnisse in einem DataFrame speichern
pred_quant3 = pd.DataFrame(index=quantiles, columns=['predicted_ret3'])

pred_quant3['predicted_ret3'] = predictions3[0]  # Index 0 für die Vorhersagen für 'ret1'
#%%
print(pred_quant3)

#%%

'''
---------
=> Deep Quantile Regression in Keras
For h = 4
Quantile Predictions for the log returns ret4
---------
'''

X = sm.add_constant(hist[['Rt', 'price_up', 'ret4_up']].abs().iloc[:-1])
Y = hist['ret4'].shift(-1).iloc[:-1]  # Rt+1:t+1
perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]
model4 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])
model4.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))
model4.fit(X, Y, epochs=100, verbose=0)
# %%
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret4'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret4_up': recent_return_up}, index=[0])

# Vorhersagen für die Quantile machen
predictions4 = model4.predict(new_data)

# Ergebnisse in einem DataFrame speichern
pred_quant4 = pd.DataFrame(index=quantiles, columns=['predicted_ret4'])

pred_quant4['predicted_ret4'] = predictions4[0]  # Index 0 für die Vorhersagen für 'ret1'
#%%
print(pred_quant4)


#%%
'''
---------
=> Deep Quantile Regression in Keras
For h = 5
Quantile Predictions for the log returns ret4
---------
'''
X = sm.add_constant(hist[['Rt', 'price_up', 'ret5_up']].abs().iloc[:-1])
Y = hist['ret5'].shift(-1).iloc[:-1]  # Rt+1:t+1
perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]
model5 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])
model5.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))
model5.fit(X, Y, epochs=100, verbose=0)
# %%
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret5'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret5_up': recent_return_up}, index=[0])

# Vorhersagen für die Quantile machen
predictions5 = model5.predict(new_data)

# Ergebnisse in einem DataFrame speichern
pred_quant5 = pd.DataFrame(index=quantiles, columns=['predicted_ret5'])

pred_quant5['predicted_ret5'] = predictions5[0]  # Index 0 für die Vorhersagen für 'ret1'
#%%
print(pred_quant5)
# %%
pred_1 = pred_quant1['predicted_ret1'].values
pred_2 = pred_quant2['predicted_ret2'].values
pred_3 = pred_quant3['predicted_ret3'].values
pred_4 = pred_quant4['predicted_ret4'].values
pred_5 = pred_quant5['predicted_ret5'].values
# %%
pred__1 = pred_1.reshape(-1,1)
pred__2 = pred_2.reshape(-1,1)
pred__3 = pred_3.reshape(-1,1)
pred__4 = pred_4.reshape(-1,1)
pred__5 = pred_5.reshape(-1,1)

#%%
forecasts = np.column_stack((pred__1, pred__2, pred__3, pred__4, pred__5))
#%%
forecasts_sub_deep_with_more = forecasts.T
# %%
forecasts_sub_deep_with_more
# %%
from datetime import datetime
#%%
forecastdate = datetime(2023, 12, 20, 00, 00)
# %%

df_sub_dax_More_DeepQuant = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "DAX",
    "horizon": [str(i) + " day" for i in (1,2,5,6,7)],
    "q0.025": forecasts_sub_deep_with_more[:,0],
    "q0.25": forecasts_sub_deep_with_more[:,1],
    "q0.5": forecasts_sub_deep_with_more[:,2],
    "q0.75": forecasts_sub_deep_with_more[:,3],
    "q0.975": forecasts_sub_deep_with_more[:,4]})
df_sub_dax_More_DeepQuant

 #%%
PATH = "/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/PTSFC Submission JahrMonatTag_Rachel"

df_sub_dax_More_DeepQuant.to_csv(PATH+ "DeepQuantile_with_more_Variables_DAX_Abgabe6.csv", index=False)

#%%
''' 
------------------------------------
The Deep QUantile Regression Model with the additional variables, creates inplausible results
Therefore need to check with validation data set
-----------------------------------
'''
from sklearn.model_selection import train_test_split

#%%
'''
-----
MODEL 1
-----
Grid-Search for the Rolling Window Hyperparamater
-----
'''
window_sizes = [10, 20, 30]
step_sizes = [5, 10, 15]
X1 = sm.add_constant(hist[['Rt', 'price_up', 'ret1_up']].abs().iloc[:-1])
Y1 = hist['ret1'].shift(-1).iloc[:-1]  # Rt+1:t+1
for window_size in window_sizes:
    for step_size in step_sizes:
        # Trainingsdaten und Validierungsdaten erstellen
        X1_train, X1_val, y1_train, y1_val = train_test_split(X1, Y1, test_size=0.2, shuffle=False)

        # Modellstruktur
        model_1 = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(5)
        ])

        model_1.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

        # Trainings- und Validierungsverluste überwachen
        train_losses, val_losses = [], []
        for i in range(0, len(X1_train) - window_size, step_size):
            X_window = X1_train.iloc[i:i+window_size]
            y_window = y1_train.iloc[i:i+window_size]

            # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
            X_train, X_val, y_train, y_val = train_test_split(X_window, y_window, test_size=0.2, shuffle=False)

            history = model_1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

            # Erfasse den Trainings- und Validierungsverlust
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Ausgabe der Ergebnisse für die aktuelle Hyperparameter-Kombination
        print(f"Window Size: {window_size}, Step Size: {step_size}")
        print(f"Final Training Loss: {train_losses[-1]}, Final Validation Loss: {val_losses[-1]}")
        print("=" * 50)
#%%

'''
------
Model 1
=> return 1
=> mit dem Rolling Window Ansatz für das Training des Models
------
'''
X1 = sm.add_constant(hist[['Rt', 'price_up', 'ret1_up']].abs().iloc[:-1])
Y1 = hist['ret1'].shift(-1).iloc[:-1]  # Rt+1:t+1

perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]

model_1 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])

model_1.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

# Hyperparameter für den Rolling Window
#window_size = 20
#step_size = 5

# Nach Grid Search
window_size = 20
step_size = 10

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.2, shuffle=False)
X1_test, X1_val, y1_test, y1_val = train_test_split(X1_test, y1_test, test_size=0.5, shuffle=False)

# Trainiere das Modell mit dem Rolling Window-Ansatz
for i in range(0, len(X1_train) - window_size, step_size):
    X_window = X1_train.iloc[i:i+window_size]
    y_window = y1_train.iloc[i:i+window_size]
    
    model_1.fit(X_window, y_window, epochs=10, verbose=0)


test_predictions1 = model_1.predict(X1_test)

val_predictions1 = model_1.predict(X1_val)

# Erstelle Vorhersagen für das neueste Datenfenster
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret1'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret1_up': recent_return_up}, index=[0])

# Mache Vorhersagen für die Quantile
predictions = model_1.predict(new_data)

# Ergebnisse in einem DataFrame speichern
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pred_quant1 = pd.DataFrame(index=quantiles, columns=['predicted_ret1'])

pred_quant1['predicted_ret1'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'

print(pred_quant1)

#%%
'''
Plotten um zu sehen Validierungs und Trainings Loss
'''

# Trainings- und Validierungsverluste überwachen
train_losses1, val_losses1 = [], []
for i in range(0, len(X1_train) - window_size, step_size):
    X1_window = X1_train.iloc[i:i+window_size]
    y1_window = y1_train.iloc[i:i+window_size]
    
    # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
    X_train, X_val, y_train, y_val = train_test_split(X1_window, y1_window, test_size=0.2, shuffle=False)

    history = model_1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

    # Erfasse den Trainings- und Validierungsverlust
    train_loss1 = history.history['loss'][-1]
    val_loss1 = history.history['val_loss'][-1]
    train_losses1.append(train_loss1)
    val_losses1.append(val_loss1)

# Plot des Trainings- und Validierungsverlusts über die Zeit
plt.plot(train_losses1, label='Training Loss', marker='o')
plt.plot(val_losses1, label='Validation Loss', marker='o')
plt.xlabel('Rolling Window Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Teste das Modell auf dem Testdatensatz
X1_test, y1_test = X1_val, y1_val  # Verwende den Validierungsdatensatz für Testzwecke
test_predictions = model_1.predict(X1_test)

# Visualisierung der Vorhersagen im Vergleich zu den tatsächlichen Werten
plt.plot(y1_test.index, y1_test, label='Actual Values', marker='o')
plt.plot(X1_test.index, test_predictions, label='Predictions', marker='o')
plt.xlabel('Time')
plt.ylabel('ret1')
plt.legend()
plt.show()




#%%
window_size = 20
step_size = 5

# Trainingsdaten und Validierungsdaten erstellen
X1_train, X1_val, y1_train, y1_val = train_test_split(X1, Y1, test_size=0.2, shuffle=False)

# Trainiere das Modell mit dem Rolling Window-Ansatz und überwache den Validierungsverlust
val_losses = []
for i in range(0, len(X1_train) - window_size, step_size):
    X_window = X1_train.iloc[i:i+window_size]
    y_window = y1_train.iloc[i:i+window_size]
    
    # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
    X_train, X_val, y_train, y_val = train_test_split(X_window, y_window, test_size=0.2, shuffle=False)

    history = model_1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

    # Erfasse den Validierungsverlust
    val_loss = history.history['val_loss'][-1]
    val_losses.append(val_loss)

# Plot des Validierungsverlusts über die Zeit
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Rolling Window Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Teste das Modell auf dem Testdatensatz
X1_test, y1_test = X1_val, y1_val  # Verwende den Validierungsdatensatz für Testzwecke
test_predictions = model_1.predict(X1_test)

# Visualisierung der Vorhersagen im Vergleich zu den tatsächlichen Werten
plt.plot(y1_test.index, y1_test, label='Actual Values', marker='o')
plt.plot(X1_test.index, test_predictions, label='Predictions', marker='o')
plt.xlabel('Time')
plt.ylabel('ret1')
plt.legend()
plt.show()
#%%
'''
------
MODEL 2
-----
Grid-Search for the Rolling Window Hyperparamater
-----
'''
window_sizes = [10, 20, 30]
step_sizes = [5, 10, 15]
X2 = sm.add_constant(hist[['Rt', 'price_up', 'ret2_up']].abs().iloc[:-1])
Y2 = hist['ret2'].shift(-1).iloc[:-1]  # Rt+1:t+1
for window_size in window_sizes:
    for step_size in step_sizes:
        # Trainingsdaten und Validierungsdaten erstellen
        X2_train, X2_val, y2_train, y2_val = train_test_split(X2, Y2, test_size=0.2, shuffle=False)

        # Modellstruktur
        model_2 = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(5)
        ])

        model_2.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

        # Trainings- und Validierungsverluste überwachen
        train_losses, val_losses = [], []
        for i in range(0, len(X2_train) - window_size, step_size):
            X_window = X2_train.iloc[i:i+window_size]
            y_window = y2_train.iloc[i:i+window_size]

            # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
            X_train, X_val, y_train, y_val = train_test_split(X_window, y_window, test_size=0.2, shuffle=False)

            history = model_2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

            # Erfasse den Trainings- und Validierungsverlust
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Ausgabe der Ergebnisse für die aktuelle Hyperparameter-Kombination
        print(f"Window Size: {window_size}, Step Size: {step_size}")
        print(f"Final Training Loss: {train_losses[-1]}, Final Validation Loss: {val_losses[-1]}")
        print("=" * 50)


#%%
'''
------
Model 2
=> return 2
=> mit dem Rolling Window Ansatz
------
'''
X2 = sm.add_constant(hist[['Rt', 'price_up', 'ret2_up']].abs().iloc[:-1])
Y2 = hist['ret2'].shift(-1).iloc[:-1]  # Rt+1:t+1

perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]

model_2 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])

model_2.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

# Hyperparameter für den Rolling Window
#window_size = 20
#step_size = 5

# Nach Grid Search
window_size = 20
step_size = 10

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size=0.2, shuffle=False)
X2_test, X2_val, y2_test, y2_val = train_test_split(X2_test, y2_test, test_size=0.5, shuffle=False)

# Trainiere das Modell mit dem Rolling Window-Ansatz
for i in range(0, len(X2_train) - window_size, step_size):
    X_window = X2_train.iloc[i:i+window_size]
    y_window = y2_train.iloc[i:i+window_size]
    
    model_2.fit(X_window, y_window, epochs=10, verbose=0)


test_predictions2 = model_2.predict(X2_test)

val_predictions2 = model_2.predict(X2_val)

# Erstelle Vorhersagen für das neueste Datenfenster
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret2'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret2_up': recent_return_up}, index=[0])

# Mache Vorhersagen für die Quantile
predictions = model_2.predict(new_data)

# Ergebnisse in einem DataFrame speichern
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pred_quant2 = pd.DataFrame(index=quantiles, columns=['predicted_ret2'])

pred_quant2['predicted_ret2'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'

print(pred_quant2)

#%%

'''
Plotten um zu sehen Validierungs und Trainings Loss
'''

# Trainings- und Validierungsverluste überwachen
train_losses2, val_losses2 = [], []
for i in range(0, len(X2_train) - window_size, step_size):
    X2_window = X2_train.iloc[i:i+window_size]
    y2_window = y2_train.iloc[i:i+window_size]
    
    # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
    X_train, X_val, y_train, y_val = train_test_split(X2_window, y2_window, test_size=0.2, shuffle=False)

    history = model_2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

    # Erfasse den Trainings- und Validierungsverlust
    train_loss2 = history.history['loss'][-1]
    val_loss2 = history.history['val_loss'][-1]
    train_losses2.append(train_loss2)
    val_losses2.append(val_loss2)

# Plot des Trainings- und Validierungsverlusts über die Zeit
plt.plot(train_losses2, label='Training Loss', marker='o')
plt.plot(val_losses2, label='Validation Loss', marker='o')
plt.xlabel('Rolling Window Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Teste das Modell auf dem Testdatensatz
X2_test, y2_test = X2_val, y2_val  # Verwende den Validierungsdatensatz für Testzwecke
test_predictions = model_2.predict(X2_test)

# Visualisierung der Vorhersagen im Vergleich zu den tatsächlichen Werten
plt.plot(y2_test.index, y2_test, label='Actual Values', marker='o')
plt.plot(X2_test.index, test_predictions, label='Quantile Predictions', marker='o')
plt.xlabel('Time')
plt.ylabel('ret2')
plt.legend()
plt.show()

#%%

'''
-----
MODEL 3
-----
Grid-Search for the Rolling Window Hyperparamater
-----
'''
window_sizes = [10, 20, 30]
step_sizes = [5, 10, 15]
X3 = sm.add_constant(hist[['Rt', 'price_up', 'ret3_up']].abs().iloc[:-1])
Y3 = hist['ret3'].shift(-1).iloc[:-1]  # Rt+1:t+1
for window_size in window_sizes:
    for step_size in step_sizes:
        # Trainingsdaten und Validierungsdaten erstellen
        X3_train, X3_val, y3_train, y3_val = train_test_split(X3, Y3, test_size=0.2, shuffle=False)

        # Modellstruktur
        model_3 = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(5)
        ])

        model_3.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

        # Trainings- und Validierungsverluste überwachen
        train_losses, val_losses = [], []
        for i in range(0, len(X3_train) - window_size, step_size):
            X_window = X3_train.iloc[i:i+window_size]
            y_window = y3_train.iloc[i:i+window_size]

            # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
            X_train, X_val, y_train, y_val = train_test_split(X_window, y_window, test_size=0.2, shuffle=False)

            history = model_3.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

            # Erfasse den Trainings- und Validierungsverlust
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Ausgabe der Ergebnisse für die aktuelle Hyperparameter-Kombination
        print(f"Window Size: {window_size}, Step Size: {step_size}")
        print(f"Final Training Loss: {train_losses[-1]}, Final Validation Loss: {val_losses[-1]}")
        print("=" * 50)



#%%
'''
------
Model 3
=> return 3
=> mit dem Rolling Window Ansatz
------
'''
X3 = sm.add_constant(hist[['Rt', 'price_up', 'ret3_up']].abs().iloc[:-1])
Y3 = hist['ret3'].shift(-1).iloc[:-1]  # Rt+1:t+1

perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]

model_3 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])

model_3.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

# Hyperparameter für den Rolling Window
#window_size = 20
#step_size = 5

# Nach Grid Search
window_size = 30
step_size = 5

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, Y3, test_size=0.2, shuffle=False)
X3_test, X3_val, y3_test, y3_val = train_test_split(X3_test, y3_test, test_size=0.5, shuffle=False)

# Trainiere das Modell mit dem Rolling Window-Ansatz
for i in range(0, len(X3_train) - window_size, step_size):
    X_window = X3_train.iloc[i:i+window_size]
    y_window = y3_train.iloc[i:i+window_size]
    
    model_3.fit(X_window, y_window, epochs=10, verbose=0)


test_predictions3 = model_3.predict(X3_test)

val_predictions3 = model_3.predict(X3_val)

# Erstelle Vorhersagen für das neueste Datenfenster
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret3'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret3_up': recent_return_up}, index=[0])

# Mache Vorhersagen für die Quantile
predictions = model_3.predict(new_data)

# Ergebnisse in einem DataFrame speichern
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pred_quant3 = pd.DataFrame(index=quantiles, columns=['predicted_ret3'])

pred_quant3['predicted_ret3'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'

print(pred_quant3)

#%%

'''
-----
MODEL 4
-----
Grid-Search for the Rolling Window Hyperparamater
-----
'''
window_sizes = [10, 20, 30]
step_sizes = [5, 10, 15]
X4 = sm.add_constant(hist[['Rt', 'price_up', 'ret4_up']].abs().iloc[:-1])
Y4 = hist['ret4'].shift(-1).iloc[:-1]  # Rt+1:t+1
for window_size in window_sizes:
    for step_size in step_sizes:
        # Trainingsdaten und Validierungsdaten erstellen
        X4_train, X4_val, y4_train, y4_val = train_test_split(X4, Y4, test_size=0.2, shuffle=False)

        # Modellstruktur
        model_4 = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(5)
        ])

        model_4.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

        # Trainings- und Validierungsverluste überwachen
        train_losses, val_losses = [], []
        for i in range(0, len(X4_train) - window_size, step_size):
            X_window = X4_train.iloc[i:i+window_size]
            y_window = y4_train.iloc[i:i+window_size]

            # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
            X_train, X_val, y_train, y_val = train_test_split(X_window, y_window, test_size=0.2, shuffle=False)

            history = model_4.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

            # Erfasse den Trainings- und Validierungsverlust
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Ausgabe der Ergebnisse für die aktuelle Hyperparameter-Kombination
        print(f"Window Size: {window_size}, Step Size: {step_size}")
        print(f"Final Training Loss: {train_losses[-1]}, Final Validation Loss: {val_losses[-1]}")
        print("=" * 50)



#%%
'''
------
Model 4
=> return 4
=> mit dem Rolling Window Ansatz
------
'''
X4 = sm.add_constant(hist[['Rt', 'price_up', 'ret4_up']].abs().iloc[:-1])
Y4 = hist['ret4'].shift(-1).iloc[:-1]  # Rt+1:t+1

perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]

model_4 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])

model_4.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

# Hyperparameter für den Rolling Window
#window_size = 20
#step_size = 5

# Nach Grid Search
window_size = 20
step_size = 5


X4_train, X4_test, y4_train, y4_test = train_test_split(X4, Y4, test_size=0.2, shuffle=False)
X4_test, X4_val, y4_test, y4_val = train_test_split(X4_test, y4_test, test_size=0.5, shuffle=False)

# Trainiere das Modell mit dem Rolling Window-Ansatz
for i in range(0, len(X4_train) - window_size, step_size):
    X_window = X4_train.iloc[i:i+window_size]
    y_window = y4_train.iloc[i:i+window_size]
    
    model_4.fit(X_window, y_window, epochs=10, verbose=0)


test_predictions4 = model_4.predict(X1_test)

val_predictions4 = model_4.predict(X1_val)

# Erstelle Vorhersagen für das neueste Datenfenster
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret4'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret4_up': recent_return_up}, index=[0])

# Mache Vorhersagen für die Quantile
predictions = model_4.predict(new_data)

# Ergebnisse in einem DataFrame speichern
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pred_quant4 = pd.DataFrame(index=quantiles, columns=['predicted_ret4'])

pred_quant4['predicted_ret4'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'

print(pred_quant4)

#%%

'''
-----
MODEL 5
-----
Grid-Search for the Rolling Window Hyperparamater
-----
'''
window_sizes = [10, 20, 30]
step_sizes = [5, 10, 15]
X5 = sm.add_constant(hist[['Rt', 'price_up', 'ret5_up']].abs().iloc[:-1])
Y5 = hist['ret5'].shift(-1).iloc[:-1]  # Rt+1:t+1
for window_size in window_sizes:
    for step_size in step_sizes:
        # Trainingsdaten und Validierungsdaten erstellen
        X5_train, X5_val, y5_train, y5_val = train_test_split(X5, Y5, test_size=0.2, shuffle=False)

        # Modellstruktur
        model_5 = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(5)
        ])

        model_5.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

        # Trainings- und Validierungsverluste überwachen
        train_losses, val_losses = [], []
        for i in range(0, len(X5_train) - window_size, step_size):
            X_window = X5_train.iloc[i:i+window_size]
            y_window = y5_train.iloc[i:i+window_size]

            # Teile den Trainingsdatensatz in Trainings- und Validierungssätze auf
            X_train, X_val, y_train, y_val = train_test_split(X_window, y_window, test_size=0.2, shuffle=False,  random_state = 42)

            history = model_5.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

            # Erfasse den Trainings- und Validierungsverlust
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Ausgabe der Ergebnisse für die aktuelle Hyperparameter-Kombination
        print(f"Window Size: {window_size}, Step Size: {step_size}")
        print(f"Final Training Loss: {train_losses[-1]}, Final Validation Loss: {val_losses[-1]}")
        print("=" * 50)



#%%
'''
------
Model 5
=> return 5
=> mit dem Rolling Window Ansatz
------
'''
X5 = sm.add_constant(hist[['Rt', 'price_up', 'ret5_up']].abs().iloc[:-1])
Y5 = hist['ret5'].shift(-1).iloc[:-1]  # Rt+1:t+1

perc_points = [0.025, 0.25, 0.5, 0.75, 0.975]

model_5 = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])

model_5.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))

# Hyperparameter für den Rolling Window
#window_size = 20
#step_size = 5

# Nach Grid Search
window_size = 10
step_size = 15

X5_train, X5_test, y5_train, y5_test = train_test_split(X5, Y5, test_size=0.2, shuffle=False)
X5_test, X5_val, y5_test, y5_val = train_test_split(X5_test, y5_test, test_size=0.5, shuffle=False)

# Trainiere das Modell mit dem Rolling Window-Ansatz
for i in range(0, len(X5_train) - window_size, step_size):
    X_window = X5_train.iloc[i:i+window_size]
    y_window = y5_train.iloc[i:i+window_size]
    
    model_5.fit(X_window, y_window, epochs=10, verbose=0)


test_predictions5 = model_5.predict(X5_test)

val_predictions5 = model_5.predict(X5_val)

# Erstelle Vorhersagen für das neueste Datenfenster
recent_Rt = hist['Rt'].iloc[-1]
recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
recent_return_up = 1 if hist['ret5'].iloc[-1] > 0 else 0

new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret5_up': recent_return_up}, index=[0])

# Mache Vorhersagen für die Quantile
predictions = model_5.predict(new_data)

# Ergebnisse in einem DataFrame speichern
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pred_quant5 = pd.DataFrame(index=quantiles, columns=['predicted_ret5'])

pred_quant5['predicted_ret5'] = predictions[0]  # Index 0 für die Vorhersagen für 'ret1'

print(pred_quant5)

#%%
forecast_1 = pred_quant1['predicted_ret1'].values
forecast_2 = pred_quant2['predicted_ret2'].values
forecast_3 = pred_quant3['predicted_ret3'].values
forecast_4 = pred_quant4['predicted_ret4'].values
forecast_5 = pred_quant5['predicted_ret5'].values
#%%
fore__1 = forecast_1.reshape(-1,1)
fore__2 = forecast_2.reshape(-1,1)
fore__3 = forecast_3.reshape(-1,1)
fore__4 = forecast_4.reshape(-1,1)
fore__5 = forecast_5.reshape(-1,1)
#%%
forecasts = np.column_stack((fore__1, fore__2, fore__3, fore__4, fore__5))
#%%
forecasts_rolling_window = forecasts.T
# %%
forecasts_rolling_window
#%%
forecastdate = datetime(2023, 12, 20, 00, 00)
# %%

df_sub_dax_rolling_window = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "DAX",
    "horizon": [str(i) + " day" for i in (1,2,5,6,7)],
    "q0.025": forecasts_rolling_window[:,0],
    "q0.25": forecasts_rolling_window[:,1],
    "q0.5": forecasts_rolling_window[:,2],
    "q0.75": forecasts_rolling_window[:,3],
    "q0.975": forecasts_rolling_window[:,4]})
df_sub_dax_rolling_window
#%%
'''
-----------
Evaluation
-----------
'''

#%%
import yfinance as yf
dax_ticker = yf.Ticker("^GDAXI")
dax_data = dax_ticker.history(period ="max")

#%%
dax_data = dax_data.reset_index()
#%%
#dax_data['date'] = dax_data['Date']
#%%
for i in range(5):
    dax_data["ret"+str(i+1)] = compute_return(dax_data["Close"].values, h=i+1)


#%%
del dax_data['Open']
del dax_data['High']
del dax_data['Low']
del dax_data['Volume']
del dax_data['Dividends']
#%%
del dax_data['Stock Splits']
# %%
dax_data.set_index('Date', inplace = True)
#%%
#del dax_data['index']
#%%
dax_data.index = pd.to_datetime(dax_data.index)
#%%
dax_data.index = dax_data.index.date
#%%
dax_data
# %%
forecast_date = '2023-11-14'

# %%
forecast_date = pd.to_datetime(forecast_date)

# Extrahiere die Daten für die nächsten 5 Tage ab dem 'forecast_date'
next_5_days = dax_data.loc[forecast_date:forecast_date + pd.DateOffset(days=6)]

# Ergebnisse anzeigen
next_5_days
# %%
'''
-----------------------
Quantile Score Funktion
-----------------------
'''
def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        # Fall, in dem q_hat gleich y ist (Randfall)
        return 0
# %%
tau_values = [0.025, 0.25, 0.5, 0.75, 0.975]
#%%
# Ergebnisse in einem neuen DataFrame speichern

#quantile_results = pd.DataFrame()
#%%
true_return_1 = next_5_days['ret1'].iloc[0]
#%%
true_return_2 = next_5_days['ret2'].iloc[1]
# %%
true_return_3 = next_5_days['ret3'].iloc[2]
# %%
true_return_4 = next_5_days['ret4'].iloc[3]
# %%
true_return_5 = next_5_days['ret5'].iloc[4]
#%%
quant1_hat = df_sub_dax_DeepQuant['q0.025']
quant2_hat = df_sub_dax_DeepQuant['q0.25']
quant3_hat = df_sub_dax_DeepQuant['q0.5']
quant4_hat = df_sub_dax_DeepQuant['q0.75']
quant5_hat = df_sub_dax_DeepQuant['q0.975']
#%%
#%%
#quant1_hat = df_sub_dax_More_DeepQuant['q0.025']
#quant2_hat = df_sub_dax_More_DeepQuant['q0.25']
#quant3_hat = df_sub_dax_More_DeepQuant['q0.5']
#quant4_hat = df_sub_dax_More_DeepQuant['q0.75']
#quant5_hat = df_sub_dax_More_DeepQuant['q0.975']
#%%
#quant1_hat = df_sub_dax_rolling_window['q0.025']
#quant2_hat = df_sub_dax_rolling_window['q0.25']
#quant3_hat = df_sub_dax_rolling_window['q0.5']
#quant4_hat = df_sub_dax_rolling_window['q0.75']
#quant5_hat = df_sub_dax_rolling_window['q0.975']
#%%

quant1_hat = df_sub_dax_DeepQuant_without['q0.025']
quant2_hat = df_sub_dax_DeepQuant_without['q0.25']
quant3_hat = df_sub_dax_DeepQuant_without['q0.5']
quant4_hat = df_sub_dax_DeepQuant_without['q0.75']
quant5_hat = df_sub_dax_DeepQuant_without['q0.975']

# %%
tau_values = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
#%%
vectorized_quantile_score_ret1 = np.vectorize(quantile_score, otypes=[float])
scores_ret1 = vectorized_quantile_score_ret1(quant1_hat, true_return_1, tau_values)
print('Quantile Scores Quantile Regression for y = ret1 and the 5 quantiles')
print(scores_ret1)

average_score_ret1 = np.mean(scores_ret1)
average_score_ret1

#%%
vectorized_quantile_score_ret2 = np.vectorize(quantile_score, otypes=[float])
scores_ret2 = vectorized_quantile_score_ret2(quant2_hat, true_return_2, tau_values)
print('Quantile Scores Quantile Regression for y = ret2 and the 5 quantiles')
print(scores_ret2)

average_score_ret2 = np.mean(scores_ret2)
average_score_ret2
#%%                                             
vectorized_quantile_score_ret3 = np.vectorize(quantile_score, otypes=[float])
scores_ret3 = vectorized_quantile_score_ret3(quant3_hat, true_return_3, tau_values)
print('Quantile Scores Quantile Regression for y = ret3 and the 5 quantiles')
print(scores_ret3)

average_score_ret3 = np.mean(scores_ret3)
average_score_ret3
#%%
vectorized_quantile_score_ret4 = np.vectorize(quantile_score, otypes=[float])
scores_ret4 = vectorized_quantile_score_ret4(quant4_hat, true_return_4, tau_values)
print('Quantile Scores Quantile Regression for y = ret4 and the 5 quantiles')
print(scores_ret4)

average_score_ret4 = np.mean(scores_ret4)
average_score_ret4
#%%                                             
vectorized_quantile_score_ret5 = np.vectorize(quantile_score, otypes=[float])
scores_ret5 = vectorized_quantile_score_ret5(quant5_hat, true_return_5, tau_values)
print('Quantile Scores Quantile Regression for y = ret5 and the 5 quantiles')
print(scores_ret5)

average_score_ret5 = np.mean(scores_ret5)
average_score_ret5
#%%































#%%



























# %%
