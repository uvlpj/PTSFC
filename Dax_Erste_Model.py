#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow as tf

#%%
np.random.seed(42)
tf.random.set_seed(42)

# %%
df = pd.read_csv("/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Abagbe3_^GDAXI.csv")
#%%
df
# %%
df.index
#%%
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
# %%
df.plot.line(y ="Close", use_index = True )
# %%
# Removing the columns we dont need
del df['Volume']
del df['Adj Close']
# %%
df

# %%
# Prepare the data
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))
prediction_days = 60

# %%
x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x,0])


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
# %%
model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units=1 ))
#model.add(Dropout(0.2))

model.compile(optimizer= 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 10, batch_size = 32)

#%%
'''Test The Model Accurarcy on Existing Date'''
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

#%%
test_data = df.loc[test_start:test_end]
actual_prices = test_data['Close'].values

#%%
total_dataset = pd.concat((df['Close'], test_data['Close']), axis = 0)
#%%
model_inputs = total_dataset[len(total_dataset)- len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
# %%
model_inputs = scaler.transform(model_inputs)
# %%
# make Predictions on test data
x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#%%
predicted_price = model.predict(x_test)
#%%

predicted_price = scaler.inverse_transform(predicted_price)
#%%

#%%
# Plot the predictions
plt.plot(actual_prices, color = "black")
plt.plot(predicted_price, color = "green")
plt.xlabel('Time')
plt.ylabel('Closing stock Dax')
# %%
# Predicting the next day

real_data = [model_inputs[len(model_inputs)+1 -prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(prediction)

# %%
'''
Vorhersagen für die nächsten 5 Tage machen
'''

t = 5

# Initialisierung der realen Daten mit den letzten prediction_days Werten
real_data = model_inputs[-prediction_days:]

# Schleife für die Vorhersagen
vorhersagen_lstm = []
for i in range(t):
    # Reshape der Daten für das LSTM-Modell
    real_data_reshaped = np.reshape(real_data, (1, prediction_days, 1))

    # Vorhersage für den nächsten Tag
    prediction = model.predict(real_data_reshaped)
    prediction = scaler.inverse_transform(prediction)

    # Hinzufügen der Vorhersage zum realen Datenarray
    real_data = np.append(real_data[1:], prediction[0])

    # Hinzufügen der Vorhersage zum Vorhersagen-Array
    vorhersagen_lstm.append(prediction[0, 0])

# Ausgabe der Vorhersagen für die nächsten t Tage
print("Vorhersagen für die nächsten", t, "Tage:")
print(vorhersagen_lstm)


#%%

'''
Die Vorhersagen Plotten
[14769.058, 15320.064, 16117.806, 16971.191, 17771.592]
'''

import matplotlib.pyplot as plt

# Plot der tatsächlichen Preise
plt.plot(actual_prices, color="black", label="Actual Prices")

# Plot der vorhergesagten Preise
plt.plot(predicted_price[:-5], color="green", label="Predicted Prices")

# Plot der nächsten 5 Tage in Orange
plt.plot(range(len(predicted_price), len(predicted_price) + 5), vorhersagen_lstm, color="orange", label="Next 5 Days Prediction")

# Legende hinzufügen
plt.legend()

# Achsentitel hinzufügen
plt.xlabel('Time')
plt.ylabel('Closing stock Dax')

# Diagramm anzeigen
plt.show()
#%%

'''
Die Vorhersagen zum Datensatz hinzufügen um dann die Renditen später zu berechnen
'''
#%%
datensatz_mit_vorhersagen = df

#%%
last_date = datensatz_mit_vorhersagen.index[-1]
#%%
neue_daten_index = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=5)
#%%
df_vorhersagen = pd.DataFrame({'Close': vorhersagen_lstm}, index=neue_daten_index)
#%%
data_set_with_predictions = pd.concat([datensatz_mit_vorhersagen, df_vorhersagen])


#################################################################################################################
#%%

'''
Jetzt die Renditen berchnen
'''

def compute_return(y, r_type="log", h=1):
    
    # exclude first h observations
    y2 = y[h:] # Schließt die erste h Beobachtungen von y aus und gibt den Rest zurück
    # exclude last h observations
    y1 = y[:-h] # Schließt die letzten h Beobachtungen von y aus und gibt den Rest zurück
    
    if r_type == "log":
        ret = np.concatenate(([np.nan]*h, 100 * (np.log(y2) - np.log(y1))))
    else:
        ret = np.concatenate(([np.nan]*h, 100 * (y2-y1)/y1))
        
    return ret

# %%
hist = data_set_with_predictions
#%%
for i in range(5):
    hist["ret"+str(i+1)] = compute_return(hist["Close"].values, h=i+1)

#%%
print(hist)

#%%
letzte_5_zeilen = hist.tail(5).copy()

# Erstellen Sie einen separaten DataFrame für die letzten 5 Zeilen
datensatz_letzte_5 = pd.DataFrame(letzte_5_zeilen)

#%%
werte_der_spalten = letzte_5_zeilen.iloc[:, 4:9].values
# %%
from datetime import datetime, timedelta
forecastdate = datetime(2023, 11, 14, 00, 00)
# %%
df_sub = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "DAX",
    "horizon": [str(i) + " day" for i in (1,2,5,6,7)],
    "q0.025": werte_der_spalten[:,0],
    "q0.25": werte_der_spalten[:,1],
    "q0.5": werte_der_spalten[:,2],
    "q0.75": werte_der_spalten[:,3],
    "q0.975": werte_der_spalten[:,4]})
df_sub

#%%

PATH = "/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/PTSFC Submission JahrMonatTag_Rachel"

df_sub.to_csv(PATH+ "LSTM_Dax_Abgab3.csv", index=False)


# %%
