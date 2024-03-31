#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
#%%
np.random.seed(42)
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
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/Data DAX/1/Sub1.csv') # Sub1
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/Data DAX/2/Sub2.csv') # Sub2
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/Data DAX/3/Sub3.csv') # Sub3
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/Data DAX/4/Sub4.csv') # Sub4
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/Data DAX/5/Sub5.csv') # Sub5
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/Data DAX/6/Sub6.csv') # Sub6
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/Data DAX/7/Sub7.csv') # Sub7
hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/Data DAX/8/Sub8.csv') # Sub8
#%%
for i in range(5):
    hist["ret"+str(i+1)] = compute_return(hist["Close"].values, h=i+1)

#%%
hist = hist.dropna()    

#%%
#start = '2023-11-16'
#end = '2023-11-22'
#next_5_days = hist[(hist['Date'] >= start) & (hist['Date'] <= end)]
#hist = hist[hist['Date'] <= '2023-11-15']
#%%
#start = '2023-11-23'
#end = '2023-11-29'
#next_5_days = hist[(hist['Date'] >= start) & (hist['Date'] <= end)]
#hist = hist[hist['Date'] <= '2023-11-22']
#%%
#start = '2023-11-30'
#end = '2023-12-06'
#next_5_days = hist[(hist['Date'] >= start) & (hist['Date'] <= end)]
#hist = hist[hist['Date'] <= '2023-11-29']
#%%
#start = '2023-12-07'
#end = '2023-12-13'
#next_5_days = hist[(hist['Date'] >= start) & (hist['Date'] <= end)]
#hist = hist[hist['Date'] <= '2023-12-06']
#%%
#start = '2023-12-14'
#end = '2023-12-20'
#next_5_days = hist[(hist['Date'] >= start) & (hist['Date'] <= end)]
#hist = hist[hist['Date'] <= '2023-12-13']
#%%
#start = '2024-01-04'
#end = '2024-01-10'
#next_5_days = hist[(hist['Date'] >= start) & (hist['Date'] <= end)]
#hist = hist[hist['Date'] <= '2024-01-03']
#%%
#start = '2024-01-11'
#end = '2024-01-17'
#next_5_days = hist[(hist['Date'] >= start) & (hist['Date'] <= end)]
#hist = hist[hist['Date'] <= '2024-01-10']
#%%
#start = '2024-01-18'
#end = '2024-01-24'
#next_5_days = hist[(hist['Date'] >= start) & (hist['Date'] <= end)]
#hist = hist[hist['Date'] <= '2024-01-17']

#%%
'''
-----------------------------------------------------------------
Ein Neuronales Netz bauen
dass die Varianz und den Mean schätzt für den nächten Tag
=> dann können aus der geschätzen Varianz und dem Mean die Quantile berechnet werden unter der Annahme der Normalverteilung
----------------------------------------------------------------
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
#%%
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
#%%
'''
-----------------------------------------------------------------------
Ret1
----------------------------------------------------------------------
'''
data1 = hist['ret1']
data1 = pd.Series(data1)
data1
#%%
def create_input_output(data, sequence_length):
    x, y_mean, y_var = [], [], []
    for i in range(len(data1) - sequence_length):
        x.append(data1[i:i+sequence_length])
        y_mean.append(np.mean(data1[i+1:i+sequence_length+1]))
        y_var.append(np.var(data1[i+1:i+sequence_length+1]))
    return np.array(x), np.array(y_mean), np.array(y_var)

#%%

sequence_length = 15  # Sie können dies anpassen
units = 100
# %%
x, y_mean, y_var = create_input_output(data1, sequence_length)
# %%
#inputs = Input(shape=(sequence_length, 1))
#lstm_out = LSTM(units)(inputs)

inputs = Input(shape=(sequence_length,))
dense_layer = Dense(units, activation='relu')(inputs)
#%%
# Ausgabe für den geschätzten Mittelwert
#mean_output = Dense(1, name='mean')(lstm_out)

mean_output = Dense(1, name='mean', activation = 'linear')(dense_layer)
#%%
# Ausgabe für die geschätzte Varianz
variance_output = Dense(1, name='variance', activation = 'softplus')(dense_layer)
#%%
# Modell erstellen
model = Model(inputs=inputs, outputs=[mean_output, variance_output])
#%%
# Modell kompilieren
model.compile(optimizer='adam', loss={'mean': 'mse', 'variance': 'mse'})
#%%
# Modell trainieren
model.fit(x, {'mean': y_mean, 'variance': y_var}, epochs=10, batch_size=32)
#%%

new_data1 = np.array(hist['ret1'].iloc[-sequence_length:])
predicted_mean1, predicted_var1 = model.predict(new_data1.reshape((1, sequence_length)))

#%%
#predicted_mean, predicted_var = model.predict(x_new.reshape((1, sequence_length, 1)))

print("Geschätzter Mittelwert für den nächsten Tag:", predicted_mean1[0][0])
print("Geschätzte Varianz für den nächsten Tag:", predicted_var1[0][0])



#%%

'''
Quantile berechnen

'''
mu1 = predicted_mean1[0][0]
variance1 = predicted_var1[0][0]
alpha_values = [0.025, 0.25, 0.5, 0.75, 0.975]
sigma1 = (variance1)**0.5

quantiles1 = [mu1 + norm.ppf(alpha) * sigma1 for alpha in alpha_values]

quantiles_ret1 = pd.DataFrame([quantiles1])
quantiles_ret1.columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']
quantiles_ret1

#%%
from keras.layers import Input, Dense, Concatenate, Multiply, Add
from keras.models import Model
from keras.optimizers import RMSprop
import numpy as np



#%%
'''
------------------------------------------------------------------------
Return 1
=> MEAN VARIANCE MODEL WIE IM PAPER
-------------------------------------------------------------------
'''
data1 = hist['ret1']
data1 = pd.Series(data1)
data1
mean_return = data1.mean()
variance_return = data1.var()
# Eingabedaten vorbereiten
data1_df = pd.DataFrame(data1)
input_data1 = data1_df.values.reshape(-1, 1)

# Definieren der Eingabeschicht
input1 = Input(shape=(1,), dtype='float32', name='Input1')
input2 = Input(shape=(1,), dtype='float32', name='Input2')

# Liste der Eingaben
inputs = [input1, input2]

# Versteckte Schicht
concatenated_inputs = Concatenate()([input1, input2])
hidden = Dense(units=64, activation='tanh')(concatenated_inputs)

# Erwartung
expectation = Dense(units=32, activation='tanh')(hidden)
expectation = Dense(units=1, activation='linear', name='expectation')(expectation)

# Standardabweichung
std_deviation = Dense(units=32, activation='tanh')(hidden)
std_deviation = Dense(units=1, activation='softplus', name='std_deviation')(std_deviation)

# Varianz
variance = Multiply(name='variance')([std_deviation, std_deviation])

# Zweites Moment
second_moment = Add(name='second_moment')([variance, expectation ** 2])

# Modell erstellen
model = Model(inputs=inputs, outputs=[expectation, second_moment])

# Kompilieren des Modells
optimizer = RMSprop()  # Passen Sie den Optimierer entsprechend an
losses = {'expectation': 'mse', 'second_moment': 'mse'}
loss_weights = {'expectation': 0.5, 'second_moment': 0.5}  # Passen Sie die Gewichtung entsprechend an
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

# Modelzusammenfassung anzeigen
model.summary()

# Modell trainieren
# Modell trainieren
# Modell trainieren
label_mean = np.array([mean_return] * len(input_data1))
label_variance = np.array([variance_return] * len(input_data1))
model.fit([input_data1, input_data1], [label_mean, label_variance], epochs=10, batch_size=32, validation_split=0.2)

# Annahme: model ist Ihr trainiertes Modell

# Vorhersagen für den Mittelwert und das zweite Moment erhalten
predictions = model.predict([input_data1, input_data1])

# Mittelwert und Varianz extrahieren
mean_prediction = predictions[0]
variance_prediction = predictions[1]

# Anzeigen der Vorhersagen
print("Mean Prediction:", mean_prediction)
print("Variance Prediction:", variance_prediction)
#%%
new_value = data1.iloc[-1]

# Vorbereiten des neuen Werts für die Vorhersage
new_input_data = np.array([[new_value]])

# Vorhersagen für den Mittelwert und die Varianz machen
mean_prediction, variance_prediction = model.predict([new_input_data, new_input_data])

# Anzeigen der Vorhersagen
print("Mean Prediction:", mean_prediction)
print("Variance Prediction:", variance_prediction)
# %%

mu1 = mean_prediction
variance1 = variance_prediction
alpha_values = [0.025, 0.25, 0.5, 0.75, 0.975]
sigma1 = (variance1)**0.5

quantiles1 = [mu1 + norm.ppf(alpha) * sigma1 for alpha in alpha_values]

quantiles_ret1 = pd.DataFrame([quantiles1])
quantiles_ret1.columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']
quantiles_ret1




#%%
'''
------------------------------------------------------------------------
Return 2
=> MEAN VARIANCE MODEL WIE IM PAPER
-------------------------------------------------------------------
'''

data2 = hist['ret2']
data2 = pd.Series(data2)
data2
mean_return = data2.mean()
variance_return = data2.var()
# Eingabedaten vorbereiten
data2_df = pd.DataFrame(data2)
input_data2 = data2_df.values.reshape(-1, 1)

# Definieren der Eingabeschicht
input1 = Input(shape=(1,), dtype='float32', name='Input1')
input2 = Input(shape=(1,), dtype='float32', name='Input2')

# Liste der Eingaben
inputs = [input1, input2]

# Versteckte Schicht
concatenated_inputs = Concatenate()([input1, input2])
hidden = Dense(units=64, activation='tanh')(concatenated_inputs)

# Erwartung
expectation = Dense(units=32, activation='tanh')(hidden)
expectation = Dense(units=1, activation='linear', name='expectation')(expectation)

# Standardabweichung
std_deviation = Dense(units=32, activation='tanh')(hidden)
std_deviation = Dense(units=1, activation='softplus', name='std_deviation')(std_deviation)

# Varianz
variance = Multiply(name='variance')([std_deviation, std_deviation])

# Zweites Moment
second_moment = Add(name='second_moment')([variance, expectation ** 2])

# Modell erstellen
model = Model(inputs=inputs, outputs=[expectation, second_moment])

# Kompilieren des Modells
optimizer = RMSprop()  # Passen Sie den Optimierer entsprechend an
losses = {'expectation': 'mse', 'second_moment': 'mse'}
loss_weights = {'expectation': 0.5, 'second_moment': 0.5}  # Passen Sie die Gewichtung entsprechend an
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

# Modelzusammenfassung anzeigen
model.summary()

label_mean = np.array([mean_return] * len(input_data1))
label_variance = np.array([variance_return] * len(input_data1))
model.fit([input_data2, input_data2], [label_mean, label_variance], epochs=10, batch_size=32, validation_split=0.2)

# Annahme: model ist Ihr trainiertes Modell

# Vorhersagen für den Mittelwert und das zweite Moment erhalten
predictions = model.predict([input_data2, input_data2])

# Mittelwert und Varianz extrahieren
mean_prediction = predictions[0]
variance_prediction = predictions[1]

# Anzeigen der Vorhersagen
print("Mean Prediction:", mean_prediction)
print("Variance Prediction:", variance_prediction)
#%%
new_value = data2.iloc[-1]

# Vorbereiten des neuen Werts für die Vorhersage
new_input_data = np.array([[new_value]])

# Vorhersagen für den Mittelwert und die Varianz machen
mean_prediction, variance_prediction = model.predict([new_input_data, new_input_data])

# Anzeigen der Vorhersagen
print("Mean Prediction:", mean_prediction)
print("Variance Prediction:", variance_prediction)
#%%
new_value = data2.iloc[-1]

# Vorbereiten des neuen Werts für die Vorhersage
new_input_data = np.array([[new_value]])

# Vorhersagen für den Mittelwert und die Varianz machen
mean_prediction, variance_prediction = model.predict([new_input_data, new_input_data])

# Anzeigen der Vorhersagen
print("Mean Prediction:", mean_prediction)
print("Variance Prediction:", variance_prediction)
# %%

mu2 = mean_prediction
variance2 = variance_prediction
alpha_values = [0.025, 0.25, 0.5, 0.75, 0.975]
sigma2 = (variance2)**0.5

quantiles2 = [mu2 + norm.ppf(alpha) * sigma2 for alpha in alpha_values]

quantiles_ret2 = pd.DataFrame([quantiles2])
quantiles_ret2.columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']
quantiles_ret2



#%%
'''
------------------------------------------------------------------------
Return 3
=> MEAN VARIANCE MODEL WIE IM PAPER
-------------------------------------------------------------------
'''

data3 = hist['ret3']
data3 = pd.Series(data3)
data3
mean_return = data3.mean()
variance_return = data3.var()
# Eingabedaten vorbereiten
data3_df = pd.DataFrame(data3)
input_data3= data3_df.values.reshape(-1, 1)

# Definieren der Eingabeschicht
input1 = Input(shape=(1,), dtype='float32', name='Input1')
input2 = Input(shape=(1,), dtype='float32', name='Input2')

# Liste der Eingaben
inputs = [input1, input2]

# Versteckte Schicht
concatenated_inputs = Concatenate()([input1, input2])
hidden = Dense(units=64, activation='tanh')(concatenated_inputs)

# Erwartung
expectation = Dense(units=32, activation='tanh')(hidden)
expectation = Dense(units=1, activation='linear', name='expectation')(expectation)

# Standardabweichung
std_deviation = Dense(units=32, activation='tanh')(hidden)
std_deviation = Dense(units=1, activation='softplus', name='std_deviation')(std_deviation)

# Varianz
variance = Multiply(name='variance')([std_deviation, std_deviation])

# Zweites Moment
second_moment = Add(name='second_moment')([variance, expectation ** 2])

# Modell erstellen
model = Model(inputs=inputs, outputs=[expectation, second_moment])

# Kompilieren des Modells
optimizer = RMSprop()  # Passen Sie den Optimierer entsprechend an
losses = {'expectation': 'mse', 'second_moment': 'mse'}
loss_weights = {'expectation': 0.5, 'second_moment': 0.5}  # Passen Sie die Gewichtung entsprechend an
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

# Modelzusammenfassung anzeigen
#model.summary()

label_mean = np.array([mean_return] * len(input_data1))
label_variance = np.array([variance_return] * len(input_data1))
model.fit([input_data2, input_data2], [label_mean, label_variance], epochs=10, batch_size=32, validation_split=0.2)



# Vorhersagen für den Mittelwert und das zweite Moment erhalten
predictions = model.predict([input_data3, input_data3])

# Mittelwert und Varianz extrahieren
mean_prediction = predictions[0]
variance_prediction = predictions[1]

# Anzeigen der Vorhersagen
print("Mean Prediction:", mean_prediction)
print("Variance Prediction:", variance_prediction)
#%%
new_value = data3.iloc[-1]

# Vorbereiten des neuen Werts für die Vorhersage
new_input_data = np.array([[new_value]])

# Vorhersagen für den Mittelwert und die Varianz machen
mean_prediction, variance_prediction = model.predict([new_input_data, new_input_data])

# Anzeigen der Vorhersagen
print("Mean Prediction:", mean_prediction)
print("Variance Prediction:", variance_prediction)
#%%
new_value = data3.iloc[-1]

# Vorbereiten des neuen Werts für die Vorhersage
new_input_data = np.array([[new_value]])

# Vorhersagen für den Mittelwert und die Varianz machen
mean_prediction, variance_prediction = model.predict([new_input_data, new_input_data])

# Anzeigen der Vorhersagen
print("Mean Prediction:", mean_prediction)
print("Variance Prediction:", variance_prediction)
# %%

mu3 = mean_prediction
variance3 = variance_prediction
alpha_values = [0.025, 0.25, 0.5, 0.75, 0.975]
sigma3 = (variance2)**0.5

quantiles3 = [mu3 + norm.ppf(alpha) * sigma3 for alpha in alpha_values]

quantiles_ret3 = pd.DataFrame([quantiles3])
quantiles_ret3.columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']
quantiles_ret3



#%%
'''
------------------------------------------------------------------------
Return 4
=> MEAN VARIANCE MODEL WIE IM PAPER
-------------------------------------------------------------------
'''

data4 = hist['ret4']
data4= pd.Series(data4)
data4

mean_return = data4.mean()
variance_return = data4.var()
# Eingabedaten vorbereiten
data4_df = pd.DataFrame(data4)
input_data4 = data4_df.values.reshape(-1, 1)

# Definieren der Eingabeschicht
input1 = Input(shape=(1,), dtype='float32', name='Input1')
input2 = Input(shape=(1,), dtype='float32', name='Input2')

# Liste der Eingaben
inputs = [input1, input2]

# Versteckte Schicht
concatenated_inputs = Concatenate()([input1, input2])
hidden = Dense(units=64, activation='tanh')(concatenated_inputs)

# Erwartung
expectation = Dense(units=32, activation='tanh')(hidden)
expectation = Dense(units=1, activation='linear', name='expectation')(expectation)

# Standardabweichung
std_deviation = Dense(units=32, activation='tanh')(hidden)
std_deviation = Dense(units=1, activation='softplus', name='std_deviation')(std_deviation)

# Varianz
variance = Multiply(name='variance')([std_deviation, std_deviation])

# Zweites Moment
second_moment = Add(name='second_moment')([variance, expectation ** 2])

# Modell erstellen
model_4 = Model(inputs=inputs, outputs=[expectation, second_moment])

# Kompilieren des Modells
optimizer = RMSprop()  # Passen Sie den Optimierer entsprechend an
losses = {'expectation': 'mse', 'second_moment': 'mse'}
loss_weights = {'expectation': 0.5, 'second_moment': 0.5}  # Passen Sie die Gewichtung entsprechend an
model_4.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

# Modelzusammenfassung anzeigen
#model.summary()

label_mean = np.array([mean_return] * len(input_data4))
label_variance = np.array([variance_return] * len(input_data4))
model_4.fit([input_data4, input_data4], [label_mean, label_variance], epochs=10, batch_size=32, validation_split=0.2)



# Vorhersagen für den Mittelwert und das zweite Moment erhalten
predictions = model_4.predict([input_data4, input_data4])

# Mittelwert und Varianz extrahieren
mean_prediction = predictions[0]
variance_prediction = predictions[1]

# Anzeigen der Vorhersagen
print("Mean Prediction:", mean_prediction)
print("Variance Prediction:", variance_prediction)
#%%
new_value = data4.iloc[-1]

# Vorbereiten des neuen Werts für die Vorhersage
new_input_data = np.array([[new_value]])

# Vorhersagen für den Mittelwert und die Varianz machen
mean_prediction, variance_prediction = model.predict([new_input_data, new_input_data])

# Anzeigen der Vorhersagen
print("Mean Prediction:", mean_prediction)
print("Variance Prediction:", variance_prediction)
#%%
new_value = data4.iloc[-1]

# Vorbereiten des neuen Werts für die Vorhersage
new_input_data = np.array([[new_value]])

# Vorhersagen für den Mittelwert und die Varianz machen
mean_prediction, variance_prediction = model.predict([new_input_data, new_input_data])

# Anzeigen der Vorhersagen
print("Mean Prediction:", mean_prediction)
print("Variance Prediction:", variance_prediction)
# %%

mu4 = mean_prediction
variance4 = variance_prediction
alpha_values = [0.025, 0.25, 0.5, 0.75, 0.975]
sigma4 = (variance2)**0.5

quantiles4 = [mu4 + norm.ppf(alpha) * sigma4 for alpha in alpha_values]

quantiles_ret4 = pd.DataFrame([quantiles4])
quantiles_ret4.columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']
quantiles_ret4

#%%

'''
----------------
Return 5
--------------
'''
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from scipy.stats import norm

# Annahme: hist ist Ihr DataFrame mit den historischen Daten

# Datenvorbereitung für Return 5
data5 = hist['ret5']  # Daten für Return 5 aus hist extrahieren
data5 = pd.Series(data5)
mean_return = data5.mean()
variance_return = data5.var()

# Eingabedaten vorbereiten
data5_df = pd.DataFrame(data5)
input_data5 = data5_df.values.reshape(-1, 1)

# Definieren der Eingabeschicht
input1 = Input(shape=(1,), dtype='float32', name='Input1')
input2 = Input(shape=(1,), dtype='float32', name='Input2')

# Liste der Eingaben
inputs = [input1, input2]

# Versteckte Schicht
concatenated_inputs = Concatenate()([input1, input2])
hidden = Dense(units=64, activation='tanh')(concatenated_inputs)

# Erwartung
expectation = Dense(units=32, activation='tanh')(hidden)
expectation = Dense(units=1, activation='linear', name='expectation')(expectation)

# Standardabweichung
std_deviation = Dense(units=32, activation='tanh')(hidden)
std_deviation = Dense(units=1, activation='softplus', name='std_deviation')(std_deviation)

# Varianz
variance = Multiply(name='variance')([std_deviation, std_deviation])

# Zweites Moment
second_moment = Add(name='second_moment')([variance, expectation ** 2])

# Modell erstellen
model_5 = Model(inputs=inputs, outputs=[expectation, second_moment])

# weightes:
normalized_data5 = (data5 - data5.mean()) / data5.std()
mean_return_normalized = normalized_data5.mean()
variance_return_normalized = normalized_data5.var()

n = len(normalized_data5)
y_squared_sum = (normalized_data5 ** 2).sum()
w1 = 1 / (n * y_squared_sum)
w1 = w1 ** -1

y_squared_sum_w2 = np.sum(((normalized_data5 ** 2) - (1 / w1))**2)
w2 = 1 / (n * y_squared_sum_w2)
w2 = w2 ** -1


# Kompilieren des Modells
optimizer = RMSprop()  # Passen Sie den Optimierer entsprechend an
losses = {'expectation': 'mse', 'second_moment': 'mse'}
loss_weights = {'expectation': w1, 'second_moment': w2}  # Passen Sie die Gewichtung entsprechend an
model_5.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

# Modell trainieren
label_mean = np.array([mean_return] * len(input_data5))
label_variance = np.array([variance_return] * len(input_data5))
model_5.fit([input_data5, input_data5], [label_mean, label_variance], epochs=15)

# Annahme: model_5 ist Ihr trainiertes Modell

# Vorhersagen für den Mittelwert und das zweite Moment erhalten
predictions = model_5.predict([input_data5, input_data5])

# Mittelwert und Varianz extrahieren
mean_prediction = predictions[0]
variance_prediction = predictions[1]

# Anzeigen der Vorhersagen
print("Mean Prediction:", mean_prediction)
print("Variance Prediction:", variance_prediction)

# Neuen Wert vorbereiten und vorhersagen
new_value = data15iloc[-1]
new_input_data = np.array([[new_value]])
mean_prediction, variance_prediction = model_5.predict([new_input_data, new_input_data])
print("Mean Prediction for new value:", mean_prediction)
print("Variance Prediction for new value:", variance_prediction)

# Berechnung von Quantilen basierend auf den Vorhersagen
mu5 = mean_prediction
sigma5 = variance_prediction ** 0.5
alpha_values = [0.025, 0.25, 0.5, 0.75, 0.975]
quantiles5 = [mu5 + norm.ppf(alpha) * sigma5 for alpha in alpha_values]

# Erstellen eines DataFrames für die Quantile
quantiles_ret5 = pd.DataFrame([quantiles5], columns=['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975'])
print(quantiles_ret5)




# %%
q_returns = pd.concat([quantiles_ret1, quantiles_ret2, quantiles_ret3, quantiles_ret4, quantiles_ret5], axis = 0)
q_returns

#%%
import datetime as dt
#%%
forecastdate = dt.datetime(2024, 2, 7, 00, 00)
#%%
df_sub_dax_MVE = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "DAX",
    "horizon": [str(i) + " day" for i in (1,2,5,6,7)],
    "q0.025": q_returns['quantile_0.025'],
    "q0.25": q_returns['quantile_0.25'],
    "q0.5": q_returns['quantile_0.5'],
    "q0.75": q_returns['quantile_0.75'],
    "q0.975": q_returns['quantile_0.975']})
df_sub_dax_MVE 



PATH = "/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/PTSFC Submission JahrMonatTag_Rachel"

df_sub_dax_MVE.to_csv(PATH+ "MVE_DAX_Abgabe11.csv", index=False)
