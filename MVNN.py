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
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Abagbe3_^GDAXI.csv')
#data = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Abgab2^GDAXI.csv')
#data = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Abgabe_29_11_23^GDAXI.csv')
#data = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission5/^GDAXI Kopie.csv')
#data = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission6/^GDAXI.csv')
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission9/^GDAXI.csv')
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission10/^GDAXI.csv')
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission11/^GDAXI.csv')

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
start = '2024-01-18'
end = '2024-01-24'
next_5_days = hist[(hist['Date'] >= start) & (hist['Date'] <= end)]
hist = hist[hist['Date'] <= '2024-01-17']
#%%

#%%
#for i in range(5):
 #   data["ret"+str(i+1)] = compute_return(data["Close"].values, h=i+1)

#%%
for i in range(5):
    hist["ret"+str(i+1)] = compute_return(hist["Close"].values, h=i+1)    

#%%
'''Hat noch den letzen Datenpunkt vom Mittwoch welcher in hist2 nicht vorhanden ist'''
#last_row = hist2.tail(1).copy()

#%%
#frames = [data, last_row]
#hist = pd.concat(frames)

#%%
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as pl

#%%
hist = hist.dropna()

# %%
#del data['Open']
#del data['High']
#del data['Low']
#del data['Volume']
#del data['Adj Close']
# %%
hist
#%%

hist['Date'] = pd.to_datetime(hist['Date'])
#%%
hist_2023 = hist[hist['Date'].dt.year == 2023]
hist_2023



















#%%
hist = hist.dropna()
#%%
hist.set_index('Date', inplace = True)
# %%
hist
#%%
del hist['Open']
#%%
del hist['High']
#%%
del hist['Low']
#%%
del hist['Adj Close']
# %%
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# %%
X = hist[['ret1']].values
X = X.reshape((len(X),))
#y_mean = hist['ret1'].values
#y_var = np.var(hist[['ret1']], axis = 1)


#%%
import scipy.stats as stats
stats.probplot(X, dist="norm", plot=plt)
plt.title("QQ-Plot")
plt.show()

#%%
hist_2023 = hist[hist.index.year == 2023]

#%%
data_one_step = hist['ret1'].values
data_two_step = hist['ret2'].values
data_three_step = hist['ret3'].values
data_four_step = hist['ret4'].values
data_five_step = hist['ret5'].values

#%%
import pylab 
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt
from scipy import stats

#%%
'''Probability Plot ret1'''
stats.probplot(data_one_step, dist="norm", plot=pylab)
plt.title("Probability Plot one-step return")
plt.ylabel("Ordered quantiles")
pylab.show()
#%%
'''Probability Plot ret2'''
stats.probplot(data_two_step, dist="norm", plot=pylab)
plt.title("Probability Plot two-step return")
plt.ylabel("Ordered quantiles")
pylab.show()
#%%
'''Probability Plot ret3'''
stats.probplot(data_three_step, dist="norm", plot=pylab)
plt.title("Probability Plot three-step return")
plt.ylabel("Ordered quantiles")
pylab.show()
#%%
'''Probability Plot ret4'''
stats.probplot(data_four_step, dist="norm", plot=pylab)
plt.title("Probability Plot four-step return")
plt.ylabel("Ordered quantiles")
pylab.show()
#%%
'''Probability Plot ret5'''
stats.probplot(data_five_step, dist="norm", plot=pylab)
plt.title("Probability Plot five-step return")
plt.ylabel("Ordered quantiles")
#plt.xlim(-10, 10)  # Hier können Sie die x-Achsenbegrenzungen anpassen
#plt.ylim(-10, 10)
pylab.show()

#%%
#%%
import statsmodels.api as sm 
import pylab as py 
sm.qqplot(data_one_step, line ='45') 
py.show()

#%%
'''QQ-Plot ret2'''
stats.probplot(data_two_step, dist="norm", plot=pylab)
plt.title("QQ Plot - Two-Step Return")
pylab.show()
#%%
'''QQ-Plot ret3'''
stats.probplot(data_three_step, dist="norm", plot=pylab)
plt.title("QQ Plot - Three-Step Return")
pylab.show()
#%%
'''QQ-Plot ret4'''
stats.probplot(data_four_step, dist="norm", plot=pylab)
plt.title("QQ Plot - Four-Step Return")
pylab.show()
#%%
'''QQ-Plot ret5'''
stats.probplot(data_five_step, dist="norm", plot=pylab)
plt.title("QQ Plot - Five-Step Return")
pylab.show()
#%%
'''====================================================================================================='''


X = hist[['ret1']].values
X = X.reshape((len(X),))
y_mean = hist['ret1'].values
y_var = np.var(hist[['ret1']], axis = 1)

# %%
X_train, X_test, y_mean_train, y_mean_test, y_var_train, y_var_test = train_test_split(
    X, y_mean, y_var, test_size=0.2, random_state=42, shuffle = False
)
# %%
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)
# %%
from keras.layers import Input, Dense, Activation
from keras.models import Model
input_layer = Input(shape=(1,))
hidden_layer_1 = Dense(64, activation='relu')(input_layer)
hidden_layer_2 = Dense(32, activation='relu')(hidden_layer_1)
# Ausgabe für den Mittelwert mit 'relu' Aktivierung
mean_output = Dense(1, activation=None)(hidden_layer_2)
# Ausgabe für die Varianz mit 'softplus' Aktivierung
variance_output = Dense(1, activation='softplus')(hidden_layer_2)

model = Model(inputs=input_layer, outputs=[mean_output, variance_output])
#%%
import CRPS.CRPS as pscore
#%%
import keras.backend as K

def regression_nll_loss(sigma_sq, epsilon = 1e-6):
    def nll_loss(y_true, y_pred):
        return 0.5 * K.mean(K.log(sigma_sq + epsilon) + K.square(y_true - y_pred) / (sigma_sq + epsilon))

    return nll_loss
#%%

#%%
x = hist['ret1'].values  # univariate Zeitreihe

# Berechne den Mittelwert und die Varianz für jeden Zeitschritt
y_mean_true = np.mean(x)
y_var_true = np.var(x)

# Teile die Daten in Trainings- und Testsets auf
x_train, x_test, _, _, _, _ = train_test_split(
    x, np.zeros_like(x), np.zeros_like(x), test_size=0.2, random_state=42, shuffle=False
)

# Wiederhole den Zielwert für die Gesamtmittelwert- und Varianzschätzung auf die Länge der Trainingsdaten
y_mean_true_train = np.tile(np.array([y_mean_true]), len(x_train))
y_var_true_train = np.tile(np.array([y_var_true]), len(x_train))

# Modelldefinition
input_layer = Input(shape=(1,))
hidden_layer = Dense(32, activation='relu')(input_layer)
mean_output = Dense(1, activation='linear', name='mean_output')(hidden_layer)
var_output = Dense(1, activation='softplus', name='var_output')(hidden_layer)

model = Model(inputs=input_layer, outputs=[mean_output, var_output])

# Verlustfunktion für den Mittelwert und die Varianz
def regression_nll_loss(sigma_sq, epsilon=1e-6):
    def nll_loss(y_true, y_pred):
        return 0.5 * K.mean(K.log(sigma_sq + epsilon) + K.square(y_true - y_pred) / (sigma_sq + epsilon))

    return nll_loss

# Verlustfunktion für beide Ausgänge
loss = {
    'mean_output': regression_nll_loss(var_output),
    'var_output': regression_nll_loss(var_output)
}

# Modell kompilieren
model.compile(optimizer='adam', loss=loss)

#%%
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Modell trainieren
model.fit(x_train, {'mean_output': y_mean_true_train, 'var_output': y_var_true_train}, epochs=50)


#%%
predictions_test = model.predict(X_test)
#%%
expected_mean = np.mean(predictions_test[0])
#%%
expected_var = np.var(predictions_test[1])

#%%
predictions_mean = predictions_test[0]
predictions_var = predictions_test[1]
#%%
predictions_mean = predictions_mean.reshape((208,))
predictions_var = predictions_var.reshape((208,))

#%%
mc_samples = np.random.normal(loc=predictions_mean, scale=np.sqrt(predictions_var), size=(num_simulations, len(x_test)))
#%%
overall_mean_mc = np.mean(mean_mc)
overall_mean_mc 
#%%
overall_std_mc  = np.mean(std_mc)
overall_std_mc
#%%
# Berechne das 97.5%-Quantil für jede Beobachtung
quantile_97_5_per_observation = np.percentile(mc_samples, q=97.5, axis=0)

# Berechne den Durchschnitt über alle Beobachtungen
overall_quantile_97_5 = np.mean(quantile_97_5_per_observation)

print(f"97.5%-Quantil (Durchschnitt über Beobachtungen): {overall_quantile_97_5}")
#%%

# Berechne das 97.5%-Quantil für jede Beobachtung
quantile_7_5_per_observation = np.percentile(mc_samples, q=75, axis=0)

# Berechne den Durchschnitt über alle Beobachtungen
overall_quantile_7_5 = np.mean(quantile_7_5_per_observation)

print(f"75.0%-Quantil (Durchschnitt über Beobachtungen): {overall_quantile_7_5}")

#%%
# Berechne das 97.5%-Quantil für jede Beobachtung
quantile_7_5_per_observation = np.percentile(mc_samples, q=50, axis=0)

# Berechne den Durchschnitt über alle Beobachtungen
overall_quantile_7_5 = np.mean(quantile_7_5_per_observation)

print(f"50.0%-Quantil (Durchschnitt über Beobachtungen): {overall_quantile_7_5}")

#%%
# Berechne das 97.5%-Quantil für jede Beobachtung
quantile_7_5_per_observation = np.percentile(mc_samples, q=25, axis=0)

# Berechne den Durchschnitt über alle Beobachtungen
overall_quantile_7_5 = np.mean(quantile_7_5_per_observation)

print(f"25.0%-Quantil (Durchschnitt über Beobachtungen): {overall_quantile_7_5}")

#%%
# Berechne das 97.5%-Quantil für jede Beobachtung
quantile_7_5_per_observation = np.percentile(mc_samples, q=2.5, axis=0)

# Berechne den Durchschnitt über alle Beobachtungen
overall_quantile_7_5 = np.mean(quantile_7_5_per_observation)

print(f"2.5%-Quantil (Durchschnitt über Beobachtungen): {overall_quantile_7_5}")
#%%






















#%%
plt.hist(predictions_test[0], bins=30, density=True, alpha=0.7, color='blue')
plt.xlabel('Vorhergesagter Mittelwert')
plt.ylabel('Häufigkeit')
plt.title('Histogramm der vorhergesagten Mittelwerte')
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Daten für das Histogramm
data = predictions_test[0]

# Histogramm erstellen
plt.hist(data, bins=50, density=True, alpha=0.7, color='blue', label='Histogramm')

# Mittelwert und Standardabweichung der Daten berechnen
mean_data, std_data = np.mean(data), np.std(data)

# Erzeuge Daten für die Normalverteilung mit den gleichen Parametern
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_data, std_data)

# Plotte die PDF der Normalverteilung
plt.plot(x, p, 'k', linewidth=2, label='Normalverteilung')

plt.xlabel('Vorhergesagter Mittelwert')
plt.ylabel('Häufigkeit')
plt.title('Histogramm der vorhergesagten Mittelwerte mit Normalverteilung')
plt.legend()
plt.show()

#%%
lower_quantile, upper_quantile = np.percentile(data, [2.5, 97.5])

print("Untere Grenze des 95%-Quantils:", lower_quantile)
print("Obere Grenze des 95%-Quantils:", upper_quantile)
#%%
# Überprüfe, wie viele Datenpunkte innerhalb des 95%-Quantils liegen
within_quantile = np.sum((data >= lower_quantile) & (data <= upper_quantile))
total_data_points = len(data)

percentage_within_quantile = (within_quantile / total_data_points) * 100

print(f"{percentage_within_quantile:.2f}% der Daten liegen im 95%-Quantil.")


#%%


quantiles_var = np.percentile(simulated_samples_var, [5, 25, 50, 75, 95], axis=0)



#%%
data_one_step = hist['ret1'].values
data_two_step = hist['ret2'].values
data_three_step = hist['ret3'].values
data_four_step = hist['ret4'].values
data_five_step = hist['ret5'].values
#%%
from scipy.stats import norm #, genhybrid
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))

data_list = [data_one_step, data_two_step, data_three_step, data_four_step, data_five_step]
titles = ['One-Step-Return','Two-Step Return', 'Three-Step Return', 'Four-Step Return', 'Five-Step Return']

for i, data in enumerate(data_list):
    ax = axes[i]
    ax.hist(data, bins=100, density=True, alpha=0.7, color='blue', label='Histogram')
    ax.set_title(titles[i])

    # Fit mit normaler Verteilung
    mean, std = norm.fit(data)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)
    ax.plot(x, p, 'k', linewidth=2, label='Normal Distribution')

    # Fit mit Log Normal distribution
    #params = genhybrid.fit(data)
    #ghy_p = genhybrid.pdf(x, *params)
    #ax.plot(x, ghy_p, 'r', linewidth=2, label='Genhybrid Distribution')

    ax.legend()

plt.tight_layout()
plt.show()
# %%


'''
-------------------------------
Plotten der ret1 Daten von 2023
-------------------------------
'''

data_one_step = hist['ret1'].values


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, genextreme

data_one_step = hist['ret1'].values
returns = data_one_step  

# Histogramm der Zeitreihe
plt.hist(returns, bins=40, density=True, alpha=0.7, color='blue', label='Histogram')

# Fit mit normaler Verteilung
mean, std = norm.fit(returns)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
plt.xlim(-6, 6)
plt.ylim(0, 0.8)

plt.xlabel('Returns')
plt.ylabel('Density')
plt.title('One-Step Return')
plt.legend()
# %%

'''
-------------------------------
Plotten der ret2 Daten von 2023
-------------------------------
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, genextreme

data_two_step = hist['ret2'].values
returns = data_one_step  

# Histogramm der Zeitreihe
plt.hist(returns, bins=40, density=True, alpha=0.7, color='blue', label='Histogram')

# Fit mit normaler Verteilung
mean, std = norm.fit(returns)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
plt.xlim(-6, 6)
plt.ylim(0, 0.8)

plt.xlabel('Returns')
plt.ylabel('Density')
plt.title('Two-Step Return')
plt.legend()
# %%



'''
-------------------------------
Plotten der ret3 Daten von 2023
-------------------------------
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, genextreme

data_three_step = hist['ret3'].values
returns = data_three_step  

# Histogramm der Zeitreihe
plt.hist(returns, bins=40, density=True, alpha=0.7, color='blue', label='Histogram')

# Fit mit normaler Verteilung
mean, std = norm.fit(returns)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
plt.xlim(-10, 10)
plt.ylim(0, 0.8)

plt.xlabel('Returns')
plt.ylabel('Density')
plt.title('Three-Step Return')
plt.legend()

# %%

'''
-------------------------------
Plotten der ret4 Daten von 2023
-------------------------------
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, genextreme

data_four_step = hist['ret4'].values
returns = data_four_step  

# Histogramm der Zeitreihe
plt.hist(returns, bins=40, density=True, alpha=0.7, color='blue', label='Histogram')

# Fit mit normaler Verteilung
mean, std = norm.fit(returns)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
plt.xlim(-10, 10)
plt.ylim(0, 0.8)

plt.xlabel('Returns')
plt.ylabel('Density')
plt.title('Four-Step Return')
plt.legend()
# %%

'''
-------------------------------
Plotten der ret5 Daten von 2023
-------------------------------
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, genextreme

data_five_step = hist['ret5'].values
returns = data_five_step  

# Histogramm der Zeitreihe
plt.hist(returns, bins=40, density=True, alpha=0.7, color='blue', label='Histogram')

# Fit mit normaler Verteilung
mean, std = norm.fit(returns)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
plt.xlim(-10, 10)
plt.ylim(0, 0.8)

plt.xlabel('Returns')
plt.ylabel('Density')
plt.title('Five-Step Return')
plt.legend()
#%%


#%%
'''
-------
Return2
-------
Plotten des Histogramms um zu schauen ob wir der Annahme der Normalverteilung gerecht werden
'''
plt.hist(data, bins=30, density=True, alpha=0.7, color='blue')
plt.title('Histogramm ret2')
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Daten für das Histogramm


# Histogramm erstellen
plt.hist(data, bins=50, density=True, alpha=0.7, color='blue', label='Histogramm')

# Mittelwert und Standardabweichung der Daten berechnen
mean_data, std_data = np.mean(data), np.std(data)

# Erzeuge Daten für die Normalverteilung mit den gleichen Parametern
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_data, std_data)

# Plotte die PDF der Normalverteilung
plt.plot(x, p, 'k', linewidth=2, label='Normalverteilung')

plt.xlabel('Vorhergesagter Mittelwert')
plt.ylabel('Häufigkeit')
plt.title('Histogramm der vorhergesagten Mittelwerte mit Normalverteilung')
plt.legend()
plt.show()

#%%
lower_quantile, upper_quantile = np.percentile(data, [2.5, 97.5])

print("Untere Grenze des 95%-Quantils:", lower_quantile)
print("Obere Grenze des 95%-Quantils:", upper_quantile)
#%%
# Überprüfe, wie viele Datenpunkte innerhalb des 95%-Quantils liegen
within_quantile = np.sum((data >= lower_quantile) & (data <= upper_quantile))
total_data_points = len(data)

percentage_within_quantile = (within_quantile / total_data_points) * 100

print(f"{percentage_within_quantile:.2f}% der Daten liegen im 95%-Quantil.")


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

# %%
def create_input_output(data, sequence_length):
    x, y_mean, y_var = [], [], []
    for i in range(len(data1) - sequence_length):
        x.append(data1[i:i+sequence_length])
        y_mean.append(np.mean(data1[i+1:i+sequence_length+1]))
        y_var.append(np.var(data1[i+1:i+sequence_length+1]))
    return np.array(x), np.array(y_mean), np.array(y_var)
# %%
'''
sequence_lenght: 
---------------
- wie viele aufeinander folgende Zeitpunkt in der Eingabezeitreihe das Modell betrachten soll
- bestimmt die Anzahl der vergangenen Zeitschritte die das Modell für die Vorhersagen des nächsten Schrittes verwenden soll.
- ein längeres sequence_length ermöglicht langfristige Abhängigkeite in den Daten zu erfassen (aber ist auch höherer Rechenaufwand)

units:
------
- bezieht sich auf die Anzahl der Neuronen im LSTM-Layer
- Anzahl der Einheiten bestimmt die Kapazität des Models und komplexere Muster zu erfassen
- (kann aber auch zu Overfitting führen)
'''

sequence_length = 15  
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
# Vorhersage für neue Daten (z. B. die letzten drei Renditen in Ihrer Zeitreihe)
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

'''
----------------------------------------------------------------------
Ret2
----------------------------------------------------------------------
'''   
#%%    

data2 = hist['ret2']
data2 = pd.Series(data2)

# Funktion zum Erstellen der Eingabe und Ausgabe für das Modell
def create_input_output(data, sequence_length, forecast_step):
    x, y_mean, y_var = [], [], []
    for i in range(len(data2) - sequence_length - forecast_step + 1):
        x.append(data2[i:i+sequence_length])
        y_mean.append(np.mean(data2[i+sequence_length:i+sequence_length+forecast_step]))
        y_var.append(np.var(data2[i+sequence_length:i+sequence_length+forecast_step]))
    return np.array(x), np.array(y_mean), np.array(y_var)

# Hyperparameter
sequence_length = 15
forecast_step = 2  # Übernächster Tag
units = 100

# Daten formatieren
x, y_mean, y_var = create_input_output(data2, sequence_length, forecast_step)

# Eingabe für das Modell
inputs = Input(shape=(sequence_length, 1))
lstm_out = LSTM(units)(inputs)

# Ausgabe für den geschätzten Mittelwert
mean_output = Dense(1, name='mean', activation = 'linear')(lstm_out)

# Ausgabe für die geschätzte Varianz
variance_output = Dense(1, name='variance', activation = 'softplus')(lstm_out)

# Modell erstellen
model = Model(inputs=inputs, outputs=[mean_output, variance_output])

# Modell kompilieren
model.compile(optimizer='adam', loss={'mean': 'mse', 'variance': 'mse'})

# Modell trainieren
model.fit(x, {'mean': y_mean, 'variance': y_var}, epochs=20, batch_size=32)

# Vorhersage für neuen Datenpunkt (z. B. den letzten Log-Renditen-Wert)
new_data_ret2 = np.array(data2[-sequence_length:])
predicted_mean2, predicted_var2 = model.predict(new_data_ret2.reshape((1, sequence_length, 1)))

print("Geschätzter Mittelwert für den übernächsten Tag:", predicted_mean2[0][0])
print("Geschätzte Varianz für den übernächsten Tag:", predicted_var2[0][0])


#%%

'''
-----------------------------------------------------------------------------
Quantils Berechnung mit dem geschätzen Mittelwert und der geschätzen Varianz
2.5 %
25 %
50 %
75 %
97.5 %
----------------------------------------------------------------------------
'''
mu2 = predicted_mean2[0][0]
variance2 = predicted_var2[0][0]
alpha_values = [0.025, 0.25, 0.5, 0.75, 0.975]

sigma2 = (variance2)**0.5

quantiles2 = [mu2 + norm.ppf(alpha) * sigma2 for alpha in alpha_values]


quantiles_ret2 = pd.DataFrame([quantiles2])
quantiles_ret2.columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']
quantiles_ret2



# %%



'''
----------------------------------------------------------------------
Ret3
----------------------------------------------------------------------
'''   
#%%    

data3 = hist['ret3']
data3 = pd.Series(data3)

# Funktion zum Erstellen der Eingabe und Ausgabe für das Modell
def create_input_output(data, sequence_length, forecast_step):
    x, y_mean, y_var = [], [], []
    for i in range(len(data3) - sequence_length - forecast_step + 1):
        x.append(data3[i:i+sequence_length])
        y_mean.append(np.mean(data3[i+sequence_length:i+sequence_length+forecast_step]))
        y_var.append(np.var(data3[i+sequence_length:i+sequence_length+forecast_step]))
    return np.array(x), np.array(y_mean), np.array(y_var)

# Hyperparameter
sequence_length = 15
forecast_step = 3  # Übernächster Tag
units = 100

# Daten formatieren
x, y_mean, y_var = create_input_output(data3, sequence_length, forecast_step)

# Eingabe für das Modell
inputs = Input(shape=(sequence_length, 1))
lstm_out = LSTM(units)(inputs)

# Ausgabe für den geschätzten Mittelwert
mean_output = Dense(1, name='mean')(lstm_out)

# Ausgabe für die geschätzte Varianz
variance_output = Dense(1, name='variance')(lstm_out)

# Modell erstellen
model = Model(inputs=inputs, outputs=[mean_output, variance_output])

# Modell kompilieren
model.compile(optimizer='adam', loss={'mean': 'mse', 'variance': 'mse'})

# Modell trainieren
model.fit(x, {'mean': y_mean, 'variance': y_var}, epochs=20, batch_size=32)

# Vorhersage für neuen Datenpunkt (z. B. den letzten Log-Renditen-Wert)
new_data_ret3 = np.array(data3[-sequence_length:])
predicted_mean3, predicted_var3 = model.predict(new_data_ret3.reshape((1, sequence_length, 1)))

print("Geschätzter Mittelwert für den übernächsten Tag:", predicted_mean3[0][0])
print("Geschätzte Varianz für den übernächsten Tag:", predicted_var3[0][0])


#%%
'''
-----------------------------------------------------------------------------
Quantils Berechnung mit dem geschätzen Mittelwert und der geschätzen Varianz
2.5 %
25 %
50 %
75 %
97.5 %
----------------------------------------------------------------------------
'''
mu3 = predicted_mean3[0][0]
variance3 = predicted_var3[0][0]
alpha_values = [0.025, 0.25, 0.5, 0.75, 0.975]

sigma3 = (variance3)**0.5

quantiles3 = [mu3 + norm.ppf(alpha) * sigma3 for alpha in alpha_values]


quantiles_ret3 = pd.DataFrame([quantiles3])
quantiles_ret3.columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']
quantiles_ret3


#%%

'''
----------------------------------------------------------------------
Ret4
----------------------------------------------------------------------
'''   
#%%    

data4 = hist['ret4']
data4 = pd.Series(data4)

# Funktion zum Erstellen der Eingabe und Ausgabe für das Modell
def create_input_output(data, sequence_length, forecast_step):
    x, y_mean, y_var = [], [], []
    for i in range(len(data4) - sequence_length - forecast_step + 1):
        x.append(data4[i:i+sequence_length])
        y_mean.append(np.mean(data4[i+sequence_length:i+sequence_length+forecast_step]))
        y_var.append(np.var(data4[i+sequence_length:i+sequence_length+forecast_step]))
    return np.array(x), np.array(y_mean), np.array(y_var)

# Hyperparameter
sequence_length = 15
forecast_step = 4  # Übernächster Tag
units = 100

# Daten formatieren
x, y_mean, y_var = create_input_output(data4, sequence_length, forecast_step)

# Eingabe für das Modell
inputs = Input(shape=(sequence_length, 1))
lstm_out = LSTM(units)(inputs)

# Ausgabe für den geschätzten Mittelwert
mean_output = Dense(1, name='mean', activation = 'linear')(lstm_out)

# Ausgabe für die geschätzte Varianz
variance_output = Dense(1, name='variance', activation = 'softplus')(lstm_out)

# Modell erstellen
model = Model(inputs=inputs, outputs=[mean_output, variance_output])

# Modell kompilieren
model.compile(optimizer='adam', loss={'mean': 'mse', 'variance': 'mse'})

# Modell trainieren
model.fit(x, {'mean': y_mean, 'variance': y_var}, epochs=20, batch_size=32)

# Vorhersage für neuen Datenpunkt (z. B. den letzten Log-Renditen-Wert)
new_data_ret4 = np.array(data4[-sequence_length:])
predicted_mean4, predicted_var4 = model.predict(new_data_ret4.reshape((1, sequence_length, 1)))

print("Geschätzter Mittelwert für den übernächsten Tag:", predicted_mean4[0][0])
print("Geschätzte Varianz für den übernächsten Tag:", predicted_var4[0][0])


#%%
'''
-----------------------------------------------------------------------------
Quantils Berechnung mit dem geschätzen Mittelwert und der geschätzen Varianz
2.5 %
25 %
50 %
75 %
97.5 %
----------------------------------------------------------------------------
'''
mu4 = predicted_mean4[0][0]
variance4 = predicted_var4[0][0]
alpha_values = [0.025, 0.25, 0.5, 0.75, 0.975]

sigma4 = (variance4)**0.5

quantiles4 = [mu4 + norm.ppf(alpha) * sigma4 for alpha in alpha_values]


quantiles_ret4 = pd.DataFrame([quantiles4])
quantiles_ret4.columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']
quantiles_ret4

#%%
    
'''
----------------------------------------------------------------------
Ret5
----------------------------------------------------------------------
'''   
#%%    

data5 = hist['ret5']
data5 = pd.Series(data5)

# Funktion zum Erstellen der Eingabe und Ausgabe für das Modell
def create_input_output(data, sequence_length, forecast_step):
    x, y_mean, y_var = [], [], []
    for i in range(len(data5) - sequence_length - forecast_step + 1):
        x.append(data5[i:i+sequence_length])
        y_mean.append(np.mean(data5[i+sequence_length:i+sequence_length+forecast_step]))
        y_var.append(np.var(data5[i+sequence_length:i+sequence_length+forecast_step]))
    return np.array(x), np.array(y_mean), np.array(y_var)

# Hyperparameter
sequence_length = 15
forecast_step = 5  
units = 100

# Daten formatieren
x, y_mean, y_var = create_input_output(data5, sequence_length, forecast_step)

# Eingabe für das Modell
inputs = Input(shape=(sequence_length, 1))
lstm_out = LSTM(units)(inputs)

# Ausgabe für den geschätzten Mittelwert
mean_output = Dense(1, name='mean')(lstm_out)

# Ausgabe für die geschätzte Varianz
variance_output = Dense(1, name='variance')(lstm_out)

# Modell erstellen
model = Model(inputs=inputs, outputs=[mean_output, variance_output])

# Modell kompilieren
model.compile(optimizer='adam', loss={'mean': 'mse', 'variance': 'mse'})

# Modell trainieren
model.fit(x, {'mean': y_mean, 'variance': y_var}, epochs=20, batch_size=32)

# Vorhersage für neuen Datenpunkt (z. B. den letzten Log-Renditen-Wert)
new_data_ret5 = np.array(data5[-sequence_length:])
predicted_mean5, predicted_var5 = model.predict(new_data_ret5.reshape((1, sequence_length, 1)))

print("Geschätzter Mittelwert für den übernächsten Tag:", predicted_mean5[0][0])
print("Geschätzte Varianz für den übernächsten Tag:", predicted_var5[0][0])


#%%
'''
-----------------------------------------------------------------------------
Quantils Berechnung mit dem geschätzen Mittelwert und der geschätzen Varianz
2.5 %
25 %
50 %
75 %
97.5 %
----------------------------------------------------------------------------
'''
mu5 = predicted_mean5[0][0]
variance5 = predicted_var5[0][0]
alpha_values = [0.025, 0.25, 0.5, 0.75, 0.975]

sigma5 = (variance5)**0.5


quantiles5 = [mu5 + norm.ppf(alpha) * sigma5 for alpha in alpha_values]


quantiles_ret5 = pd.DataFrame([quantiles5])
quantiles_ret5.columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']
quantiles_ret5


#%%
'''
Datensätze zusammen fügen
'''
q_returns = pd.concat([quantiles_ret1, quantiles_ret2, quantiles_ret3, quantiles_ret4, quantiles_ret5], axis = 0)
print(q_returns)

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



#PATH = "/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/PTSFC Submission JahrMonatTag_Rachel"

#df_sub_dax_MVE.to_csv(PATH+ "MVE_DAX_Abgabe11.csv", index=False)
#%%

'''
--------------
Quantile Loss
--------------
'''

def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        # Fall, in dem q_hat gleich y ist (Randfall)
        return 0
#%%
true_return_1 = next_5_days['ret1'].iloc[0]
true_return_1
#%%
true_return_2 = next_5_days['ret2'].iloc[1]
true_return_2 
# %%
true_return_3 = next_5_days['ret3'].iloc[2]
true_return_3
# %%
true_return_4 = next_5_days['ret4'].iloc[3]
true_return_4
# %%
true_return_5 = next_5_days['ret5'].iloc[4]
true_return_5
#%%
quant1_hat = df_sub_dax_MVE ['q0.025']
quant2_hat = df_sub_dax_MVE ['q0.25']
quant3_hat = df_sub_dax_MVE ['q0.5']
quant4_hat = df_sub_dax_MVE ['q0.75']
quant5_hat = df_sub_dax_MVE ['q0.975']

#%%
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
actual_returns = [true_return_1, true_return_2, true_return_3, true_return_4, true_return_5]
quantile_hats = [quant1_hat, quant2_hat, quant3_hat, quant4_hat, quant5_hat]
quantile_hats_df = pd.DataFrame(quantile_hats, columns=['q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975'])
quantile_hats_df = quantile_hats_df.T

quantiles = [0.25, 0.025]

coverage_probabilities = {}

for i, actual_return in enumerate(actual_returns, start=1):
    coverage_probabilities[f'Return_{i}'] = {}

    for q in quantiles:
        # Benennung der Spalten für untere und obere Grenzen
        lower_bound_col = f'q{q}'
        upper_bound_col = f'q{1.0 - q}'  # Oberstes Quantil ist 1.0 minus das unterste Quantil

        if q < 0.5:
            # Coverage Probability für jedes Quantil
            is_covered = (actual_return >= quantile_hats[lower_bound_col]) & (actual_return <= quantile_hats[upper_bound_col])
            coverage_probability = is_covered.mean()

            # Speichern die Coverage Probability für das aktuelle Quantil und die aktuelle Rendite
            coverage_probabilities[f'Return_{i}'][f'quantile_{q}'] = coverage_probability

# Ausgabe der Ergebnisse
for return_key, return_coverage in coverage_probabilities.items():
    for q, coverage in return_coverage.items():
        print(f'{return_key} {q}: {coverage:.2%}')







































































































# %%

'''
LSTM Netz
----------
- Trainings Daten (70%)
- Test Daten (15%)
- Validierungs Daten (15%)
'''

#%%

'''
------
ret 1
------
'''
from sklearn.model_selection import train_test_split
#%%
# Daten laden
data1 = hist['ret1']
data1 = pd.Series(data1)

#%%
# Funktion zum Erstellen der Eingabe und Ausgabe für das Modell
def create_input_output(data, sequence_length, forecast_step):
    x, y_mean, y_var = [], [], []
    for i in range(len(data) - sequence_length - forecast_step + 1):
        x.append(data1[i:i+sequence_length])
        y_mean.append(np.mean(data1[i+sequence_length:i+sequence_length+forecast_step]))
        y_var.append(np.var(data1[i+sequence_length:i+sequence_length+forecast_step]))
    return np.array(x), np.array(y_mean), np.array(y_var)
#%%
# Hyperparameter
sequence_length = 15
forecast_step = 1  # Nächster Tag
units = 50
#%%
# Daten formatieren
x, y_mean, y_var = create_input_output(data1, sequence_length, forecast_step)
#%%
# Aufteilung in Trainings-, Validierungs- und Testdaten
x_train, x_temp, y_train_mean, y_temp_mean, y_train_var, y_temp_var = train_test_split(x, y_mean, y_var, test_size=0.3, random_state=42, shuffle = False)
x_val, x_test, y_val_mean, y_test_mean, y_val_var, y_test_var = train_test_split(x_temp, y_temp_mean, y_temp_var, test_size=0.5, random_state=42, shuffle= False)
#%%
# Eingabe für das Modell
inputs = Input(shape=(sequence_length, 1))
lstm_out = LSTM(units)(inputs)
#%%
# Ausgabe für den geschätzten Mittelwert
mean_output = Dense(1, name='mean', activation = 'linear')(lstm_out)

# Ausgabe für die geschätzte Varianz
variance_output = Dense(1, name='variance', activation = 'softplus')(lstm_out)
#%%
# Modell erstellen
model = Model(inputs=inputs, outputs=[mean_output, variance_output])
#%%
# Modell kompilieren
model.compile(optimizer='adam', loss={'mean': 'mse', 'variance': 'mse'})
#%%
# Modell trainieren
model.fit(x_train, {'mean': y_train_mean, 'variance': y_train_var}, validation_data=(x_val, {'mean': y_val_mean, 'variance': y_val_var}), epochs=50, batch_size=32)
#model.fit(x, {'mean': y_mean, 'variance': y_var}, epochs=20, batch_size=32)

#%%
'''
------------------------------------------------------------------------------
Überwachen des Validierungsdatensatzes während des Trainings 
Training wird gestoppt wenn der Validierungsverlust nicht mehr verbessert wird
-----------------------------------------------------------------------------
'''
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(x_train, {'mean': y_train_mean, 'variance': y_train_var},
          validation_data=(x_val, {'mean': y_val_mean, 'variance': y_val_var}),
          epochs=50, batch_size=32, callbacks=[early_stopping])
#%%
# Evaluierung auf Testdaten
test_loss = model.evaluate(x_test, {'mean': y_test_mean, 'variance': y_test_var}, batch_size=32)
print("Test Loss:", test_loss)
#%%
# Vorhersage für neuen Datenpunkt (z. B. den letzten Log-Renditen-Wert)
new_data_1 = np.array(data1[-sequence_length:])
predicted_mean_1, predicted_var_1 = model.predict(new_data_1.reshape((1, sequence_length, 1)))

print("Geschätzter Mittelwert für den übernächsten Tag:", predicted_mean_1[0][0])
print("Geschätzte Varianz für den übernächsten Tag:", predicted_var_1[0][0])


#%%

model.fit(x, {'mean': y_mean, 'variance': y_var},
          epochs=22, batch_size=32)


new_data = np.array(data[-sequence_length:])
predicted_mean_1, predicted_var_1 = model.predict(new_data.reshape((1, sequence_length, 1)))

print("Geschätzter Mittelwert für den übernächsten Tag:", predicted_mean_1[0][0])
print("Geschätzte Varianz für den übernächsten Tag:", predicted_var_1[0][0])


#%%

mu_1 = predicted_mean_1[0][0]
variance_1 = predicted_var_1[0][0]
alpha_values = [0.025, 0.25, 0.5, 0.75, 0.975]

sigma_1 = (variance_1)**0.5


quantiles_1 = [mu_1 + norm.ppf(alpha) * sigma_1 for alpha in alpha_values]


q_ret_1 = pd.DataFrame([quantiles_1])
q_ret_1.columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']
q_ret_1









