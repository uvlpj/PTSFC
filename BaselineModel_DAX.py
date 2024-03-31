#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
import yfinance as yf

msft = yf.Ticker("^GDAXI")
#%%
hist = msft.history(period="max")
#%%
hist
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

#pythonic alternative (however: slower) to do the same thing:
def compute_return_2(df, r_type="log", h=1):
    
    if r_type == "log":
        return (np.log(df) - np.log(df.shift(h))) * 100
    else:
        return ((df-df.shift(h))/df) * 10
# %%
#hist = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission4/^GDAXI-2.csv')
#data = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission5/^GDAXI Kopie.csv')
hist = pd.read_csv('/Users/sophiasiefert/Downloads/^GDAXI.csv')
#%%
#hist2 = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/22_11_23^GDAXI.csv')
#hist2 = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission5/^GDAXI.csv')

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

# %%
for i in range(5):
    hist["ret"+str(i+1)] = compute_return(hist["Close"].values, h=i+1)
# %%
tau = [.025, .25, .5, .75, .975]
# %%
pred_baseline = np.zeros((5,5))
# %%
last_t = 800

for i in range(5):
    ret_str = "ret"+str(i+1)
    
    pred_baseline[i,:] = np.quantile(hist[ret_str].iloc[-last_t:], q=tau)
# %%
pred_baseline
# %%
from datetime import datetime
forecastdate = datetime(2023, 11, 15, 00, 00)
# %%
df_baseline_sub = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "DAX",
    "horizon": [str(i) + " day" for i in (1,2,5,6,7)],
    "q0.025": pred_baseline[:,0],
    "q0.25": pred_baseline[:,1],
    "q0.5": pred_baseline[:,2],
    "q0.75": pred_baseline[:,3],
    "q0.975": pred_baseline[:,4]})
df_baseline_sub
# %%
from datetime import datetime
date_str = datetime.today().strftime('%Y%m%d')
# %%
PATH = "/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/PTSFC Submission JahrMonatTag_Rachel"

df_baseline_sub.to_csv(PATH+ "Abgabe5_Baseline_DAX.csv", index=False)

#%%



hist.set_index('Date', inplace=True)
#%%
import pandas as pd

# Annahme: hist ist ein DataFrame mit dem Datum als Index

# Konvertiere den Index in einen Pandas-DatetimeIndex
hist.index = pd.to_datetime(hist.index)

start_date = '2021-01-01'
current_end_date = '2023-11-15'

while True:
    # Wähle die Daten aus
    df_dax = hist.loc[start_date:current_end_date]
    
    # Füge den Wochentag als neue Spalte hinzu
    df_dax["weekday"] = df_dax.index.dayofweek
    
    # Drucke die ausgewählten Daten oder führe weitere Operationen durch
    print(df_dax)
    
    # Aktualisiere das Enddatum für die nächste Woche
    current_end_date = pd.to_datetime(current_end_date) + pd.DateOffset(weeks=1)
    
    # Optional: Setze ein Enddatum, um die Schleife zu begrenzen
    if current_end_date >= pd.to_datetime('2024-01-01'):
        break

#%%
hist
#%%
df_dax
#%%

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
#%%

np.random.seed(42)
#%%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Annahme: hist ist Ihr DataFrame mit den Close-Preisen des DAX
# compute_return ist Ihre Funktion zum Berechnen der Renditen für verschiedene Zeithorizonte

# Quantile definieren
tau = [0.025, 0.25, 0.5, 0.75, 0.975]

# Anzahl der letzten Tage für die Schätzung
last_t = 800

# Liste für die Ergebnisse initialisieren
dfs_baseline_sub = []

# Startdatum für die Prognose festlegen
forecast_date = datetime(2023, 11, 15)

# Schleife über die nächsten 10 Wochen
for week_number in range(1, 11):
    # Aktuelles Datum für die Prognose
    current_date = forecast_date
    
    # Wenn wir uns in Woche 6 oder 7 befinden, überspringen wir die Schleife
    if 6 <= week_number <= 7:
        forecast_date += timedelta(weeks=1)
        continue
    
    # Schätzungen für die nächsten 5 Arbeitstage ab dem aktuellen Datum berechnen
    pred_baseline = np.zeros((5, 5))  # 5 Quantile für 5 Arbeitstage
    for j in range(5):
        ret_str = "ret" + str(j + 1)
        
        # Bestimmen des Start- und Enddatums für die Schätzungen
        start_date = current_date - timedelta(days=last_t - 1)  # Startdatum = aktuelles Datum - 799 Tage
        end_date = current_date  # Enddatum = aktuelles Datum
        
        # Quantilschätzungen basierend auf den letzten 800 Werten ab dem aktuellen Datum berechnen
        pred_baseline[j, :] = np.quantile(hist[ret_str].loc[start_date:end_date], q=tau)
    
    # DataFrame mit den Ergebnissen erstellen
    df_baseline_sub = pd.DataFrame({
        "forecast_date": [current_date.strftime("%Y-%m-%d")] * 5,
        "target": ["DAX"] * 5,
        "horizon": [str(horizon) + " day" for horizon in [1, 2, 5, 6, 7]],
        "q0.025": pred_baseline[:, 0],
        "q0.25": pred_baseline[:, 1],
        "q0.5": pred_baseline[:, 2],
        "q0.75": pred_baseline[:, 3],
        "q0.975": pred_baseline[:, 4]
    })
    
    # DataFrame der Liste hinzufügen
    dfs_baseline_sub.append(df_baseline_sub)
    
    # Gehe eine Woche voran für die nächste Iteration
    forecast_date += timedelta(weeks=1)

# Ergebnisse anzeigen
for idx, df in enumerate(dfs_baseline_sub):
    print(f"DataFrame {idx + 1}:")
    print(df)



#%%

dfs_next_5_days = []

# Startdatum für die Prognose festlegen
forecast_date = '2023-11-14'

# Schleife über die nächsten 10 Wochen
for week_number in range(1, 11):
    # Aktuelles Datum für die Prognose
    current_date = pd.to_datetime(forecast_date)
    
    # Wenn wir uns in Woche 6 oder 7 befinden, überspringen wir die Schleife
    if 6 <= week_number <= 7:
        forecast_date = (current_date + pd.DateOffset(weeks=1)).strftime('%Y-%m-%d')
        continue
    
    # Extrahiere die Daten für die nächsten 5 Tage ab dem nächsten Tag nach dem aktuellen Datum
    next_5_days = hist.loc[current_date + pd.DateOffset(days=1):current_date + pd.DateOffset(days=7)]
    
    # DataFrame mit den Ergebnissen erstellen und zur Liste hinzufügen
    dfs_next_5_days.append(next_5_days)
    
    # Gehe eine Woche voran für die nächste Iteration
    forecast_date = (current_date + pd.DateOffset(weeks=1)).strftime('%Y-%m-%d')

# Ergebnisse anzeigen
for idx, df in enumerate(dfs_next_5_days):
    print(f"DataFrame {idx + 1}:")
    print(df)


#%%


'''
--------------------------
Quantile Score Funktion
--------------------------
'''
def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        # Fall, in dem q_hat gleich y ist (Randfall)
        return 0

# Beispielaufruf mit tau=0.5 (Median)
q_hat =  -4.90643 # Beispielwert für die vorhergesagte Quantile
y = 1.740577	  # Beispielwert für die Realisierung
tau = 0.25  # Beispielwert für das Quantil

score = quantile_score(q_hat, y, tau)
print(f"Quantile Score für tau={tau}: {score}")
#%%

tau_values = [0.025, 0.25, 0.5, 0.75, 0.975]
#%%
# Ergebnisse in einem neuen DataFrame speichern

quantile_results = pd.DataFrame()
#%%
true_return_1 = dfs_next_5_days[7]['ret1'].iloc[0]
true_return_1
#%%
true_return_2 = dfs_next_5_days[7]['ret2'].iloc[1]
true_return_2 
# %%
true_return_3 = dfs_next_5_days[7]['ret3'].iloc[2]
true_return_3
# %%
true_return_4 = dfs_next_5_days[7]['ret4'].iloc[3]
true_return_4
# %%
true_return_5 = dfs_next_5_days[7]['ret5'].iloc[4]
true_return_5
#%%

#%%
quant1_hat = dfs_baseline_sub[7]['q0.025']
quant2_hat = dfs_baseline_sub[7]['q0.25']
quant3_hat = dfs_baseline_sub[7]['q0.5']
quant4_hat = dfs_baseline_sub[7]['q0.75']
quant5_hat = dfs_baseline_sub[7]['q0.975']


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


actual_returns = [true_return_1, true_return_2, true_return_3, true_return_4, true_return_5]
actual_returns
#%%
quantile_hats = [quant1_hat, quant2_hat, quant3_hat, quant4_hat, quant5_hat]
quantile_hats = pd.DataFrame(quantile_hats)
quantile_hats = quantile_hats.T 
#%%
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
        print(f'Coverage Probability für Rendite {return_key} und Quantil {q}: {coverage:.2%}')




















# %%

# ==================================================================================================
'''
ENSEMBEL MODEL 
--------------
=> COMBINING BASELINE AND QUANTILE MODEL
=> Weighted Ensembel Model
'''
df_quant = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/PTSFC Submission JahrMonatTag_RachelQuantile_Regression_DAX_Abgabe5.csv')
# %%
weight1 = 0.4
weight2 = 0.6

#%%
df_ensembel_dax = (weight1*df_quant[['q0.025', 'q0.25','q0.5', 'q0.75', 'q0.975' ]] + weight2*df_baseline_sub[['q0.025', 'q0.25','q0.5', 'q0.75', 'q0.975' ]])/(weight1 + weight2)
df_ensembel_dax
# %%
df_ensembel_sub = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "DAX",
    "horizon": [str(i) + " day" for i in (1,2,5,6,7)],
    "q0.025": df_ensembel_dax['q0.025'],
    "q0.25": df_ensembel_dax['q0.25'],
    "q0.5": df_ensembel_dax['q0.5'],
    "q0.75": df_ensembel_dax['q0.75'],
    "q0.975": df_ensembel_dax['q0.975']})
df_ensembel_sub

# %%
PATH = "/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/PTSFC Submission JahrMonatTag_Rachel"

df_baseline_sub.to_csv(PATH+ "Abgabe5_ENSEMBEL_DAX.csv", index=False)
# %%
