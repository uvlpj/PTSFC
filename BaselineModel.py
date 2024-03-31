#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode
def get_energy_data():

    # get all available time stamps
    stampsurl = "https://www.smard.de/app/chart_data/410/DE/index_quarterhour.json"
    response = requests.get(stampsurl)
    #ignore first 4 years (don't need those in the baseline and speeds the code up a bit)
    timestamps = list(response.json()["timestamps"])[4*52:]

 
    col_names = ['date_time','Netzlast_Gesamt']
    energydata = pd.DataFrame(columns=col_names)
    
    # loop over all available timestamps
    for stamp in tqdm(timestamps):

        dataurl = "https://www.smard.de/app/chart_data/410/DE/410_DE_quarterhour_" + str(stamp) + ".json"
        response = requests.get(dataurl)
        rawdata = response.json()["series"]

        for i in range(len(rawdata)):

            rawdata[i][0] = datetime.fromtimestamp(int(str(rawdata[i][0])[:10])).strftime("%Y-%m-%d %H:%M:%S")

        energydata = pd.concat([energydata, pd.DataFrame(rawdata, columns=col_names)])

    energydata = energydata.dropna()
    energydata["date_time"] = pd.to_datetime(energydata.date_time)
    #set date_time as index
    energydata.set_index("date_time", inplace=True)
    #resample
    energydata = energydata.resample("1h", label = "left").sum()

    return energydata

# ==============================================================================================================================
#%%
import requests
from datetime import datetime
from tqdm import tqdm

data_e = get_energy_data()
#%%
data_e.head()
#%%
data_e = data_e.rename(columns={"Netzlast_Gesamt": "gesamt"})
#%%
data_e['gesamt'] = data_e['gesamt'] / 1000
#%%
data_e.head()
# %%
'''
2019    56.767923
'''
df_2019 = data_e[data_e.index.year == 2019]

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(df_2019.index, df_2019['gesamt'], label='Stromverbrauch 2019')

plt.title('Stromverbrauch im Jahr 2019')
plt.xlabel('Datum')
plt.ylabel('Stromverbrauch')
plt.legend()
plt.show()
# %%
average_consumption_per_year = df_2019.groupby(df_2019.index.year)['gesamt'].mean()
average_consumption_per_year
# %%
'''
2020    55.247086
'''
df_2020 = data_e[data_e.index.year == 2020]

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(df_2020.index, df_2020['gesamt'], label='Stromverbrauch 2020')

plt.title('Stromverbrauch im Jahr 2020')
plt.xlabel('Datum')
plt.ylabel('Stromverbrauch')
plt.legend()
plt.show()
# %%
average_consumption_per_year = df_2020.groupby(df_2020.index.year)['gesamt'].mean()
average_consumption_per_year
# %%
'''
2021    57.593144
'''
df_2021 = data_e[data_e.index.year == 2021]

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(df_2021.index, df_2021['gesamt'], label='Stromverbrauch 2021')

plt.title('Stromverbrauch im Jahr 2021')
plt.xlabel('Datum')
plt.ylabel('Stromverbrauch')
plt.legend()
plt.show()
# %%
average_consumption_per_year = df_2021.groupby(df_2021.index.year)['gesamt'].mean()
average_consumption_per_year
# %%
'''
2022   55.09562
'''
df_2022 = data_e[data_e.index.year == 2022]

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(df_2022.index, df_2022['gesamt'], label='Stromverbrauch 2022')

plt.title('Stromverbrauch im Jahr 2022')
plt.xlabel('Datum')
plt.ylabel('Stromverbrauch')
plt.legend()
plt.show()
# %%
average_consumption_per_year = df_2022.groupby(df_2022.index.year)['gesamt'].mean()
average_consumption_per_year
# %%
'''
2023   52.136079
'''
df_2023 = data_e[data_e.index.year == 2023]

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(df_2023.index, df_2023['gesamt'], label='Stromverbrauch 2023')

plt.title('Stromverbrauch im Jahr 2023')
plt.xlabel('Datum')
plt.ylabel('Stromverbrauch')
plt.legend()
plt.show()
# %%
average_consumption_per_year = df_2023.groupby(df_2023.index.year)['gesamt'].mean()
average_consumption_per_year
# %%

'''
=======================================================================================================================
'''
'''
Nur das Jahr 2023 betrachten
'''

df_2023_2024 = data_e[(data_e.index.year == 2023) | (data_e.index.year == 2024)]


#%%
df_2023_2024["weekday"] = df_2023_2024.index.weekday #Monday=0, Sunday=6

#%%

'''Den Datenzeitraum von 01/01/2021 bis 2023 verwenden'''

start1 = '2021-01-01'
end1 = '2023-11-22'

df_e = data_e[(data_e.index >= start1) & (data_e.index < end1)]
print(df_e)
#%%
df_e["weekday"] = df_e.index.weekday

#%%

'''
==========================================================================
Immer Dienstag 23:00 letzte Datenpunkt für die Vorhesage für
- Freitag (12:00, 16:00, 20:00)
- Samsatg (12:00, 16:00, 20:00)

------------------------------------------
             Deadline
Submission1: 15/11/23 (17/11 und 18/11)
Submission2: 22/11/23 (24/11 und 25/11)
Submission3: 29/11/23 
Submission4: 06/12/23
Submission5: 13/12/23
Submission6: 20/12/23
Submission : 27/12/23
Submission : 03/01/24
Submission7: 10/01/24
Submission8: 17/01/24
Submission9: 24/01/23
-------------------------------------------
'''
np.random.seed(42)
#%%

#end_date= pd.to_datetime('2023-12-20 00:00:00')
#sub_dataset = df_2023_2024.loc[df_2023_2024.index < end_date]
sub_dataset = df_e.copy()

#%%
'''
-------------
Baseline Model
------------
'''
#horizons = [36, 40, 44, 60, 64, 68]#[24 + 12*i for i in range(5)]
horizons = [61, 65, 69, 85, 89, 93]

horizons
# %%
def get_date_from_horizon(last_ts, horizon):
    return last_ts + pd.DateOffset(hours=horizon)

#%%
LAST_IDX = -1
LAST_DATE = sub_dataset.iloc[LAST_IDX].name
# %%
horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
horizon_date
# %%
tau = [.025, .25, .5, .75, .975]
# %%
pred_baseline = np.zeros((6,5))

#%%
np.random.seed(42)
last_t = 500

for i,d in enumerate(horizon_date):
    
    weekday = d.weekday()
    hour = d.hour
    
    df_tmp = df_e.iloc[:LAST_IDX]
    
    cond = (df_tmp.weekday == weekday) & (df_tmp.index.time == d.time())
    
    pred_baseline[i,:] = np.quantile(df_tmp[cond].iloc[-last_t:]["gesamt"], q=tau)
# %%
x = horizons
_ = plt.plot(x,pred_baseline, ls="", marker="o", c="black")
_ = plt.xticks(x, x)
_ = plt.plot((x,x),(pred_baseline[:,0], pred_baseline[:,-1]),c='black')

#%%
forecastdate = datetime(2023, 11, 22, 00, 00)

#%%
horizons2 = [36, 40, 44, 60, 64, 68]
df_baseline = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "energy",
    "horizon": [str(h) + " hour" for h in horizons2],
    "q0.025": pred_baseline[:,0],
    "q0.25": pred_baseline[:,1],
    "q0.5": pred_baseline[:,2],
    "q0.75": pred_baseline[:,3],
    "q0.975": pred_baseline[:,4]})
df_baseline


#%%
start_date = '2021-01-01'
current_end_date = '2023-11-15'

while True:
    # Wähle die Daten aus
    df_e = data_e[(data_e.index >= start_date) & (data_e.index < current_end_date)]
    df_e["weekday"] = df_e.index.weekday
    sub_dataset = df_e.copy()
    
    # Drucke die ausgewählten Daten oder führe weitere Operationen durch
    print(sub_dataset)
    
    # Aktualisiere das Enddatum für die nächste Woche
    current_end_date = pd.to_datetime(current_end_date) + pd.DateOffset(weeks=1)
    
    # Optional: Setze ein Enddatum, um die Schleife zu begrenzen
    if current_end_date >= pd.to_datetime('2024-01-01'):
        break


#%%
import pandas as pd
import numpy as np

'''
------------------------------------------------------
Für die 10 Submission Rounds
werden automatisch in einem Durchlauf die Vorhersagen gemacht
-----------------------------------------------------
'''

def select_data(data, start_date, end_date):
    selected_data = data[(data.index >= start_date) & (data.index < end_date)]
    selected_data["weekday"] = selected_data.index.weekday
    sub_dataset = selected_data.copy()
    return sub_dataset


def make_predictions(df):
    np.random.seed(42)
    last_t = 500
    tau = [.025, .25, .5, .75, .975]
    
    # Initialisiere das Modell
    LAST_IDX = -1
    LAST_DATE = df.iloc[LAST_IDX].name
    horizons = [61, 65, 69, 85, 89, 93]
    horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
    pred_baseline = np.zeros((6, 5))

    # Mache Vorhersagen mit dem Modell
    for i, d in enumerate(horizon_date):
        weekday = d.weekday()
        hour = d.hour

        df_tmp = df.iloc[:LAST_IDX]
        cond = (df_tmp.weekday == weekday) & (df_tmp.index.time == d.time())
        pred_baseline[i, :] = np.quantile(df_tmp[cond].iloc[-last_t:]["gesamt"], q=tau)
    
    return pred_baseline




start_date = '2021-01-01'
current_end_date = '2023-11-15'

predictions_dfs = []
# Schleife für jede Woche
while True:
    # Wähle die Daten aus
    df_e = select_data(data_e, start_date, current_end_date)
    
    # Mach Vorhersagen mit deinem Modell
    predictions = make_predictions(df_e)

    current_predictions_df = pd.DataFrame({
        #'forecast_date': [current_end_date] * 6,
        'forecast_date': [pd.to_datetime(current_end_date) + pd.Timedelta(hours=h) for h in [60, 64, 68, 84, 88, 92]],
        'horizon': [str(h) + ' hour' for h in [36, 40, 44, 60, 64, 68]],
        'q0.025': predictions[:, 0],
        'q0.25': predictions[:, 1],
        'q0.5': predictions[:, 2],
        'q0.75': predictions[:, 3],
        'q0.975': predictions[:, 4],
    })
    predictions_dfs.append(current_predictions_df)




    # Drucke die Vorhersagen oder führe weitere Operationen durch
    print(f'Vorhersagen für Woche bis {current_end_date}:\n{predictions}')
    
    # Aktualisiere das Enddatum für die nächste Woche
    current_end_date = pd.to_datetime(current_end_date) + pd.DateOffset(weeks=1)
    
    # Optional: Setze ein Enddatum, um die Schleife zu begrenzen
    if current_end_date >= pd.to_datetime('2024-01-18'):
        break













#%%
def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        # Fall, in dem q_hat gleich y ist (Randfall)
        return 0
#%%
import numpy as np
import pandas as pd

class MyModel:
    def __init__(self, horizon_date, df_e, tau):
        self.model = None  # Hier sollte dein Modell initialisiert werden
        last_t = 500

        self.horizon_date = horizon_date
        self.tau = tau
        self.pred_baseline = np.zeros((len(horizon_date), len(tau)))

        for i, d in enumerate(horizon_date):
            weekday = d.weekday()
            hour = d.hour
            df_tmp = df_e.iloc[:LAST_IDX]
            cond = (df_tmp.weekday == weekday) & (df_tmp.index.time == d.time())
            self.pred_baseline[i, :] = np.quantile(df_tmp[cond].iloc[-last_t:]["gesamt"], q=tau)

    def fit(self, X_train, y_train):
        # Hier sollte dein Trainingsprozess stattfinden
        pass

    def predict_quantiles(self, X, taus):
        # Hier sollte deine Vorhersage für die gegebenen Quantile stattfinden
        # Der Rückgabewert sollte eine Liste von Quantilen für jede Tau sein
        predictions = []
        for tau in taus:
            # Hier werden die vorberechneten Quantilschätzungen verwendet
            prediction = self.pred_baseline[:, self.tau.index(tau)]
            predictions.append(prediction)
        return predictions

# Beispiel-Modell erstellen
horizons = [61, 65, 69, 85, 89, 93]
horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
tau = [.025, .25, .5, .75, .975]
model = MyModel(horizon_date, df_e, tau)

# Beispiel-Funktion für Quantil-Vorhersagen
def predict_quantiles_for_rolling_window(model, df, date_column, feature_column, window_size, taus, submission_deadline):
    df_sorted = df.sort_values(by=[date_column], ascending=True)
    quantile_predictions = {tau: [] for tau in taus}

    for i in range(window_size - 1, len(df_sorted)):
        current_date = df_sorted.iloc[i][date_column]
        if current_date < submission_deadline:
            continue  # Überspringe Daten vor dem Submission Deadline

        window_data = df_sorted.iloc[i - window_size + 1:i + 1]

        X_window = window_data[feature_column].values  # Annahme: Feature ist 'gesamt', anpassen wenn nötig
        quantile_preds = model.predict_quantiles(X_window, taus)

        for tau, pred in zip(taus, quantile_preds):
            quantile_predictions[tau].append(pred)

    return {tau: np.array(preds) for tau, preds in quantile_predictions.items()}

# Beispielaufruf
submission_deadline = pd.to_datetime('2023-11-15')  # Annahme des Submission Deadline
window_size = 5
taus = [0.025, 0.25, 0.5, 0.75, 0.975]
feature_column = 'gesamt'

# Überprüfe die vorhandenen Spaltennamen und den Index
print(df_e.columns)
print(df_e.index.name)

quantile_predictions = predict_quantiles_for_rolling_window(model, df_e, df_e.index.name, feature_column, window_size, taus, submission_deadline)

# Ausgabe der Ergebnisse für jede Tau-Einstellung
for tau, preds in quantile_predictions.items():
    print(f"Quantile Predictions (tau={tau}):", preds)
























#%%
'''
Funktion die für die Forecast Evaluation
=> Gibt die wahren y basierend auf den horizon h und dem forecast_date heraus
'''
forecast_date = pd.to_datetime('2023-11-15 00:00:00')  # Hier dein tatsächliches Prognosedatum einfügen
horizon = [36, 40, 44, 60, 64, 68]

# Extrahiere den Wert für jedes Zeitfenster nach dem Prognosedatum
stromverbrauch_true = {}

for h in horizon:
    target_datetime = forecast_date + pd.Timedelta(hours=h)
    stromverbrauch_true[h] = data_e.loc[target_datetime, 'gesamt']
    stromverbrauch_true_df = pd.DataFrame(list(stromverbrauch_true.items()), columns=['horizon', 'gesamt'])
    stromverbrauch_true_df.set_index('horizon', inplace=True)
print(stromverbrauch_true)


#%%

'''
'''

forecast_dates = pd.date_range(start='2023-11-15', periods=10, freq='W-Fri')
horizon = [12, 16, 20]

stromverbrauch_true_fr = {'datetime': [], 'horizon': [], 'gesamt': []}

for date in forecast_dates:
    for h in horizon:
        target_datetime = date + pd.Timedelta(hours=h)
        stromverbrauch_true_fr['datetime'].append(target_datetime)
        stromverbrauch_true_fr['horizon'].append(h)
        stromverbrauch_true_fr['gesamt'].append(data_e.loc[target_datetime, 'gesamt'])

stromverbrauch_true_df_fr = pd.DataFrame(stromverbrauch_true_fr)
stromverbrauch_true_df_fr.set_index('datetime', inplace=True)
print(stromverbrauch_true_df_fr)


#%%
forecast_dates = pd.date_range(start='2023-11-15', periods=10, freq='W-SAT')
horizon = [12, 16, 20]

stromverbrauch_true_sa = {'datetime': [], 'horizon': [], 'gesamt': []}

for date in forecast_dates:
    for h in horizon:
        target_datetime = date + pd.Timedelta(hours=h)
        stromverbrauch_true_sa['datetime'].append(target_datetime)
        stromverbrauch_true_sa['horizon'].append(h)
        stromverbrauch_true_sa['gesamt'].append(data_e.loc[target_datetime, 'gesamt'])

stromverbrauch_true_df_sa = pd.DataFrame(stromverbrauch_true_sa)
stromverbrauch_true_df_sa.set_index('datetime', inplace=True)
print(stromverbrauch_true_df_sa)

# %%
'''Einfach nur eine Überwachung um zu sehen ob auch wirklich die Wahren Werte extrahiert werden'''
specific_datetime = pd.to_datetime('2024-01-20 20:00:00')
specific_value = data_e.loc[specific_datetime, 'gesamt']
print(f'Wert für {specific_datetime}: {specific_value}')
#%%
combined_df = pd.concat([stromverbrauch_true_df_sa, stromverbrauch_true_df_fr])

# Sortiere das kombinierte DataFrame nach dem Index (Datum)
combined_df_sorted = combined_df.sort_index()

# Zeige das sortierte kombinierte DataFrame an
print(combined_df_sorted)


#%%
new_horizon = [36, 40, 44, 60, 64, 68] * (len(combined_df_sorted) // len([36, 40, 44, 60, 64, 68]) + 1)
combined_df_sorted['horizon'] = new_horizon[:len(combined_df_sorted)]
combined_df_sorted
#%%
'''
---------------
Quantile Score 
für das Baseline Model
Hier werden die 8 letzten Submissions simuliert
für jede der Submissions werden die Quantile Scores berechnet und am Ende der Durchschnitt
AV QS
2.5%, 25%, 50%, 75%, 97.5%

'''
def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        # Fall, in dem q_hat gleich y ist (Randfall)
        return 0

# Iteriere durch alle DataFrames in der Liste predictions_dfs
for df_index, predictions_df in enumerate(predictions_dfs):
    predictions_df['forecast_date'] = pd.to_datetime(predictions_df['forecast_date'])

    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        predictions_df,
        left_index=True,
        right_on='forecast_date',
        how='inner'
    )
    print(combined_and_predictions)

    # Liste zum Speichern der Durchschnittslosses für jeden Quantil-Level
    average_losses = []

    # Iteriere durch die Quantile und Berechnung des linearen Quantile Scores
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'q{quantile_level}'  # Anpassung der Genauigkeit auf 3 Dezimalstellen
        combined_and_predictions[f'quantile_score_{quantile_level}_{df_index}'] = combined_and_predictions.apply(
            lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
            axis=1
        )

        # Berechne den Loss für jedes Quantil-Level und speichere es in der Liste
        average_loss = combined_and_predictions[f'quantile_score_{quantile_level}_{df_index}'].mean()
        average_losses.append(average_loss)

    # Berechne den Durchschnittsloss über alle Quantile
    overall_average_loss = sum(average_losses) / len(average_losses)

    # Gib die Ergebnisse aus
    print(f'Durchschnittlicher Loss für alle Quantile (DataFrame {df_index}): {overall_average_loss}')

    # Gib die Durchschnittslosses für jedes Quantil-Level aus
    for quantile_level, avg_loss in zip([0.025, 0.25, 0.5, 0.75, 0.975], average_losses):
        print(f'Durchschnittlicher Loss für Quantil {quantile_level} (DataFrame {df_index}): {avg_loss}')


#%%
'''
Gesamt Quantile über alle Date Frames und so mit allen Submissions
'''
def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        # Fall, in dem q_hat gleich y ist (Randfall)
        return 0

# Liste zum Speichern der Durchschnittslosses für jeden Quantil-Level
average_losses = []

# Liste zum Speichern der Zwischenergebnisse pro DataFrame
intermediate_results = []

# Iteriere durch alle DataFrames in der Liste predictions_dfs
for df_index, predictions_df in enumerate(predictions_dfs):
    predictions_df['forecast_date'] = pd.to_datetime(predictions_df['forecast_date'])

    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        predictions_df,
        left_index=True,
        right_on='forecast_date',
        how='inner'
    )
    print(combined_and_predictions)

    # Iteriere durch die Quantile und Berechnung des linearen Quantile Scores
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'q{quantile_level}'  # Anpassung der Genauigkeit auf 3 Dezimalstellen
        combined_and_predictions[f'quantile_score_{quantile_level}_{df_index}'] = combined_and_predictions.apply(
            lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
            axis=1
        )

    # Berechne den Loss für jedes Quantil-Level für diesen DataFrame und speichere es in der Liste
    df_average_losses = []
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        df_average_loss = combined_and_predictions[f'quantile_score_{quantile_level}_{df_index}'].mean()
        df_average_losses.append(df_average_loss)

    # Füge die Liste der Durchschnittslosses für diesen DataFrame zu der Gesamtliste hinzu
    average_losses.append(df_average_losses)

    # Speichere die Zwischenergebnisse pro DataFrame
    intermediate_results.append(combined_and_predictions)

# Berechne den Durchschnittsloss über alle Quantile für alle DataFrames
overall_average_losses = np.mean(average_losses, axis=0)

# Gib die Ergebnisse aus
print(f'Durchschnittlicher Loss für alle Quantile über alle DataFrames: {overall_average_losses}')

# Gib die Durchschnittslosses für jedes Quantil-Level über alle DataFrames aus
for quantile_level, avg_loss in zip([0.025, 0.25, 0.5, 0.75, 0.975], overall_average_losses):
    print(f'Durchschnittlicher Loss für Quantil {quantile_level} über alle DataFrames: {avg_loss}')

# Gib die Zwischenergebnisse pro DataFrame aus
for df_index, intermediate_result in enumerate(intermediate_results):
    intermediate_result.to_csv(f'intermediate_result_df_{df_index}.csv', index=False)




#%%
'''
---------
Sharpness
[95%]
[0.025 und 0.975]
---------
'''

def calculate_sharpness(quantiles):
    # Berechnen Sie die Differenzen zwischen oberen und unteren Quantilen
    interval_widths = quantiles[:, 1] - quantiles[:, 0]
    
    # Berechnen Sie den Durchschnitt der Intervallbreiten
    average_sharpness = np.mean(interval_widths)
    
    return average_sharpness

# Beispiel: Annahme, dass Sie eine Liste von Datensätzen (predictions_dfs) haben

all_sharpness_values = []

# Iterieren Sie über jeden Datensatz
for df_index, predictions_df in enumerate(predictions_dfs):
    predictions_df['forecast_date'] = pd.to_datetime(predictions_df['forecast_date'])

    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        predictions_df,
        left_index=True,
        right_on='forecast_date',
        how='inner'
    )

    # Quantile für einen bestimmten Datensatz
    quantiles = combined_and_predictions[['q0.025', 'q0.975']].values  # Ersetzen Sie 'q0.025' und 'q0.975' durch die entsprechenden Spaltennamen

    # Sharpness berechnen
    sharpness_value = calculate_sharpness(quantiles)

    # Speichern Sie den Sharpness-Wert für diesen Datensatz
    all_sharpness_values.append(sharpness_value)

    # Ausgabe des Sharpness-Werts
    print(f'Sharpness für Datensatz {df_index}: {sharpness_value:.5f}\n')

# Durchschnittliche Sharpness über alle Datensätze berechnen
overall_average_sharpness = np.mean(all_sharpness_values)

# Ausgabe des gesamten durchschnittlichen Sharpness-Werts
print(f'Gesamtdurchschnittliche Sharpness über alle Datensätze: {overall_average_sharpness:.5f}')


#%%

'''
Sharpness
[50%]
[75%, 25%]
'''
def calculate_sharpness(quantiles):
    # Berechnen Sie die Differenzen zwischen oberen und unteren Quantilen
    interval_widths = quantiles[:, 1] - quantiles[:, 0]
    
    # Berechnen Sie den Durchschnitt der Intervallbreiten
    average_sharpness = np.mean(interval_widths)
    
    return average_sharpness

# Beispiel: Annahme, dass Sie eine Liste von Datensätzen (predictions_dfs) haben

all_sharpness_values = []

# Iterieren Sie über jeden Datensatz
for df_index, predictions_df in enumerate(predictions_dfs):
    predictions_df['forecast_date'] = pd.to_datetime(predictions_df['forecast_date'])

    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        predictions_df,
        left_index=True,
        right_on='forecast_date',
        how='inner'
    )

    # Quantile für einen bestimmten Datensatz
    quantiles = combined_and_predictions[['q0.25', 'q0.75']].values  # Ersetzen Sie 'q0.025' und 'q0.975' durch die entsprechenden Spaltennamen

    # Sharpness berechnen
    sharpness_value = calculate_sharpness(quantiles)

    # Speichern Sie den Sharpness-Wert für diesen Datensatz
    all_sharpness_values.append(sharpness_value)

    # Ausgabe des Sharpness-Werts
    print(f'Sharpness für Datensatz {df_index}: {sharpness_value:.5f}\n')

# Durchschnittliche Sharpness über alle Datensätze berechnen
overall_average_sharpness = np.mean(all_sharpness_values)

# Ausgabe des gesamten durchschnittlichen Sharpness-Werts
print(f'Gesamtdurchschnittliche Sharpness über alle Datensätze: {overall_average_sharpness:.5f}')


#%%






















































scores_seasonal6 = np.array([quantile_score(q_hat, true_energy_6, tau) for q_hat in quant6_hat])
print("Quantile Scores für horizon h68:", scores_seasonal6)


