#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode

# %%
import ephem
from datetime import datetime, timedelta
import pytz
# =================================================================================================================================
#%%
def get_sunrise_sunset(latitude, longitude, date, timezone):
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.date = date

    sun = ephem.Sun(observer)
    
    sunrise = observer.previous_rising(sun).datetime()
    sunset = observer.next_setting(sun).datetime()

    # Konvertiere Zeiten in die angegebene Zeitzone
    timezone = pytz.timezone(timezone)
    sunrise = timezone.localize(sunrise)
    sunset = timezone.localize(sunset)

    return sunrise, sunset

# Beispiel für Edermünde => Geographischer Schwerpunkt von Deutschland
latitude = 51.2163421
longitude = 9.3989666
timezone = 'Europe/Berlin'

# Startdatum
start_date = datetime(2018, 12, 24)

# Enddatum
end_date = datetime(2024, 12, 31)

# Liste zum Speichern der Ergebnisse
sunrise_sunset_data = []

# Schleife durch alle Tage im Jahr
current_date = start_date
while current_date <= end_date:
    sunrise, sunset = get_sunrise_sunset(latitude, longitude, current_date, timezone)
    sunrise_sunset_data.append({
        'Date': current_date,
        'Sunrise': sunrise,
        'Sunset': sunset
    })
    current_date += timedelta(days=1)

# DataFrame erstellen
df_sunrise_sunset = pd.DataFrame(sunrise_sunset_data)

# Ergebnisse anzeigen
print(df_sunrise_sunset)
#%%
df_sunrise_sunset
#%%
from datetime import datetime

def calculate_sun_hours(row):
    sunrise_time = datetime.strptime(str(row['Sunrise']), '%Y-%m-%d %H:%M:%S.%f%z').time()
    sunset_time = datetime.strptime(str(row['Sunset']), '%Y-%m-%d %H:%M:%S.%f%z').time()

    daylight_duration = datetime.combine(datetime.min, sunset_time) - datetime.combine(datetime.min, sunrise_time)
    daylight_hours = daylight_duration.total_seconds() / 3600

    return round(daylight_hours, 0)

# Assuming df_sunrise_sunset['Sunrise'] and df_sunrise_sunset['Sunset'] are in the correct format

# Sonnenstunden berechnen und in neue Spalte einfügen
df_sunrise_sunset['DaylightHours'] = df_sunrise_sunset.apply(calculate_sun_hours, axis=1)
df_sunrise_sunset
#%%
print(df_sunrise_sunset)
#%%
df_sunrise_sunset['Sunrise'] = pd.to_datetime(df_sunrise_sunset['Sunrise'])
df_sunrise_sunset['Sunset'] = pd.to_datetime(df_sunrise_sunset['Sunset'])

# Extrahiere nur Stunden und Minuten
df_sunrise_sunset['Sunrise'] = df_sunrise_sunset['Sunrise'].dt.strftime('%H:%M')
df_sunrise_sunset['Sunset'] = df_sunrise_sunset['Sunset'].dt.strftime('%H:%M')
#%%
# Date als Index setzen
df_sunrise_sunset.set_index('Date', inplace=True)
#%%
df_sunrise_sunset
#%%

df_sunrise_sunset.dtypes
#%%
df_sunrise_sunset = df_sunrise_sunset.resample("H").ffill().reset_index().rename(
    {"Date": "Timestamp"}, axis=1
)
#%%
df_rise_set = df_sunrise_sunset.copy()
df_rise_set.set_index('Timestamp', inplace = True)
df_rise_set .index.names = ['date_time']
df_rise_set
#%%
# ===============================================================================================================================
#%%
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

data_energy = get_energy_data()
#%%
data_energy.head()
#%%
data_energy = data_energy.rename(columns={"Netzlast_Gesamt": "gesamt"})
#%%
data_energy['gesamt'] = data_energy['gesamt'] / 1000
#%%
data_energy.head()
#%%
# Beiden Datensätze zusammenfügen 
data_energy = pd.merge(df_rise_set, data_energy, on = 'date_time')
data_energy
#%%
data_energy['day of the week'] = data_energy.index.weekday
data_energy['weekday_String'] = data_energy.index.day_name()
# %%
data_energy['hour'] = data_energy.index.hour
# %%
data_energy['month'] = data_energy.index.month
# %%
data_energy.head()
# %%
data_energy['weekday'] = (data_energy.index.weekday < 5).astype(int)
data_energy['weekend'] = (data_energy.index.weekday > 5).astype(int)
#%%
from workalendar.europe import Germany
cal = Germany()
#%%

data_energy['Holiday'] = [date in cal.holidays(date.year) for date in data_energy.index]
data_energy.head()
#%%
data_energy['Holiday'] = data_energy['Holiday'].astype(int)
#%%
data_energy
#%%
data_energy['season'] = pd.cut(data_energy['month'], bins=[0, 2, 5, 8, 11, float('Inf')],
                               labels=['winter', 'spring', 'summer', 'autumn', 'winter'], ordered=False)
data_energy
#%%
data_energy['daytime'] = pd.cut(data_energy['hour'], bins=[-1, 6, 21, float('Inf')],
                               labels=['night', 'day', 'night'], ordered=False)
data_energy

#%%
# Dummy Variablen erstellen ---
# Baseline Variablen: Monday, January, hour00:00, 8 Sunhours ---
dummy_data_energy = pd.get_dummies(data_energy, columns=['day of the week', 'month', 'hour', 'DaylightHours', 'season', 'daytime'], drop_first=True, dtype=int, prefix=['weekday', 'month', 'hour', 'DaylightHours', 'season', 'daytime'])
#%%
dummy_data_energy
#%%
dummy_data_energy.columns
# =======================================================================================================================
#%%
import statsmodels.api as sm
np.random.seed(42)
#%%
dummy_data = dummy_data_energy.copy()
# %%
'''Hinzufügen einer Konstante für (ß0)'''
dummy_data['Konstante'] = 1
# =======================================================================================================================
#%%
start_date1 = '2021-01-01'
end_date1 = '2024-2-14'

data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date1)]
print(data_q1)
#%%
data_q1.columns
# %%
X = (data_q1[[
   # 'Holiday',
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
       'weekday_5', 'weekday_6',
        'month_2', 'month_3', 'month_4', 'month_5',
       'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11',
       'month_12',
         'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
       'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
       'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
       'DaylightHours_9.0', 'DaylightHours_10.0', 'DaylightHours_11.0',
       'DaylightHours_12.0', 'DaylightHours_13.0', 'DaylightHours_14.0',
       'DaylightHours_15.0', 'DaylightHours_16.0', 'DaylightHours_17.0',
       #'season_spring', 'season_summer', 
       #'season_winter', 
       #'daytime_night',
       'Konstante']])

y = data_q1['gesamt']
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
# %%
import statsmodels.api as sm
import pandas as pd
from statsmodels.tools import add_constant

def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.QuantReg(y, add_constant(X[included + [new_column]])).fit(q=0.5)
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        model = sm.QuantReg(y, add_constant(X[included])).fit(q=0.5)
        pvalues = model.pvalues
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
#%%
predictions_dict = {}
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

for quantile in quantiles:
    model1 = sm.QuantReg(y_train, X_train, random_seed = 42 ).fit(q=quantile)
    print(f"Quantile {quantile}:")
    print(model1.summary())
    predictions1 = model1.predict(X_test)
    print('-------------------')
    print(predictions1)
    predictions_dict[f'quantile_{quantile}'] = predictions1

#%%
predictions_model1 = pd.DataFrame(predictions_dict)
predictions_model1

#%%

'''
Quantile Model which only includes Energy data from November, December of 2023, 2022, 2021
'''
data_nov_dec_21_to_23 = dummy_data[(dummy_data.index.month.isin([11, 12])) & (dummy_data.index.year.isin([2020, 2021, 2022, 2023]))]
data_nov_dec_21_to_23

#%%
data_sub6 = data_nov_dec_21_to_23[data_nov_dec_21_to_23.index <= '2023-12-20']
data_sub6
#%%
X6 = (data_sub6[['weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
       'weekday_5', 'weekday_6', 'month_11','month_12',
         'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
       'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
       'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
       'DaylightHours_9.0', 'DaylightHours_10.0', 'DaylightHours_11.0',
       'DaylightHours_12.0', 'DaylightHours_13.0', 'DaylightHours_14.0',
       'DaylightHours_15.0', 'DaylightHours_16.0', 'DaylightHours_17.0',
        'season_winter', 'daytime_night',
       'Konstante']])

y6 = data_sub6['gesamt']
#%%
# %%

'''
---------
Pinball Loss
=> Für die Quantile Regression
---------
 '''
def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        # Fall, in dem q_hat gleich y ist (Randfall)
        return 0

# Annahme: y_test enthält die wahren Werte, predictions_model1 enthält die Vorhersagen für jedes Quantil
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pinball_losses = {}

for quantile in quantiles:
    predictions = predictions_model1[f'quantile_{quantile}']
    loss = np.mean([quantile_score(q_hat, y, quantile) for q_hat, y in zip(predictions, y_test)])
    pinball_losses[quantile] = loss
    print(f"Pinball Loss for Quantile {quantile}: {loss}")

# Berechne den durchschnittlichen Pinball Loss über alle Quantile
average_pinball_loss = np.mean(list(pinball_losses.values()))
print(f"Average Pinball Loss: {average_pinball_loss}")
# %%

# ======================================================================================================================================================
'''
---------------------------------
Random Forest Quantile Regression
---------------------------------
'''
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import mean_absolute_error
#%%
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
qrf = RandomForestQuantileRegressor(random_state = 42)

# %%
qrf.fit(X_train, y_train)
predictions_rfqt = qrf.predict(X_test, quantiles=[0.025,0.25,0.5, 0.75,0.975])
#%%
current_hyperparameters = qrf.get_params()

print("Aktuelle Hyperparameter:")
print(current_hyperparameters)
#%%
predictions_rfqt_df = pd.DataFrame(predictions_rfqt, columns=[f'quantile_{q}' for q in quantiles])

# Füge den Index von X_test zum DataFrame hinzu
predictions_rfqt_df['date_time'] = X_test.index
#%%
predictions_rfqt_df = predictions_rfqt_df.set_index('date_time')
#%%
'''
-----
QRF 
=> mit eigens gewählten Hyperparametern
----
'''
from sklearn.model_selection import train_test_split

# Setze den Zufallsseed für die Reproduzierbarkeit
random_seed = 42

# Teile die Daten in Trainings- und Testdaten auf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=random_seed)

# Definiere deine eigenen Hyperparameter
custom_hyperparameters = {
    'n_estimators': 15,
    'max_depth': 50,
    'random_state': random_seed,
}


custom_qrf = RandomForestQuantileRegressor(**custom_hyperparameters)


custom_qrf.fit(X_train, y_train)


predictions_custom_model = custom_qrf.predict(X_test, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975])

predictions_custom_model = pd.DataFrame(predictions_custom_model, columns=['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975'])
predictions_custom_model['date_time'] = X_test.index
predictions_custom_model = predictions_custom_model.set_index('date_time')


predictions_custom_model


#%%
from sklearn.model_selection import GridSearchCV

'''
-----------------------------------
Random Forest Quantile Regression
==================================
- Mehrere Bootstrapping Stichproben werden erstellt
- Für jeden der Stichproben wird ein Entscheidungsbaum erstellt
- 
-----------------------------------
Grid Search
n_estimators: Anzahld der Bäume
max_depth: maximale Tiefe jedes Entscheidungsbaumes
.....................................
Die Leistung wird mit Kreuzvalidierung geprüft => 5-Fold Cross-Validierung
Es werden X_train und y_train verwendet um die Kombination der Besten Hyperparameter zu finden
Best_estimator => Das Model das durch die besten gewählten Hyperparmater definiert ist
'''
start_date1 = '2021-01-01'
end_date1 = '2024-01-22'

data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date1)]
print(data_q1)
#%%
data_q1.columns
# %%
X = (data_q1[[
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
       'weekday_5', 'weekday_6',
        'month_2', 'month_3', 'month_4', 'month_5',
       'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11',
       'month_12',
         'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
       'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
       'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
       'DaylightHours_9.0', 'DaylightHours_10.0', 'DaylightHours_11.0',
       'DaylightHours_12.0', 'DaylightHours_13.0', 'DaylightHours_14.0',
       'DaylightHours_15.0', 'DaylightHours_16.0', 'DaylightHours_17.0',
       #'season_spring', 'season_summer', 
       #'season_winter', 
       #'daytime_night',
       'Konstante']])

y = data_q1['gesamt']
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

np.random.seed(42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [10, 15, 20],
    # Weitere Hyperparameter hier hinzufügen
}

grid_search = GridSearchCV(RandomForestQuantileRegressor(random_state = 42), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Verwende bestes Modell für Vorhersagen
predictions_best_model = best_model.predict(X_test, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975])
print("Beste Hyperparameter:")
print(grid_search.best_params_)
#%%
predictions_best_model = pd.DataFrame(predictions_best_model, columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975'] )
#%%

predictions_best_model['date_time'] = X_test.index
#%%
predictions_best_model = predictions_best_model.set_index('date_time')
predictions_best_model

#%%
from sklearn.metrics import quantile_loss

# Berechne den Quantile Loss für das beste Modell
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
quantile_loss_value = quantile_loss(y_test, predictions_best_model.values, quantiles=quantiles)
print("Quantile Loss für das beste Modell:", quantile_loss_value)

#%%

'''
-----
QRF 
=> mit eigens gewählten Hyperparametern nach Grid Search
----
'''
from sklearn.model_selection import train_test_split

# Setze den Zufallsseed für die Reproduzierbarkeit
random_seed = 42

# Teile die Daten in Trainings- und Testdaten auf
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=random_seed)

# Definiere deine eigenen Hyperparameter
custom_hyperparameters = {
    'n_estimators': 200,
    'max_depth': None,
    'random_state': random_seed,
}

# Erstelle ein benutzerdefiniertes QRF-Modell mit den festgelegten Hyperparametern
custom_qrf = RandomForestQuantileRegressor(**custom_hyperparameters)

# Trainiere das Modell mit den Trainingsdaten
#custom_qrf.fit(X_train, y_train)

custom_qrf.fit(X, y)


#%%

submission_deadline = datetime(2023, 11, 15, 00, 00)
#%%
df_horizon = X
df_horizon = df_horizon.reset_index()
#%%
df_horizon['date_time'] = pd.to_datetime(df_horizon['date_time'])
df_horizon['weekday'] = df_horizon['date_time'].dt.dayofweek
#%%
df_horizon.set_index("date_time", inplace=True)
df_horizon.head()
#%%
#horizons = [42, 46, 50, 66, 70, 74]
horizons = [61, 65, 69, 85, 89, 93]
LAST_IDX = -1
LAST_DATE = df_horizon.iloc[LAST_IDX].name


def get_date_from_horizon(last_ts, horizon):
    return last_ts + pd.DateOffset(hours=horizon)

#%%
horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
horizon_date
#%%
#hours_offset = [61, 65, 69, 85, 89, 93]
#horizons = [42, 46, 50, 66, 70, 74]
horizons = [61, 65, 69, 85, 89, 93]
#%%
tau = [0.025, 0.25, 0.5, 0.75, 0.975]
#%%
# leeres DataFrame für future_X
future_X = pd.DataFrame(index=range(len(horizon_date)), columns=X.columns)
future_X

#%%
for i, target_datetime in enumerate(horizon_date):
    # Extrahieren Sie die relevanten Features für den Zielzeitpunkt
    month = target_datetime.month
    hour = target_datetime.hour
    weekday = target_datetime.weekday()
    duration_sunlight = df_rise_set.loc[target_datetime, 'DaylightHours']

    is_9_sunlight = 1 if duration_sunlight in [9] else 0
    is_10_sunlight = 1 if duration_sunlight in [10] else 0
    is_11_sunlight = 1 if duration_sunlight in [11] else 0
    is_12_sunlight = 1 if duration_sunlight in [12] else 0
    is_13_sunlight = 1 if duration_sunlight in [13] else 0
    is_14_sunlight = 1 if duration_sunlight in [14] else 0
    is_15_sunlight = 1 if duration_sunlight in [15] else 0
    is_16_sunlight = 1 if duration_sunlight in [16] else 0
    is_17_sunlight = 1 if duration_sunlight in [17] else 0
    #is_spring = 1 if month in [3, 4, 5] else 0
    #is_summer = 1 if month in [6, 7, 8] else 0
    #is_winter = 1 if month in [12, 1, 2] else 0
    #is_daytime_night = 1 if hour in [22,23,1,2,3,4,5,6] else 0


    # Erstellen Sie den Feature-Vektor für die Vorhersage
    #input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight] + [is_spring, is_summer, is_winter,is_daytime_night ] + [1] 
    input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight]  + [1] 

    print(input_data)
    # Fügen Sie die Features zum DataFrame future_X hinzu
    future_X.loc[i, :] = input_data

#%%
#
predictions_custum_qrf = custom_qrf.predict(future_X, quantiles = [0.025, 0.25, 0.5, 0.75, 0.975])
predictions_custum_qrf




# Erstelle ein DataFrame für die Vorhersagen mit dem gleichen Zufallsseed
predictions_custum_qrf = pd.DataFrame(predictions_custum_qrf, columns=['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975'])
predictions_custum_qrf['date_time'] = future_X.index
predictions_custum_qrf= predictions_custum_qrf.set_index('date_time')

# Zeige die Ergebnisse an
predictions_custum_qrf
#%%
#%%
#hours_offset = [61, 65, 69, 85, 89, 93]
#horizons = [42, 46, 50, 66, 70, 74]
horizons = [61, 65, 69, 85, 89, 93]
#%%
tau = [0.025, 0.25, 0.5, 0.75, 0.975]
#%%
# leeres DataFrame für future_X
future_X = pd.DataFrame(index=range(len(horizon_date)), columns=X_full.columns)
future_X
#%%
future_timestamps = [target_datetime - pd.DateOffset(hours=lag) for lag in range(1, lag_periods + 1)]
df_future = pd.DataFrame(index=future_timestamps, columns=df_horizon.columns)
#%%
for i, target_datetime in enumerate(horizon_date):
    # Extrahieren Sie die relevanten Features für den Zielzeitpunkt
    month = target_datetime.month
    hour = target_datetime.hour
    weekday = target_datetime.weekday()
    duration_sunlight = df_rise_set.loc[target_datetime, 'DaylightHours']

    is_9_sunlight = 1 if duration_sunlight in [9] else 0
    is_10_sunlight = 1 if duration_sunlight in [10] else 0
    is_11_sunlight = 1 if duration_sunlight in [11] else 0
    is_12_sunlight = 1 if duration_sunlight in [12] else 0
    is_13_sunlight = 1 if duration_sunlight in [13] else 0
    is_14_sunlight = 1 if duration_sunlight in [14] else 0
    is_15_sunlight = 1 if duration_sunlight in [15] else 0
    is_16_sunlight = 1 if duration_sunlight in [16] else 0
    is_17_sunlight = 1 if duration_sunlight in [17] else 0
    is_spring = 1 if month in [3, 4, 5] else 0
    is_summer = 1 if month in [6, 7, 8] else 0
    is_winter = 1 if month in [12, 1, 2] else 0
    is_daytime_night = 1 if hour in [22,23,1,2,3,4,5,6] else 0


    lag_variables = [combined_y.loc[target_datetime - pd.DateOffset(hours=lag)] for lag in range(1, lag_periods + 1)]


    # Erstellen Sie den Feature-Vektor für die Vorhersage
    input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight] + [is_spring, is_summer, is_winter,is_daytime_night ] + [1] + lag_variables
    print(input_data)
    # Fügen Sie die Features zum DataFrame future_X hinzu
    future_X.loc[i, :] = input_data

#%%
predictions = rf_model.predict(future_X, quantiles = [0.025, 0.25, 0.5, 0.75, 0.975])
predictions


#%%
forecastdate = datetime(2024, 2, 7, 00, 00)
#%%
horizons2 = [36, 40, 44, 60, 64, 68]
df_sub_RandomForest_lagged = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "energy",
    "horizon": [str(h) + " hour" for h in horizons2],
    "q0.025": predictions[:,0],
    "q0.25": predictions[:,1],
    "q0.5": predictions[:,2],
    "q0.75": predictions[:,3],
    "q0.975": predictions[:,4],
})
df_sub_RandomForest_lagged


#%%
PATH = "/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/PTSFC Submission JahrMonatTag_Rachel"

df_sub_RandomForest_lagged.to_csv(PATH+ "Submission11_Energy_RandomForest_Lagged.csv", index=False)



#%%
'''
---------------------------------------------------------------------------------------------------------------------
'''
# %%

#%%
'''
-------------------------
Jetzt Vorhersagen für die Zukunft machen und dann den gesamten Datensatz verwenden
Freitag: 12:00, 16:00, 20:00
Samstag: 12:00, 16:00, 20:00
--------------------------
'''
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])
#%%
best_model.fit(X_full, y_full)

#%%
submission_deadline = datetime(2023, 12, 20, 00, 00)
#%%
df_horizon = X_full
df_horizon = df_horizon.reset_index()
#%%
df_horizon['date_time'] = pd.to_datetime(df_horizon['date_time'])
df_horizon['weekday'] = df_horizon['date_time'].dt.dayofweek
#%%
df_horizon.set_index("date_time", inplace=True)
df_horizon.head()
#%%
#horizons = [61, 65, 69, 85, 89, 93] # horizons wenn der letze Datenpunk der Dientag 23:00 ist
horizons = [42, 46, 50, 66, 70, 74]
LAST_IDX = -1
LAST_DATE = df_horizon.iloc[LAST_IDX].name


def get_date_from_horizon(last_ts, horizon):
    return last_ts + pd.DateOffset(hours=horizon)

#%%
horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
horizon_date
#%%
#hours_offset = [61, 65, 69, 85, 89, 93]
horizons = [42, 46, 50, 66, 70, 74]
#%%
tau = [0.025, 0.25, 0.5, 0.75, 0.975]
#%%
# leeres DataFrame für future_X
future_X = pd.DataFrame(index=range(len(horizon_date)), columns=X_full.columns)
future_X
#%%
for i, target_datetime in enumerate(horizon_date):
    # Extrahieren Sie die relevanten Features für den Zielzeitpunkt
    month = target_datetime.month
    hour = target_datetime.hour
    weekday = target_datetime.weekday()
    duration_sunlight = df_rise_set.loc[target_datetime, 'DaylightHours']

    is_9_sunlight = 1 if duration_sunlight in [9] else 0
    is_10_sunlight = 1 if duration_sunlight in [10] else 0
    is_11_sunlight = 1 if duration_sunlight in [11] else 0
    is_12_sunlight = 1 if duration_sunlight in [12] else 0
    is_13_sunlight = 1 if duration_sunlight in [13] else 0
    is_14_sunlight = 1 if duration_sunlight in [14] else 0
    is_15_sunlight = 1 if duration_sunlight in [15] else 0
    is_16_sunlight = 1 if duration_sunlight in [16] else 0
    is_17_sunlight = 1 if duration_sunlight in [17] else 0
    is_spring = 1 if month in [3, 4, 5] else 0
    is_summer = 1 if month in [6, 7, 8] else 0
    is_winter = 1 if month in [12, 1, 2] else 0
    is_daytime_night = 1 if hour in [22,23,1,2,3,4,5,6] else 0
    # Erstellen Sie den Feature-Vektor für die Vorhersage
    input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight] + [is_spring, is_summer, is_winter,is_daytime_night ] + [1]
    print(input_data)
    # Fügen Sie die Features zum DataFrame future_X hinzu
    future_X.loc[i, :] = input_data

#%%
future_X
future_X.index = horizon_date
future_X.index.name = 'date_time'
future_X
#%%
predictions_horizon = best_model.predict(future_X, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975])
predictions_horizon
#%%
predictions_horizon

#%%
forecastdate = datetime(2023, 12, 20, 00, 00)
#%%
horizons2 = [36, 40, 44, 60, 64, 68]
df_sub_RandomForest = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "energy",
    "horizon": [str(h) + " hour" for h in horizons2],
    "q0.025": predictions_horizon[:,0],
    "q0.25": predictions_horizon[:,1],
    "q0.5": predictions_horizon[:,2],
    "q0.75": predictions_horizon[:,3],
    "q0.975": predictions_horizon[:,4],
})
df_sub_RandomForest

#%%
PATH = "/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/PTSFC Submission JahrMonatTag_Rachel"

df_sub_RandomForest.to_csv(PATH+ "Submission6_Energy_RandomForest.csv", index=False)
#%%
'''
---------
Quantile Regression
----------
Vorhersagen für die 6 horizons mit der Quantilsregression machen
=> Die trainierte Quantilssregression basieren auf dem gesamten Datensatz
   um dan Vorhersagen zu machen für die 6 Zeitfenster
-------
'''

predictions_dict = {}
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

for quantile in quantiles:
    model1_forecast = sm.QuantReg(y, X, random_seed = 42).fit(q=quantile)
    print(f"Quantile {quantile}:")
    print(model1_forecast.summary())

#%%
future_X_qaunt_reg = pd.DataFrame(index=range(len(horizon_date)), columns=X.columns)
future_X_qaunt_reg
#%%
predictions_dict = {}
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

for i, target_datetime in enumerate(horizon_date):
    # Extrahieren Sie die relevanten Features für den Zielzeitpunkt
    month = target_datetime.month
    hour = target_datetime.hour
    weekday = target_datetime.weekday()
    duration_sunlight = df_rise_set.loc[target_datetime, 'DaylightHours']

    is_9_sunlight = 1 if duration_sunlight in [9] else 0
    is_10_sunlight = 1 if duration_sunlight in [10] else 0
    is_11_sunlight = 1 if duration_sunlight in [11] else 0
    is_12_sunlight = 1 if duration_sunlight in [12] else 0
    is_13_sunlight = 1 if duration_sunlight in [13] else 0
    is_14_sunlight = 1 if duration_sunlight in [14] else 0
    is_15_sunlight = 1 if duration_sunlight in [15] else 0
    is_16_sunlight = 1 if duration_sunlight in [16] else 0
    is_17_sunlight = 1 if duration_sunlight in [17] else 0
   # is_spring = 1 if month in [3, 4, 5] else 0
    #is_summer = 1 if month in [6, 7, 8] else 0
    #is_winter = 1 if month in [12, 1, 2] else 0
    #is_daytime_night = 1 if hour in [22,23,1,2,3,4,5,6] else 0
    # Erstellen Sie den Feature-Vektor für die Vorhersage
    #input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+1)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight] + [is_spring, is_summer, is_winter,is_daytime_night ] + [1]
    input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+1)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight]  + [1]

    #input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [1]

    print(input_data)
    # Fügen Sie die Features zum DataFrame future_X hinzu
    future_X_qaunt_reg.loc[i, :] = input_data



for quantile in quantiles:
    # Mache Vorhersagen für Quantilsregression
    model1_forecast = sm.QuantReg(y, X, random_seed = 42).fit(q = quantile)
    predictions_quant_reg = model1_forecast.predict(future_X_qaunt_reg)

    # Speichere Vorhersagen im Dictionary
    predictions_dict[f'quantile_{quantile}'] = predictions_quant_reg

#%%
forecast_quantreg = pd.DataFrame(predictions_dict)
forecast_quantreg 

#%%
forecastdate = datetime(2024, 2, 14, 00, 00)
horizons2 = [36, 40, 44, 60, 64, 68]
df_sub_quant = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "energy",
    "horizon": [str(h) + " hour" for h in horizons2],
    "q0.025": forecast_quantreg ['quantile_0.025'],
    "q0.25": forecast_quantreg ['quantile_0.25'],
    "q0.5":  forecast_quantreg ['quantile_0.5'],
    "q0.75": forecast_quantreg ['quantile_0.75'],
    "q0.975": forecast_quantreg ['quantile_0.975'],
})
df_sub_quant
#%%
PATH = "/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/PTSFC Submission JahrMonatTag_Rachel"

df_sub_quant.to_csv(PATH+ "Submission12_Energy_QaunttileReg.csv", index=False)
#%%

#%%
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Beispiel: Anzahl der Lags von 1 bis 10 testen
lag_values = range(1, 11)
cv_scores = []

for lag in lag_values:
    # Erstellen von Lag-Features
    lag_data = pd.concat([data.shift(i) for i in range(1, lag + 1)], axis=1)
    lag_data.columns = [f'lag_{i}' for i in range(1, lag + 1)]
    lag_data = lag_data.dropna()

    # Aufteilen der Daten in Trainings- und Validierungssets
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_index, valid_index in tscv.split(lag_data):
        train_data, valid_data = lag_data.iloc[train_index], lag_data.iloc[valid_index]
        y_train = lag_data.iloc[train_index]
        y_valid = lag_data.iloc[valid_index]

        # Modell erstellen und auf Trainingsdaten trainieren
        knn_model = KNeighborsRegressor(n_neighbors=k_neighbors)
        knn_model.fit(train_data, y_train)

        # Quantilschätzungen für Validierungsdaten durchführen
        quantile_predictions_valid = knn_model.predict(valid_data)

        # Berechnung des Pinball Loss
        pinball_loss = mean_absolute_error(y_valid, quantile_predictions_valid)
        scores.append(pinball_loss)

    # Durchschnittlicher Pinball Loss über alle Folds
    cv_scores.append(np.mean(scores))

# Wählen Sie die Anzahl der Lags mit dem niedrigsten durchschnittlichen Pinball Loss
best_lag = lag_values[np.argmin(cv_scores)]
print(f'Beste Anzahl von Lags: {best_lag}')
#%%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Beispiel Zeitreihendaten
data = data_energy['gesamt']

# Hyperparameter
k_neighbors = 5
lags = 1  # Anzahl der Lag-Features
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]  # Die gewünschten Quantile

# Erstellen von Lag-Features
lag_data = pd.concat([data.shift(i) for i in range(lags)], axis=1)
lag_data.columns = [f'lag_{i+1}' for i in range(lags)]
lag_data = lag_data.dropna()

# Aufteilen der Daten in Trainings- und Testsets
train_size = int(len(lag_data) * 0.8)  # 80% für das Training, 20% für den Test
train_data, test_data = lag_data.iloc[:train_size], lag_data.iloc[train_size:]
y_train, y_test = data.iloc[lags:train_size+lags], data.iloc[train_size+lags:]

# Modell erstellen und auf Trainingsdaten trainieren
knn_model = KNeighborsRegressor(n_neighbors=k_neighbors)
knn_model.fit(train_data, y_train)

# Quantilschätzungen für Testdaten durchführen
quantile_predictions_test = {q: knn_model.predict(test_data) + np.percentile(np.random.normal(size=len(test_data)), q * 100) for q in quantiles}

#%%
quantile_predictions_test 
#%%
quantile_predictions_laged_model = pd.DataFrame(quantile_predictions_test)
quantile_predictions_laged_model
#%%
quantile_predictions_laged_model['date_time'] = test_data.index
#%%
quantile_predictions_laged_model = quantile_predictions_laged_model.set_index('date_time')
quantile_predictions_laged_model
#%%
neue_spaltennamen = {'0.025': 'quantile_0.025', '0.25': 'quantile_0.25', '0.5': 'quantile_0.5', '0.75': 'quantile_0.75', '0.975': 'quantile_0.975'}

# DataFrame umbenennen und Präfix hinzufügen
quantile_predictions_laged_model = quantile_predictions_laged_model.rename(columns=neue_spaltennamen)
quantile_predictions_laged_model = quantile_predictions_laged_model.add_prefix('quantile_')
quantile_predictions_laged_model 

#%%
'''
-------------
Pinball Loss
für das Laged Model
------------
'''
y_true = test_data.values

# Pinball Loss für jedes Quantil berechnen
def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        # Fall, in dem q_hat gleich y ist (Randfall)
        return 0


quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pinball_losses = {}

for quantile in quantiles:
    predictions = quantile_predictions_laged_model[f'quantile_{quantile}']
    loss = np.mean([quantile_score(q_hat, y, quantile) for q_hat, y in zip(predictions, y_true)])
    pinball_losses[quantile] = loss
    print(f"Pinball Loss for Quantile {quantile}: {loss}")

# Berechne den durchschnittlichen Pinball Loss über alle Quantile
average_pinball_loss = np.mean(list(pinball_losses.values()))
print(f"Average Pinball Loss: {average_pinball_loss}")



#%%

'''
-----------------------------
QRF with 3 lagged Variables
-----------------------------
yt-1 (the value an hour before) => to get hourly seasonality
yt-24 (the value 24 hours before) => to get daily seasonality
yt-247 (the value one week before) => to get weekly seasonality
'''
start_date1 = '2021-01-01'
end_date1 = '2024-2-7'

data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date1)]
print(data_q1)
#%%

X = (data_q1[[
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
       'weekday_5', 'weekday_6',
        'month_2', 'month_3', 'month_4', 'month_5',
       'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11',
       'month_12',
         'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
       'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
       'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
       'DaylightHours_9.0', 'DaylightHours_10.0', 'DaylightHours_11.0',
       'DaylightHours_12.0', 'DaylightHours_13.0', 'DaylightHours_14.0',
       'DaylightHours_15.0', 'DaylightHours_16.0', 'DaylightHours_17.0',
       'season_spring', 'season_summer', 
       'season_winter', 
       'daytime_night',
       'Konstante']])

y = data_q1['gesamt']

#%%
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
rf_model = RandomForestQuantileRegressor(random_state=42)

X['gesamt_lag_1'] = y.shift(1)  # yt-1
X['gesamt_lag_24'] = y.shift(24)  # yt-24
X['gesamt_lag_247'] = y.shift(247)  # yt-247

X = X.dropna()
y = y[X.index] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Trainiere das Modell
rf_model.fit(X_train, y_train)

# Mache Vorhersagen auf den Testdaten
predictions = rf_model.predict(X_test, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975])
predictions_df = pd.DataFrame(predictions, columns=[f'quantile_{q}' for q in quantiles])
predictions_df['date_time'] = X_test.index

predictions_df = predictions_df.set_index('date_time')
predictions_df


#%%

'''
Die Lag Werte Schätzen
'''

submission_deadline = datetime(2024, 2, 7, 00, 00)

#%%
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])
#%%
df_horizon = X_full
df_horizon = df_horizon.reset_index()
#%%
df_horizon['date_time'] = pd.to_datetime(df_horizon['date_time'])
df_horizon['weekday'] = df_horizon['date_time'].dt.dayofweek
#%%
df_horizon.set_index("date_time", inplace=True)
df_horizon.head()
#%%
horizons = list(range(1,94))
LAST_IDX = -1
LAST_DATE = df_horizon.iloc[LAST_IDX].name


def get_date_from_horizon(last_ts, horizon):
    return last_ts + pd.DateOffset(hours=horizon)


#%%
horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
horizon_date
#%%
tau = [0.025, 0.25, 0.5, 0.75, 0.975]
#%%
future_X = pd.DataFrame(index=range(len(horizon_date)), columns=X_full.columns)
future_X
#%%
#X_linear_model = X_full.drop(columns=[f'gesamt_lag_{lag}' for lag in range(1, lag_periods + 1)])
#X_linear_model
lag_columns = ['gesamt_lag_1', 'gesamt_lag_24', 'gesamt_lag_247']

# Entferne die Lag-Spalten aus dem DataFrame X_full
X_linear_model = X_full.drop(columns=lag_columns)

# Zeige das aktualisierte DataFrame X_linear_model an
X_linear_model.head()

#%%
linear_model = sm.OLS(y_full, X_linear_model)
results_linear_model = linear_model.fit()

#%%
for i, target_datetime in enumerate(horizon_date):
    # Extrahieren Sie die relevanten Features für den Zielzeitpunkt
    weekday = target_datetime.weekday()
    month = target_datetime.month
    hour = target_datetime.hour
    duration_sunlight = df_rise_set.loc[target_datetime, 'DaylightHours']

    is_9_sunlight = 1 if duration_sunlight in [9] else 0
    is_10_sunlight = 1 if duration_sunlight in [10] else 0
    is_11_sunlight = 1 if duration_sunlight in [11] else 0
    is_12_sunlight = 1 if duration_sunlight in [12] else 0
    is_13_sunlight = 1 if duration_sunlight in [13] else 0
    is_14_sunlight = 1 if duration_sunlight in [14] else 0
    is_15_sunlight = 1 if duration_sunlight in [15] else 0
    is_16_sunlight = 1 if duration_sunlight in [16] else 0
    is_17_sunlight = 1 if duration_sunlight in [17] else 0
    is_spring = 1 if month in [3, 4, 5] else 0
    is_summer = 1 if month in [6, 7, 8] else 0
    is_winter = 1 if month in [12, 1, 2] else 0
    is_daytime_night = 1 if hour in [22,23,1,2,3,4,5,6] else 0

    lag_variables = [X.loc[target_datetime, column] for column in lag_columns]


    # Erstellen Sie den Feature-Vektor für die Vorhersage
    input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight] + [is_spring, is_summer, is_winter,is_daytime_night ] + [1] + lag_variables
    print(input_data)
    # Fügen Sie die Features zum DataFrame future_X hinzu
    future_X.loc[i, :] = input_data

#%%
future_X
future_X.index = horizon_date
future_X.index.name = 'date_time'
future_X
#%%
predictions_linear = results_linear_model.predict(future_X)
predictions_linear
#%%
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])
#%%
combined_y = pd.concat([y_full, predictions_linear], axis=0, ignore_index=False)
combined_y
#%%
rf_model.fit(X_full, y_full)
#%%
submission_deadline = datetime(2024, 2, 7, 00, 00)

df_horizon = X_full
df_horizon = df_horizon.reset_index()

df_horizon['date_time'] = pd.to_datetime(df_horizon['date_time'])
df_horizon['weekday'] = df_horizon['date_time'].dt.dayofweek

#%%
df_horizon.set_index("date_time", inplace=True)
df_horizon.head()
#%%
#horizons = [42, 46, 50, 66, 70, 74]
horizons = [61, 65, 69, 85, 89, 93]
LAST_IDX = -1
LAST_DATE = df_horizon.iloc[LAST_IDX].name


def get_date_from_horizon(last_ts, horizon):
    return last_ts + pd.DateOffset(hours=horizon)

#%%    

horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
horizon_date

#%%
horizons = [61, 65, 69, 85, 89, 93]
#%%
tau = [0.025, 0.25, 0.5, 0.75, 0.975]
#%%
future_X = pd.DataFrame(index=range(len(horizon_date)), columns=X_full.columns)
future_X
#%%
lag_periods = 3

# future_timestamps erstellen, eine Liste von zukünftigen Zeitstempeln basierend auf den Lag-Werten
future_timestamps = [target_datetime - pd.DateOffset(hours=lag) for lag in [1, 24, 247]]

# Ein leeres DataFrame df_future erstellen, das die gleichen Spalten wie df_horizon hat, aber Zeilen für die zukünftigen Zeitpunkte enthält
df_future = pd.DataFrame(index=future_timestamps, columns=df_horizon.columns)

# df_future anzeigen
df_future

#%%
for i, target_datetime in enumerate(horizon_date):
    # Extrahieren Sie die relevanten Features für den Zielzeitpunkt
    month = target_datetime.month
    hour = target_datetime.hour
    weekday = target_datetime.weekday()
    duration_sunlight = df_rise_set.loc[target_datetime, 'DaylightHours']

    is_9_sunlight = 1 if duration_sunlight in [9] else 0
    is_10_sunlight = 1 if duration_sunlight in [10] else 0
    is_11_sunlight = 1 if duration_sunlight in [11] else 0
    is_12_sunlight = 1 if duration_sunlight in [12] else 0
    is_13_sunlight = 1 if duration_sunlight in [13] else 0
    is_14_sunlight = 1 if duration_sunlight in [14] else 0
    is_15_sunlight = 1 if duration_sunlight in [15] else 0
    is_16_sunlight = 1 if duration_sunlight in [16] else 0
    is_17_sunlight = 1 if duration_sunlight in [17] else 0
    is_spring = 1 if month in [3, 4, 5] else 0
    is_summer = 1 if month in [6, 7, 8] else 0
    is_winter = 1 if month in [12, 1, 2] else 0
    is_daytime_night = 1 if hour in [22,23,1,2,3,4,5,6] else 0


    lag_variables = [X.loc[target_datetime, column] for column in lag_columns]



    # Erstellen Sie den Feature-Vektor für die Vorhersage
    input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight] + [is_spring, is_summer, is_winter,is_daytime_night ] + [1] + lag_variables
    print(input_data)
    # Fügen Sie die Features zum DataFrame future_X hinzu
    future_X.loc[i, :] = input_data
#%%
predictions = rf_model.predict(future_X, quantiles = [0.025, 0.25, 0.5, 0.75, 0.975])
predictions
#%%    



#%%
'''
------------------------------
Gradient Boosting Regressor

-----------------------------
'''
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error

all_models = {}
common_params = dict(
    learning_rate=0.1,
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=9,
    min_samples_split=13,
)
#%%
for alpha in [0.025, 0.25 ,0.5, 0.75 ,0.975]:
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, **common_params)
    all_models["quantile_%1.3f" % alpha] = gbr.fit(X_train, y_train)

#%%
gbr_ls = GradientBoostingRegressor(loss="squared_error", **common_params)
all_models["mse"] = gbr_ls.fit(X_train, y_train)
#%%
# Vorhersagen für Quantilregressionsmodelle
quantile_predictions = {}
for key, model in all_models.items():
    if "q" in key:
        quantile_predictions[key] = model.predict(X_test)

# Vorhersagen für das MSE-Modell (mean squared error)
mse_predictions = all_models["mse"].predict(X_test)
#%%
quantile_predictions_GBR = pd.DataFrame(quantile_predictions)
quantile_predictions_GBR
#%%
quantile_predictions_GBR['date_time'] = X_test.index
#%%
quantile_predictions_GBR = quantile_predictions_GBR.set_index('date_time')
quantile_predictions_GBR
#%%
quantile_predictions_GBR
#%%
quantile_predictions_GBR.columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75','quantile_0.975']
#%%
quantile_predictions_GBR
#%%
'''
---------------------------------------------------------------------------------------------
                                        |Evaluation|
---------------------------------------------------------------------------------------------
'''

# %%
'''
-----------------
Pinball Loss 
=> Für Random Forest Quantile Regression
------------------
'''

quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pinball_losses = {}

for quantile in quantiles:
    predictions = quantile_predictions_GBR[f'quantile_{quantile}']
    loss = np.mean([quantile_score(q_hat, y, quantile) for q_hat, y in zip(predictions, y_test)])
    pinball_losses[quantile] = loss
    print(f"Pinball Loss for Quantile {quantile}: {loss}")

# Berechne den durchschnittlichen Pinball Loss über alle Quantile
average_pinball_loss = np.mean(list(pinball_losses.values()))
print(f"Average Pinball Loss: {average_pinball_loss}")


#%%

quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pinball_losses = {}

for quantile in quantiles:
    predictions = predictions_rfqt_df[f'quantile_{quantile}']
    loss = np.mean([quantile_score(q_hat, y, quantile) for q_hat, y in zip(predictions, y_test)])
    pinball_losses[quantile] = loss
    print(f"Pinball Loss for Quantile {quantile}: {loss}")

# Berechne den durchschnittlichen Pinball Loss über alle Quantile
average_pinball_loss = np.mean(list(pinball_losses.values()))
print(f"Average Pinball Loss: {average_pinball_loss}")
#%%

quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pinball_losses = {}

for quantile in quantiles:
    predictions = predictions_best_model[f'quantile_{quantile}']
    loss = np.mean([quantile_score(q_hat, y, quantile) for q_hat, y in zip(predictions, y_test)])
    pinball_losses[quantile] = loss
    print(f"Pinball Loss for Quantile {quantile}: {loss}")

# Berechne den durchschnittlichen Pinball Loss über alle Quantile
average_pinball_loss = np.mean(list(pinball_losses.values()))
print(f"Average Pinball Loss: {average_pinball_loss}")

#%%
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pinball_losses = {}

for quantile in quantiles:
    predictions = predictions_custom_model [f'quantile_{quantile}']
    loss = np.mean([quantile_score(q_hat, y, quantile) for q_hat, y in zip(predictions, y_test)])
    pinball_losses[quantile] = loss
    print(f"Pinball Loss for Quantile {quantile}: {loss}")

# Berechne den durchschnittlichen Pinball Loss über alle Quantile
average_pinball_loss = np.mean(list(pinball_losses.values()))
print(f"Average Pinball Loss: {average_pinball_loss}")



#%%

'''
----------------
Pinball Loss
=> Random Forest Quantile Regerssion with lagged value
----------------
'''
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pinball_losses = {}

for quantile in quantiles:
    predictions = predictions_df[f'quantile_{quantile}']
    loss = np.mean([quantile_score(q_hat, y, quantile) for q_hat, y in zip(predictions, y_test)])
    pinball_losses[quantile] = loss
    print(f"Pinball Loss for Quantile {quantile}: {loss}")

# Berechne den durchschnittlichen Pinball Loss über alle Quantile
average_pinball_loss = np.mean(list(pinball_losses.values()))
print(f"Average Pinball Loss: {average_pinball_loss}")
# %%

'''
----------------------------
PLOT DER QUANTILS REGRESSION 
Die Letzten 80 Tage aus y_test werden geplottet gegen die geschätzen Quantile
----------------------------
'''
import matplotlib.pyplot as plt



# Annahme: y_test enthält die wahren Werte, predictions_model1 enthält die Vorhersagen für jedes Quantil
quantile_columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']

# Plotten der Vorhersagen und Testdaten
plt.figure(figsize=(15, 8))

# Begrenzen Sie die Anzahl der Datenpunkte (z.B., die letzten 30)
num_points = 60

# Plot für jeden Quantil separat erstellen
for i, quantile_column in enumerate(quantile_columns):
    plt.plot(predictions_model1.index[-num_points:], predictions_model1[quantile_column].values[-num_points:],
             label=f'{quantile_column}', alpha=0.7, marker='o')

# Scatterplot für den echten Wert (Testdaten) hinzufügen
plt.plot(y_test.index[-num_points:], y_test.values[-num_points:], label='True Energy (Test Data)', color='black', marker='x')

# Hinzufügen von Beschriftungen und Legende
plt.xlabel('Date')
plt.ylabel('Energy demand')
plt.grid(True, alpha=0.5)
plt.gca().set_facecolor('whitesmoke')
custom_legend_labels = ['0.025 Quantile', '0.25 Quantile', '0.5 Quantile', '0.75 Quantile', '0.975 Quantile', 'True Energy']
plt.legend(labels=custom_legend_labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')  # Legende anzeigen und anpassen
plt.title('Quantile Forecasts and the True Energy demand')
plt.show()
# %%

'''
-----------------------------------------
PLOT DES RANDOM FOREST QUANTILE REGRESSION
Die Letzten 80 Tage aus y_test werden geplottet gegen die geschätzen Quantile
-------------------------------------------
'''
import matplotlib.pyplot as plt



# Annahme: y_test enthält die wahren Werte, predictions_model1 enthält die Vorhersagen für jedes Quantil
quantile_columns = ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']

# Plotten der Vorhersagen und Testdaten
plt.figure(figsize=(15, 8))

# Begrenzen Sie die Anzahl der Datenpunkte (z.B., die letzten 30)
num_points = 60

# Plot für jeden Quantil separat erstellen
for i, quantile_column in enumerate(quantile_columns):
    plt.plot(predictions_rfqt_df.index[-num_points:], predictions_rfqt_df[quantile_column].values[-num_points:],
             label=f'{quantile_column}', alpha=0.7, marker='o')

# Scatterplot für den echten Wert (Testdaten) hinzufügen
plt.plot(y_test.index[-num_points:], y_test.values[-num_points:], label='True Energy (Test Data)', color='black', marker='x')

# Hinzufügen von Beschriftungen und Legende
plt.xlabel('Date')
plt.ylabel('Energy demand')
plt.grid(True, alpha=0.5)
plt.gca().set_facecolor('whitesmoke')
custom_legend_labels = ['0.025 Quantile', '0.25 Quantile', '0.5 Quantile', '0.75 Quantile', '0.975 Quantile', 'True Energy']
plt.legend(labels=custom_legend_labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')  # Legende anzeigen und anpassen
plt.title('Quantile Forecasts and the True Energy demand')
plt.show()
# %%


# %%




#%%
'''
==============================================
Interval Score berechnen
=================================================
Interval Score for central (1-alpha)x100% PI
(Gneiting and Katzfuss, 2014)
'''
#alpha = 0.05
alpha = 0.5
def interval_score(l, r, y, alpha):
    """
    Berechnet den Interval Score für ein Vorhersageintervall.

    Parameter:
    - l: linke Grenze des Vorhersageintervalls
    - r: rechte Grenze des Vorhersageintervalls
    - y: tatsächlicher Wert
    - alpha: Signifikanzniveau

    Rückgabewert:
    - Interval Score nach der gegebenen Formel
    """
    indicator_left = 1 if y < l else 0
    indicator_right = 1 if y > r else 0

    score = (r - l) + (alpha / 2) * (l - y) * indicator_left + (alpha / 2) * (y - r) * indicator_right
    return score


interval_scores = []

for timestamp in predictions_best_model.index:
    y = y_test.loc[timestamp]
    #l = predictions_model1.loc[timestamp, 'quantile_0.25']
    #r = predictions_model1.loc[timestamp, 'quantile_0.75']
    l = predictions_model1.loc[timestamp, 'quantile_0.25']
    r = predictions_model1.loc[timestamp, 'quantile_0.75']
    
    score = interval_score(l, r, y, alpha)
    interval_scores.append(score)

average_interval_score = sum(interval_scores) / len(interval_scores)
print(f"Durchschnittlicher Interval Score über alle Zeitstempel: {average_interval_score}")




#%%
def interval_score(
    observations,
    alpha,
    q_dict=None,
    q_left=None,
    q_right=None,
    percent=False,
    check_consistency=True,
):
    """
    Compute interval scores (1) for an array of observations and predicted intervals.
    
    Either a dictionary with the respective (alpha/2) and (1-(alpha/2)) quantiles via q_dict needs to be
    specified or the quantiles need to be specified via q_left and q_right.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alpha : numeric
        Alpha level for (1-alpha) interval.
    q_dict : dict, optional
        Dictionary with predicted quantiles for all instances in `observations`.
    q_left : array_like, optional
        Predicted (alpha/2)-quantiles for all instances in `observations`.
    q_right : array_like, optional
        Predicted (1-(alpha/2))-quantiles for all instances in `observations`.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield a percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total interval scores.
    sharpness : array_like
        Sharpness component of interval scores.
    calibration : array_like
        Calibration component of interval scores.
        
    (1) Gneiting, T. and A. E. Raftery (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102(477), 359–378.    
    """

    if q_dict is None:
        if q_left is None or q_right is None:
            raise ValueError(
                "Either quantile dictionary or left and right quantile must be supplied."
            )
    else:
        if q_left is not None or q_right is not None:
            raise ValueError(
                "Either quantile dictionary OR left and right quantile must be supplied, not both."
            )
        q_left = q_dict.get(alpha / 2)
        if q_left is None:
            raise ValueError(f"Quantile dictionary does not include {alpha/2}-quantile")

        q_right = q_dict.get(1 - (alpha / 2))
        if q_right is None:
            raise ValueError(
                f"Quantile dictionary does not include {1-(alpha/2)}-quantile"
            )

    if check_consistency and np.any(q_left > q_right):
        raise ValueError("Left quantile must be smaller than right quantile.")

    sharpness = q_right - q_left
    calibration = (
        (
            np.clip(q_left - observations, a_min=0, a_max=None)
            + np.clip(observations - q_right, a_min=0, a_max=None)
        )
        * 2
        / alpha
    )
    if percent:
        sharpness = sharpness / np.abs(observations)
        calibration = calibration / np.abs(observations)
    total = sharpness + calibration
    return total, sharpness, calibration

#%%
'''
1) Total interval scores.
2) Sharpness: component of interval scores.
3) Calibration component of interval scores.
-------------------------------------------
Beste Random Forest Quantile Regression
'''
#%%
alpha_test = 0.5
#%%
# The true Observations
observations_test = np.array([y_test])
#%%
# quantile Forecasts
quantile_dict_test = {
    0.025: np.array(predictions_best_model['quantile_0.025']),
    0.25: np.array(predictions_best_model['quantile_0.25']),
    0.5: np.array(predictions_best_model['quantile_0.5']),
    0.75: np.array(predictions_best_model['quantile_0.75']),
    0.975: np.array(predictions_best_model['quantile_0.975'])
}

observations_test = np.array([y_test])
#%%
observations_test = np.delete(observations_test, [932, 933])
#%%
observations_test = np.delete(observations_test, [931,932, 933, 934, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 957, 958, 970, 971, 972, 973] )
#%%
scores90 = interval_score(observations_test,alpha_test,q_left=quantile_dict_test[0.025],q_right=quantile_dict_test[0.975])
#%%
scores50 = interval_score(observations_test,alpha_test,q_left=quantile_dict_test[0.25],q_right=quantile_dict_test[0.75])
#%%
mean_scores90 = np.mean(scores90[0])
mean_scores90
#%%
mean_scores50 = np.mean(scores50[0])
mean_scores50
#%%

def outside_interval(observations, lower, upper, check_consistency=True):
    """
    Indicate whether observations are outside a predicted interval for an array of observations and predicted intervals.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    lower : array_like, optional
        Predicted lower interval boundary for all instances in `observations`.
    upper : array_like, optional
        Predicted upper interval boundary for all instances in `observations`.
    check_consistency: bool, optional
        If `True`, interval boundaries are checked for consistency. Default is `True`.
        
    Returns
    -------
    Out : array_like
        Array of zeroes (False) and ones (True) counting the number of times observations where outside the interval.
    """
    if check_consistency and np.any(lower > upper):
        raise ValueError("Lower border must be smaller than upper border.")

    return ((lower > observations) + (upper < observations)).astype(int)


#%%
print(outside_interval(observations_test,lower=quantile_dict_test[0.25],upper=quantile_dict_test[0.75]))
#%%
outside1 = outside_interval(observations_test,lower=quantile_dict_test[0.25],upper=quantile_dict_test[0.75])
#%%
'''
Funktion die für die Forecast Evaluation
=> Gibt die wahren y basierend auf den horizon h und dem forecast_date heraus
'''
forecast_date = pd.to_datetime('2024-01-10 00:00:00')  # Hier dein tatsächliches Prognosedatum einfügen
horizon = [36, 40, 44, 60, 64, 68]

# Extrahiere den Wert für jedes Zeitfenster nach dem Prognosedatum
stromverbrauch_true = {}

for h in horizon:
    target_datetime = forecast_date + pd.Timedelta(hours=h)
    stromverbrauch_true[h] = data_energy.loc[target_datetime, 'gesamt']
    stromverbrauch_true_df = pd.DataFrame(list(stromverbrauch_true.items()), columns=['horizon', 'gesamt'])
    stromverbrauch_true_df.set_index('horizon', inplace=True)
print(stromverbrauch_true)

#%%
'''
------------------
Pinball Loss
-----------------
'''

true_energy_1 = stromverbrauch_true_df['gesamt'].iloc[0]
true_energy_1
#%%
true_energy_2 = stromverbrauch_true_df['gesamt'].iloc[1]
true_energy_2
# %%
true_energy_3 = stromverbrauch_true_df['gesamt'].iloc[2]
true_energy_3
# %%
true_energy_4 = stromverbrauch_true_df['gesamt'].iloc[3]
true_energy_4
# %%
true_energy_5 = stromverbrauch_true_df['gesamt'].iloc[4]
true_energy_5
#%%
true_energy_6 = stromverbrauch_true_df['gesamt'].iloc[5]
true_energy_6
#%%
y_true = [true_energy_1, true_energy_2, true_energy_3, true_energy_4, true_energy_5, true_energy_6]
y_true
#%%

'''
---------
Pinball Loss
=> Für die Quantile Regression
---------
 '''
def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        # Fall, in dem q_hat gleich y ist (Randfall)
        return 0

# Annahme: y_test enthält die wahren Werte, predictions_model1 enthält die Vorhersagen für jedes Quantil
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
pinball_losses = {}

for quantile in quantiles:
    #predictions = df_sub_quant[f'q{quantile}']
    #predictions = df_sub_RandomForest_lagged[f'q{quantile}']
    predictions = predictions_custum_qrf[f'quantile_{quantile}']
    loss = np.mean([quantile_score(q_hat, y, quantile) for q_hat, y in zip(predictions, y_true)])
    pinball_losses[quantile] = loss
    print(f"Pinball Loss for Quantile {quantile}: {loss}")

# Berechne den durchschnittlichen Pinball Loss über alle Quantile
average_pinball_loss = np.mean(list(pinball_losses.values()))
print(f"Average Pinball Loss: {average_pinball_loss}")

# %%

'''
=========================================================================================================================
'''


'''
---------
QUANTILE MODEL 1
---------
X:
- Konstante
- weekdays
- months
- hours
--------
'''


start_date1 = '2021-01-01'
end_date1 = '2023-11-15'
submission_deadline = datetime(2023, 11, 15, 0, 0)

data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date1)]

num_weeks = 9
all_forecasts = []

for week_num in range(0,num_weeks + 1):
    end_date = pd.to_datetime(submission_deadline) + pd.DateOffset(weeks=week_num)
    new_submission_deadline = end_date + pd.DateOffset(hours=0)  # Assuming 0 hour after end date
    print(f"\nIteration {week_num}:")
    print("End Date:", end_date)
    print("Submission Deadline:", new_submission_deadline)


    data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date)]

    X = data_q1[['weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
                 'weekday_5', 'weekday_6',
                 'month_2', 'month_3', 'month_4', 'month_5',
                 'month_6', 'month_7', 'month_8',
                 'month_9', 'month_10', 'month_11',
                 'month_12',
                 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
                 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
                 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
                 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                 'Konstante']]
    y = data_q1['gesamt']

    predictions_dict = {}
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    for quantile in quantiles:
        model1_forecast = sm.QuantReg(y, X, random_seed = 42).fit(q=quantile)
        print(f"Quantile {quantile}:")
        print(model1_forecast.summary())

    # Quantile Prediction
    horizons = [61, 65, 69, 85, 89, 93]
    LAST_IDX = -1
    df_horizon = data_energy.copy()
    df_horizon['weekday'] = df_horizon.index.weekday
    df_horizon = df_horizon.loc[df_horizon.index < end_date]
    LAST_DATE = df_horizon.iloc[LAST_IDX].name

    def get_date_from_horizon(last_ts, horizon):
        return last_ts + pd.DateOffset(hours=horizon)   

    horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
    print(horizon_date)


    future_X_qaunt_reg = pd.DataFrame(index=range(len(horizon_date)), columns=X.columns)
    print(future_X_qaunt_reg)

    predictions_dict = {}
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975] 

    for i, target_datetime in enumerate(horizon_date):
        weekday = target_datetime.weekday()
        month = target_datetime.month
        hour = target_datetime.hour

        input_data = [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+1)) for i in range(11)] + [int(hour == (i+1)) for i in range(23)] +[1] 

        future_X_qaunt_reg.loc[i, :] = input_data
        #print(future_X_qaunt_reg)

    for quantile in quantiles:
        model1_forecast = sm.QuantReg(y, X, random_seed = 42).fit(q = quantile)
        predictions_quant_reg = model1_forecast.predict(future_X_qaunt_reg)
        predictions_dict[f'quantile_{quantile}'] = predictions_quant_reg
        
        forecast_quantreg = pd.DataFrame(predictions_dict)
        forecast_quantreg['forecast_date'] = horizon_date
        forecast_quantreg.set_index('forecast_date', inplace=True)

    

    print(f"Quantile Predictions for Week {week_num}:")
    print(forecast_quantreg)
    all_forecasts.append(forecast_quantreg)



# %%
'''
------------------------------------------------------------------------
Die Wahren y_trues heraus extrahieren
auch automatisch für die Submission deadlines
----------------------------------------------------------------------
'''    

data_e = data_energy.copy()
# %%
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
# %%

# Eine Kopie von der Liste mit den enthaltenen Data Frames erzeugen
import copy

all_predictions_quant = copy.deepcopy(all_forecasts)
#%%

import pandas as pd

def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        return 0

# Liste zum Speichern des Durchschnittsverlusts pro Quantil und pro Horizon
average_losses_by_quantile_and_horizon = []

# Iteration durch jeden DataFrame
for df_index, forecast_quantreg in enumerate(all_predictions_quant):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions= pd.merge(
        combined_df_sorted,
        forecast_quantreg,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Iteration durch jeden Quantil-Level
    average_losses_by_horizon = []
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'quantile_{quantile_level}'
        quantile_losses_by_horizon = []
        for horizon in combined_and_predictions['horizon'].unique():
            horizon_quantile_losses = combined_and_predictions.loc[combined_and_predictions['horizon'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()
            quantile_losses_by_horizon.append(horizon_quantile_losses)
        average_losses_by_horizon.append(quantile_losses_by_horizon)
    average_losses_by_quantile_and_horizon.append(average_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon für jeden DataFrame
for df_index, average_losses_by_horizon in enumerate(average_losses_by_quantile_and_horizon):
    for quantile_index, quantile_losses_by_horizon in enumerate(average_losses_by_horizon):
        for horizon_index, horizon_loss in enumerate(quantile_losses_by_horizon):
            print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1} (DataFrame {df_index}): {horizon_loss}')


#%%
import numpy as np

# Liste zum Speichern der Durchschnittsverluste pro Quantil und pro Horizon über alle DataFrames
overall_average_losses_by_quantile_and_horizon = []

# Iteration durch jeden Quantil-Level
for quantile_index in range(len(average_losses_by_quantile_and_horizon[0])):
    overall_losses_by_horizon = []
    # Iteration durch jeden Horizon
    for horizon_index in range(len(average_losses_by_quantile_and_horizon[0][0])):
        horizon_losses = []
        # Iteration durch jeden DataFrame
        for df_losses in average_losses_by_quantile_and_horizon:
            horizon_losses.append(df_losses[quantile_index][horizon_index])
        # Berechnung des Durchschnitts für diesen Horizon über alle DataFrames
        overall_horizon_loss = np.mean(horizon_losses)
        overall_losses_by_horizon.append(overall_horizon_loss)
    overall_average_losses_by_quantile_and_horizon.append(overall_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon über alle DataFrames
for quantile_index, losses_by_horizon in enumerate(overall_average_losses_by_quantile_and_horizon):
    for horizon_index, loss in enumerate(losses_by_horizon):
        print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1}: {loss}')
#%%
# Liste zum Speichern des Durchschnittsverlusts pro DataFrame
average_losses_by_dataframe = []

# Iteration durch jeden DataFrame
for df_index, forecast_quantreg in enumerate(all_predictions_quant):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions= pd.merge(
        combined_df_sorted,
        forecast_quantreg,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Gesamtverlust pro DataFrame initialisieren
    total_loss = 0

    # Iteration durch jeden Quantil-Level
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'quantile_{quantile_level}'
        
        # Iteration durch jeden Horizont
        for horizon in combined_and_predictions['horizon'].unique():
            # Verlust pro Quantil und Horizont berechnen und zum Gesamtverlust addieren
            total_loss += combined_and_predictions.loc[combined_and_predictions['horizon'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()

    # Durchschnittsverlust pro DataFrame berechnen und zur Liste hinzufügen
    average_loss = total_loss / (len([0.025, 0.25, 0.5, 0.75, 0.975]) * len(combined_and_predictions['horizon'].unique()))
    average_losses_by_dataframe.append(average_loss)

# Ausgabe des Durchschnittsverlusts pro DataFrame
for df_index, average_loss in enumerate(average_losses_by_dataframe):
    print(f'Durchschnittlicher Loss für DataFrame {df_index}: {average_loss}')

#%%

   
'''
-------------------
      Intervel Score 
      Model 1
-------------------       
        
'''
alpha_test = 0.05

# Liste zum Speichern der Interval Scores für jedes DataFrame
interval_scores = []

for df_index, forecast_quantreg in enumerate(all_predictions_quant):
    # Merge basierend auf dem Datetime-Index
    true_predicted= pd.merge(
        combined_df_sorted,
        forecast_quantreg,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(true_predicted)

    # Wahren Werte (observations)
    true_values = true_predicted['gesamt']

    # Geschätzte Quantile (q_dict)
    quantile_dict = {
        0.025: true_predicted['quantile_0.025'],
        0.5: true_predicted['quantile_0.5'],
        0.975: true_predicted['quantile_0.975']
    }

    # Interval Score berechnen
    total_score, sharpness_score, calibration_score = interval_score(true_values, alpha_test, q_dict=quantile_dict)

    # Ergebnisse speichern
    interval_scores.append((total_score, sharpness_score, calibration_score))

# Ergebnisse anzeigen
for df_index, scores in enumerate(interval_scores):
    total_score, sharpness_score, calibration_score = scores
    print(f"DataFrame {df_index}:")
    print(f"Total Interval Score: {total_score}")
    print(f"Sharpness Score: {sharpness_score}")
    print(f"Calibration Score: {calibration_score}")

#%%

alpha_test = 0.05

# Liste zum Speichern der Interval Scores für jeden Horizon und DataFrame
interval_scores_by_horizon = []

for df_index, forecast_quantreg in enumerate(all_predictions_quant):
    # Merge basierend auf dem Datetime-Index
    true_predicted= pd.merge(
        combined_df_sorted,
        forecast_quantreg,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(true_predicted)

    # Iteriere durch die Horizonte
    for horizon in true_predicted['horizon'].unique():
        # Filtere die Daten für den aktuellen Horizont
        data_for_horizon = true_predicted[true_predicted['horizon'] == horizon]

        # Wahren Werte (observations)
        true_values = data_for_horizon['gesamt']

        # Geschätzte Quantile (q_dict)
        quantile_dict = {
            0.025: data_for_horizon['quantile_0.025'],
            0.5: data_for_horizon['quantile_0.5'],
            0.975: data_for_horizon['quantile_0.975']
        }

        # Interval Score berechnen
        total_score, sharpness_score, calibration_score = interval_score(true_values, alpha_test, q_dict=quantile_dict)

        # Ergebnisse speichern
        interval_scores_by_horizon.append((horizon, df_index, total_score, sharpness_score, calibration_score))

# Ergebnisse anzeigen
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    print(f"DataFrame {df_index}, Horizon {horizon}:")
    print(f"Total Interval Score: {total_score}")
    print(f"Sharpness Score: {sharpness_score}")
    print(f"Calibration Score: {calibration_score}")   

#%%
from collections import defaultdict

# Dictionary zum Speichern der Interval Scores für jedes Horizon über alle DataFrames
average_scores_by_horizon = defaultdict(list)

# Durchschnitt der Interval Scores für jedes Horizon über alle DataFrames berechnen
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    average_scores_by_horizon[horizon].append(total_score.values[0])


print("Durchschnittliche Scores für jedes Horizon über alle DataFrames:")
for horizon, scores in average_scores_by_horizon.items():
    average_score = sum(scores) / len(scores)
    print(f"Horizon {horizon}: {average_score}")


#%%
import numpy as np

# Liste zum Speichern der Sharpness Scores für jedes Horizon und DataFrame
sharpness_scores_by_horizon = []

# Iteration über alle Einträge in der Liste der Interval Scores
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    sharpness_scores_by_horizon.append((horizon, df_index, sharpness_score.values[0]))

# Umrechnung der Liste in einen DataFrame für die einfache Berechnung des Durchschnitts
sharpness_scores_df = pd.DataFrame(sharpness_scores_by_horizon, columns=['horizon', 'df_index', 'sharpness_score'])

# Berechnung des Durchschnitts der Sharpness-Scores für jedes Horizon über alle DataFrames
average_sharpness_scores_by_horizon = sharpness_scores_df.groupby('horizon')['sharpness_score'].mean()

# Ergebnisse ausgeben
print("Durchschnittliche Sharpness-Scores für jedes Horizon über alle DataFrames:")
print(average_sharpness_scores_by_horizon)


#%%

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
============================================================================================================================================
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
'''
---------
QUANTILE MODEL 2
---------
X:
- Konstante
- weekdays
- months
- hours
- Sunshine Duration
--------
'''


start_date1 = '2021-01-01'
end_date1 = '2023-11-15'
submission_deadline = datetime(2023, 11, 15, 0, 0)

data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date1)]


num_weeks = 9
all_forecasts_2 = []

for week_num in range(0,num_weeks + 1):
    end_date = pd.to_datetime(submission_deadline) + pd.DateOffset(weeks=week_num)
    new_submission_deadline = end_date + pd.DateOffset(hours=0)  # Assuming 0 hour after end date
    print(f"\nIteration {week_num}:")
    print("End Date:", end_date)
    print("Submission Deadline:", new_submission_deadline)


    data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date)]

    X = data_q1[['weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
                 'weekday_5', 'weekday_6',
                 'month_2', 'month_3', 'month_4', 'month_5',
                 'month_6', 'month_7', 'month_8',
                 'month_9', 'month_10', 'month_11',
                 'month_12',
                 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
                 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
                 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
                 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                 'DaylightHours_9.0', 'DaylightHours_10.0', 'DaylightHours_11.0',
                'DaylightHours_12.0', 'DaylightHours_13.0', 'DaylightHours_14.0',
                'DaylightHours_15.0', 'DaylightHours_16.0', 'DaylightHours_17.0',
                #'season_spring', 'season_summer', 
                #'season_winter', 
                #'daytime_night',
                'Konstante']]
    y = data_q1['gesamt']

    predictions_dict2 = {}
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    for quantile in quantiles:
        model1_forecast = sm.QuantReg(y, X, random_seed = 42).fit(q=quantile)
        print(f"Quantile {quantile}:")
        #print(model1_forecast.summary())

    # Quantile Prediction
    horizons = [61, 65, 69, 85, 89, 93]
    LAST_IDX = -1
    df_horizon = data_energy.copy()
    df_horizon['weekday'] = df_horizon.index.weekday
    df_horizon = df_horizon.loc[df_horizon.index < end_date]
    LAST_DATE = df_horizon.iloc[LAST_IDX].name

    def get_date_from_horizon(last_ts, horizon):
        return last_ts + pd.DateOffset(hours=horizon)   

    horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
    print(horizon_date)


    future_X_qaunt_reg = pd.DataFrame(index=range(len(horizon_date)), columns=X.columns)
    print(future_X_qaunt_reg)

    predictions_dict2 = {}
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975] 

    for i, target_datetime in enumerate(horizon_date):
        weekday = target_datetime.weekday()
        month = target_datetime.month
        hour = target_datetime.hour
        duration_sunlight = df_rise_set.loc[target_datetime, 'DaylightHours']

        is_9_sunlight = 1 if duration_sunlight in [9] else 0
        is_10_sunlight = 1 if duration_sunlight in [10] else 0
        is_11_sunlight = 1 if duration_sunlight in [11] else 0
        is_12_sunlight = 1 if duration_sunlight in [12] else 0
        is_13_sunlight = 1 if duration_sunlight in [13] else 0
        is_14_sunlight = 1 if duration_sunlight in [14] else 0
        is_15_sunlight = 1 if duration_sunlight in [15] else 0
        is_16_sunlight = 1 if duration_sunlight in [16] else 0
        is_17_sunlight = 1 if duration_sunlight in [17] else 0
        #is_spring = 1 if month in [3, 4, 5] else 0
        #is_summer = 1 if month in [6, 7, 8] else 0
        #is_winter = 1 if month in [12, 1, 2] else 0
        #is_daytime_night = 1 if hour in [22,23,1,2,3,4,5,6] else 0

        input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight]  + [1]

        #input_data = [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+1)) for i in range(11)] + [int(hour == (i+1)) for i in range(23)] +[1] 

        future_X_qaunt_reg.loc[i, :] = input_data
        #print(future_X_qaunt_reg)

    for quantile in quantiles:
        model1_forecast = sm.QuantReg(y, X, random_seed = 42).fit(q = quantile)
        predictions_quant_reg = model1_forecast.predict(future_X_qaunt_reg)
        predictions_dict2[f'quantile_{quantile}'] = predictions_quant_reg
        
        forecast_quantreg2 = pd.DataFrame(predictions_dict2)
        forecast_quantreg2['forecast_date'] = horizon_date
        forecast_quantreg2.set_index('forecast_date', inplace=True)
        print(forecast_quantreg2)

    

    print(f"Quantile Predictions for Week {week_num}:")
    print(forecast_quantreg2)
    all_forecasts_2.append(forecast_quantreg2)





#%%
 
def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        return 0

# Liste zum Speichern des Durchschnittsverlusts pro Quantil und pro Horizon
average_losses_by_quantile_and_horizon = []

# Iteration durch jeden DataFrame
for df_index, forecast_quantreg2 in enumerate(all_forecasts_2):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        forecast_quantreg2,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Iteration durch jeden Quantil-Level
    average_losses_by_horizon = []
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'quantile_{quantile_level}'
        quantile_losses_by_horizon = []
        for horizon in combined_and_predictions['horizon'].unique():
            horizon_quantile_losses = combined_and_predictions.loc[combined_and_predictions['horizon'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()
            quantile_losses_by_horizon.append(horizon_quantile_losses)
        average_losses_by_horizon.append(quantile_losses_by_horizon)
    average_losses_by_quantile_and_horizon.append(average_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon für jeden DataFrame
for df_index, average_losses_by_horizon in enumerate(average_losses_by_quantile_and_horizon):
    for quantile_index, quantile_losses_by_horizon in enumerate(average_losses_by_horizon):
        for horizon_index, horizon_loss in enumerate(quantile_losses_by_horizon):
            print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1} (DataFrame {df_index}): {horizon_loss}')


#%%
overall_average_losses_by_quantile_and_horizon = []

# Iteration durch jeden Quantil-Level
for quantile_index in range(len(average_losses_by_quantile_and_horizon[0])):
    overall_losses_by_horizon = []
    # Iteration durch jeden Horizon
    for horizon_index in range(len(average_losses_by_quantile_and_horizon[0][0])):
        horizon_losses = []
        # Iteration durch jeden DataFrame
        for df_losses in average_losses_by_quantile_and_horizon:
            horizon_losses.append(df_losses[quantile_index][horizon_index])
        # Berechnung des Durchschnitts für diesen Horizon über alle DataFrames
        overall_horizon_loss = np.mean(horizon_losses)
        overall_losses_by_horizon.append(overall_horizon_loss)
    overall_average_losses_by_quantile_and_horizon.append(overall_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon über alle DataFrames
for quantile_index, losses_by_horizon in enumerate(overall_average_losses_by_quantile_and_horizon):
    for horizon_index, loss in enumerate(losses_by_horizon):
        print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1}: {loss}')

#%%
# Liste zum Speichern des Durchschnittsverlusts pro DataFrame
average_losses_by_dataframe = []

# Iteration durch jeden DataFrame
for df_index, forecast_quantreg2 in enumerate(all_forecasts_2):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        forecast_quantreg2,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )

    # Gesamtverlust pro DataFrame initialisieren
    total_loss = 0

    # Iteration durch jeden Quantil-Level
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'quantile_{quantile_level}'
        
        # Iteration durch jeden Horizont
        for horizon in combined_and_predictions['horizon'].unique():
            # Verlust pro Quantil und Horizont berechnen und zum Gesamtverlust addieren
            total_loss += combined_and_predictions.loc[combined_and_predictions['horizon'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()

    # Durchschnittsverlust pro DataFrame berechnen und zur Liste hinzufügen
    average_loss = total_loss / (len([0.025, 0.25, 0.5, 0.75, 0.975]) * len(combined_and_predictions['horizon'].unique()))
    average_losses_by_dataframe.append(average_loss)

# Ausgabe des Durchschnittsverlusts pro DataFrame
for df_index, average_loss in enumerate(average_losses_by_dataframe):
    print(f'Durchschnittlicher Loss für DataFrame {df_index}: {average_loss}')




# %%
'''
----------------
Quantile Score 
-----------------
für das Quantile Regression Model 2
------------------
Hier werden die 8 letzten Submissions simuliert
für jede der Submissions werden die Quantile Scores berechnet und am Ende der Durchschnitt
AV QS
2.5%, 25%, 50%, 75%, 97.5%

'''


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

for df_index, forecast_quantreg2 in enumerate(all_forecasts_2):
    # Merge basierend auf dem Datetime-Index
    true_predicted2= pd.merge(
        combined_df_sorted,
        forecast_quantreg2,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(true_predicted2)

    # Iteriere durch die Quantile und Berechnung des linearen Quantile Scores
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'quantile_{quantile_level}'  # Anpassung der Genauigkeit auf 3 Dezimalstellen
        true_predicted2[f'quantile_score_{quantile_level}_{df_index}'] = true_predicted2.apply(
            lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
            axis=1
        )

    # Berechne den Loss für jedes Quantil-Level für diesen DataFrame und speichere es in der Liste
    df_average_losses = []
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        df_average_loss = true_predicted2[f'quantile_score_{quantile_level}_{df_index}'].mean()
        df_average_losses.append(df_average_loss)

    # Füge die Liste der Durchschnittslosses für diesen DataFrame zu der Gesamtliste hinzu
    average_losses.append(df_average_losses)

    # Speichere die Zwischenergebnisse pro DataFrame
    intermediate_results.append(true_predicted2)

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
# %%

'''
-------------------
      INtervel Score 
      Quantile Model 2
-------------------       
        
'''
alpha_test = 0.5

# Liste zum Speichern der Interval Scores für jedes DataFrame
interval_scores = []

for df_index, forecast_quantreg2 in enumerate(all_forecasts_2):
    # Merge basierend auf dem Datetime-Index
    true_predicted2= pd.merge(
        combined_df_sorted,
        forecast_quantreg2,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(true_predicted2)

    # Wahren Werte (observations)
    true_values = true_predicted2['gesamt']

    # Geschätzte Quantile (q_dict)
    quantile_dict = {
        0.25: true_predicted2['quantile_0.25'],
        0.5: true_predicted2['quantile_0.5'],
        0.75: true_predicted2['quantile_0.75']
    }

    # Interval Score berechnen
    total_score, sharpness_score, calibration_score = interval_score(true_values, alpha_test, q_dict=quantile_dict)

    # Ergebnisse speichern
    interval_scores.append((total_score, sharpness_score, calibration_score))

# Ergebnisse anzeigen
for df_index, scores in enumerate(interval_scores):
    total_score, sharpness_score, calibration_score = scores
    print(f"DataFrame {df_index}:")
    print(f"Total Interval Score: {total_score}")
    print(f"Sharpness Score: {sharpness_score}")
    print(f"Calibration Score: {calibration_score}")


#%%
'''
    Interval Score
    Quantil Model 2
'''
alpha_test = 0.5

# Liste zum Speichern der Interval Scores für jeden Horizon und DataFrame
interval_scores_by_horizon = []

for df_index, forecast_quantreg2 in enumerate(all_forecasts_2):
    # Merge basierend auf dem Datetime-Index
    true_predicted2= pd.merge(
        combined_df_sorted,
        forecast_quantreg2,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(true_predicted2)

    # Iteriere durch die Horizonte
    for horizon in true_predicted2['horizon'].unique():
        # Filtere die Daten für den aktuellen Horizont
        data_for_horizon = true_predicted2[true_predicted2['horizon'] == horizon]

        # Wahren Werte (observations)
        true_values = data_for_horizon['gesamt']

        # Geschätzte Quantile (q_dict)
        quantile_dict = {
            0.25: data_for_horizon['quantile_0.25'],
            0.5: data_for_horizon['quantile_0.5'],
            0.75: data_for_horizon['quantile_0.75']
        }

        # Interval Score berechnen
        total_score, sharpness_score, calibration_score = interval_score(true_values, alpha_test, q_dict=quantile_dict)

        # Ergebnisse speichern
        interval_scores_by_horizon.append((horizon, df_index, total_score, sharpness_score, calibration_score))

# Ergebnisse anzeigen
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    print(f"DataFrame {df_index}, Horizon {horizon}:")
    print(f"Total Interval Score: {total_score}")
    print(f"Sharpness Score: {sharpness_score}")
    print(f"Calibration Score: {calibration_score}") 



#%%

from collections import defaultdict

# Dictionary zum Speichern der Interval Scores für jedes Horizon über alle DataFrames
average_scores_by_horizon = defaultdict(list)

# Durchschnitt der Interval Scores für jedes Horizon über alle DataFrames berechnen
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    average_scores_by_horizon[horizon].append(total_score.values[0])

# Ergebnisse anzeigen
print("Durchschnittliche Scores für jedes Horizon über alle DataFrames:")
for horizon, scores in average_scores_by_horizon.items():
    average_score = sum(scores) / len(scores)
    print(f"Horizon {horizon}: {average_score}")

#%%
import numpy as np

# Liste zum Speichern der Sharpness Scores für jedes Horizon und DataFrame
sharpness_scores_by_horizon = []

# Iteration über alle Einträge in der Liste der Interval Scores
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    sharpness_scores_by_horizon.append((horizon, df_index, sharpness_score.values[0]))

# Umrechnung der Liste in einen DataFrame für die einfache Berechnung des Durchschnitts
sharpness_scores_df = pd.DataFrame(sharpness_scores_by_horizon, columns=['horizon', 'df_index', 'sharpness_score'])

# Berechnung des Durchschnitts der Sharpness-Scores für jedes Horizon über alle DataFrames
average_sharpness_scores_by_horizon = sharpness_scores_df.groupby('horizon')['sharpness_score'].mean()

# Ergebnisse ausgeben
print("Durchschnittliche Sharpness-Scores für jedes Horizon über alle DataFrames:")
print(average_sharpness_scores_by_horizon)


#%%



'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
============================================================================================================================================
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
'''
---------
RANDOM FOREST QUANTILE REGRESSION
---------
X:
- Konstante
- weekdays
- months
- hours
- Sunshine Duration
--------
'''
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import mean_absolute_error

quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
rf_model = RandomForestQuantileRegressor(random_state = 42)

import pandas as pd
from datetime import datetime

random_seed = 42
# Definiere deine eigenen Hyperparameter
custom_hyperparameters = {
    'n_estimators': 300,
    #'max_depth': None,
    'min_samples_leaf':10,
    'random_state': random_seed,
}
start_date1 = '2021-01-01'
end_date1 = '2023-11-15'
# Erstelle ein benutzerdefiniertes QRF-Modell mit den festgelegten Hyperparametern
custom_qrf = RandomForestQuantileRegressor(**custom_hyperparameters)
dummy_data = dummy_data_energy.copy()
data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date1)]


submission_deadline = datetime(2023, 11, 15, 0, 0)

num_weeks = 9
all_forecasts_qrf = []

for week_num in range(0, num_weeks + 1):
    end_date = pd.to_datetime(submission_deadline) + pd.DateOffset(weeks=week_num)
    print(end_date)
    new_submission_deadline = end_date + pd.DateOffset(hours=0)  # Assuming 0 hour after end date
    print(new_submission_deadline)
    data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date)]
    #print(data_q1)
    
    # Trainieren Sie Ihr Modell mit data_q1
    X = data_q1[['weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
                 'weekday_5', 'weekday_6',
                 'month_2', 'month_3', 'month_4', 'month_5',
                 'month_6', 'month_7', 'month_8',
                 'month_9', 'month_10', 'month_11',
                 'month_12',
                 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
                 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
                 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
                 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                 'DaylightHours_9.0', 'DaylightHours_10.0', 'DaylightHours_11.0',
                'DaylightHours_12.0', 'DaylightHours_13.0', 'DaylightHours_14.0',
                'DaylightHours_15.0', 'DaylightHours_16.0', 'DaylightHours_17.0']]
    y = data_q1['gesamt']
    print(y.tail())
    custom_qrf.fit(X, y)
    
    horizons = [61, 65, 69, 85, 89, 93]
    LAST_IDX = -1
    df_horizon = data_energy.copy()
    df_horizon['weekday'] = df_horizon.index.weekday
    df_horizon = df_horizon.loc[df_horizon.index < end_date]
    LAST_DATE = df_horizon.iloc[LAST_IDX].name

    def get_date_from_horizon(last_ts, horizon):
        return last_ts + pd.DateOffset(hours=horizon)   

    horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
    print(horizon_date)     

    # Generierung von Vorhersagen für diese Woche
    #horizon_date = [get_date_from_horizon(end_date, h) for h in horizons]


    #print(horizon_date)
    future_X = pd.DataFrame(index=range(len(horizon_date)), columns=X.columns)
    for i, target_datetime in enumerate(horizon_date):
        weekday = target_datetime.weekday()
        month = target_datetime.month
        hour = target_datetime.hour
        duration_sunlight = df_rise_set.loc[target_datetime, 'DaylightHours']

        is_9_sunlight = 1 if duration_sunlight in [9] else 0
        is_10_sunlight = 1 if duration_sunlight in [10] else 0
        is_11_sunlight = 1 if duration_sunlight in [11] else 0
        is_12_sunlight = 1 if duration_sunlight in [12] else 0
        is_13_sunlight = 1 if duration_sunlight in [13] else 0
        is_14_sunlight = 1 if duration_sunlight in [14] else 0
        is_15_sunlight = 1 if duration_sunlight in [15] else 0
        is_16_sunlight = 1 if duration_sunlight in [16] else 0
        is_17_sunlight = 1 if duration_sunlight in [17] else 0

        input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight]  
        future_X.loc[i, :] = input_data
        print(input_data)
    
    # Vorhersagen treffen
    predictions_custom_qrf = custom_qrf.predict(future_X, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975])
    
    # Ergebnisse für diese Woche speichern
    forecasts_qrf = pd.DataFrame(predictions_custom_qrf, columns=['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975'])
    forecasts_qrf['forecast_date'] = horizon_date
    forecasts_qrf = forecasts_qrf.set_index('forecast_date')
    
    print(forecasts_qrf)
    
    # Alle Vorhersagen speichern
    all_forecasts_qrf.append(forecasts_qrf)
    
    print(f"\nIteration {week_num}:")
    print("End Date:", end_date)
    print("Submission Deadline:", new_submission_deadline)

# %%
'''
Quantile Loss
'''

import pandas as pd

def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        return 0

# Liste zum Speichern des Durchschnittsverlusts pro Quantil und pro Horizon
average_losses_by_quantile_and_horizon = []

# Iteration durch jeden DataFrame
for df_index, forecasts_qrf in enumerate(all_forecasts_qrf):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions= pd.merge(
        combined_df_sorted,
        forecasts_qrf,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Iteration durch jeden Quantil-Level
    average_losses_by_horizon = []
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'quantile_{quantile_level}'
        quantile_losses_by_horizon = []
        for horizon in combined_and_predictions['horizon'].unique():
            horizon_quantile_losses = combined_and_predictions.loc[combined_and_predictions['horizon'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()
            quantile_losses_by_horizon.append(horizon_quantile_losses)
        average_losses_by_horizon.append(quantile_losses_by_horizon)
    average_losses_by_quantile_and_horizon.append(average_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon für jeden DataFrame
for df_index, average_losses_by_horizon in enumerate(average_losses_by_quantile_and_horizon):
    for quantile_index, quantile_losses_by_horizon in enumerate(average_losses_by_horizon):
        for horizon_index, horizon_loss in enumerate(quantile_losses_by_horizon):
            print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1} (DataFrame {df_index}): {horizon_loss}')


#%%


import numpy as np

# Liste zum Speichern der Durchschnittsverluste pro Quantil und pro Horizon über alle DataFrames
overall_average_losses_by_quantile_and_horizon = []

# Iteration durch jeden Quantil-Level
for quantile_index in range(len(average_losses_by_quantile_and_horizon[0])):
    overall_losses_by_horizon = []
    # Iteration durch jeden Horizon
    for horizon_index in range(len(average_losses_by_quantile_and_horizon[0][0])):
        horizon_losses = []
        # Iteration durch jeden DataFrame
        for df_losses in average_losses_by_quantile_and_horizon:
            horizon_losses.append(df_losses[quantile_index][horizon_index])
        # Berechnung des Durchschnitts für diesen Horizon über alle DataFrames
        overall_horizon_loss = np.mean(horizon_losses)
        overall_losses_by_horizon.append(overall_horizon_loss)
    overall_average_losses_by_quantile_and_horizon.append(overall_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon über alle DataFrames
for quantile_index, losses_by_horizon in enumerate(overall_average_losses_by_quantile_and_horizon):
    for horizon_index, loss in enumerate(losses_by_horizon):
        print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1}: {loss}')


#%%
# Liste zum Speichern des Durchschnittsverlusts pro DataFrame
average_losses_by_dataframe = []

# Iteration durch jeden DataFrame
for df_index, forecasts_qrf in enumerate(all_forecasts_qrf):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions= pd.merge(
        combined_df_sorted,
        forecasts_qrf,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Gesamtverlust pro DataFrame initialisieren
    total_loss = 0

    # Iteration durch jeden Quantil-Level
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'quantile_{quantile_level}'
        
        # Iteration durch jeden Horizont
        for horizon in combined_and_predictions['horizon'].unique():
            # Verlust pro Quantil und Horizont berechnen und zum Gesamtverlust addieren
            total_loss += combined_and_predictions.loc[combined_and_predictions['horizon'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()

    # Durchschnittsverlust pro DataFrame berechnen und zur Liste hinzufügen
    average_loss = total_loss / (len([0.025, 0.25, 0.5, 0.75, 0.975]) * len(combined_and_predictions['horizon'].unique()))
    average_losses_by_dataframe.append(average_loss)

# Ausgabe des Durchschnittsverlusts pro DataFrame
for df_index, average_loss in enumerate(average_losses_by_dataframe):
    print(f'Durchschnittlicher Loss für DataFrame {df_index}: {average_loss}')


#%%

#%%
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

for df_index, forecasts_qrf in enumerate(all_forecasts_qrf):
    # Merge basierend auf dem Datetime-Index
    true_predicted_qrf= pd.merge(
        combined_df_sorted,
        forecasts_qrf,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(true_predicted_qrf)
    # Iteriere durch die Quantile und Berechnung des linearen Quantile Scores
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'quantile_{quantile_level}'  # Anpassung der Genauigkeit auf 3 Dezimalstellen
        true_predicted_qrf[f'quantile_score_{quantile_level}_{df_index}'] = true_predicted_qrf.apply(
            lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
            axis=1
        )

    # Berechne den Loss für jedes Quantil-Level für diesen DataFrame und speichere es in der Liste
    df_average_losses = []
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        df_average_loss = true_predicted_qrf[f'quantile_score_{quantile_level}_{df_index}'].mean()
        df_average_losses.append(df_average_loss)

    # Füge die Liste der Durchschnittslosses für diesen DataFrame zu der Gesamtliste hinzu
    average_losses.append(df_average_losses)

    # Speichere die Zwischenergebnisse pro DataFrame
    intermediate_results.append(true_predicted_qrf)

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
      INtervel Score  
        
'''
alpha_test = 0.5

# Liste zum Speichern der Interval Scores für jedes DataFrame
interval_scores = []

for df_index, forecasts_qrf in enumerate(all_forecasts_qrf):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        forecasts_qrf,
        left_index=True,
        right_index=True,
        how='inner'
    )

    # Wahren Werte (observations)
    true_values = combined_and_predictions['gesamt']

    # Geschätzte Quantile (q_dict)
    quantile_dict = {
        0.25: combined_and_predictions['quantile_0.25'],
        0.5: combined_and_predictions['quantile_0.5'],
        0.75: combined_and_predictions['quantile_0.75']
    }

    # Interval Score berechnen
    total_score, sharpness_score, calibration_score = interval_score(true_values, alpha_test, q_dict=quantile_dict)

    # Ergebnisse speichern
    interval_scores.append((total_score, sharpness_score, calibration_score))

# Ergebnisse anzeigen
for df_index, scores in enumerate(interval_scores):
    total_score, sharpness_score, calibration_score = scores
    print(f"DataFrame {df_index}:")
    print(f"Total Interval Score: {total_score}")
    print(f"Sharpness Score: {sharpness_score}")
    print(f"Calibration Score: {calibration_score}")








#%%

alpha_test = 0.5

# Liste zum Speichern der Interval Scores für jeden Horizon und DataFrame
interval_scores_by_horizon = []

for df_index, forecasts_qrf in enumerate(all_forecasts_qrf):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        forecasts_qrf,
        left_index=True,
        right_index=True,
        how='inner'
    )

    # Iteriere durch die Horizonte
    for horizon in combined_and_predictions['horizon'].unique():
        # Filtere die Daten für den aktuellen Horizont
        data_for_horizon = combined_and_predictions[combined_and_predictions['horizon'] == horizon]

        # Wahren Werte (observations)
        true_values = data_for_horizon['gesamt']

        # Geschätzte Quantile (q_dict)
        quantile_dict = {
            0.025: data_for_horizon['quantile_0.25'],
            0.5: data_for_horizon['quantile_0.5'],
            0.975: data_for_horizon['quantile_0.75']
        }

        # Interval Score berechnen
        total_score, sharpness_score, calibration_score = interval_score(true_values, alpha_test, q_dict=quantile_dict)

        # Ergebnisse speichern
        interval_scores_by_horizon.append((horizon, df_index, total_score, sharpness_score, calibration_score))

# Ergebnisse anzeigen
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    print(f"DataFrame {df_index}, Horizon {horizon}:")
    print(f"Total Interval Score: {total_score}")
    print(f"Sharpness Score: {sharpness_score}")
    print(f"Calibration Score: {calibration_score}")   

#%%
alpha_test = 0.05

# Liste zum Speichern der Interval Scores für jeden Horizon und DataFrame
interval_scores_by_horizon = []

for df_index, forecasts_qrf in enumerate(all_forecasts_qrf):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        forecasts_qrf,
        left_index=True,
        right_index=True,
        how='inner'
    )

    # Iteriere durch die Horizonte
    for horizon in combined_and_predictions['horizon'].unique():
        # Filtere die Daten für den aktuellen Horizont
        data_for_horizon = combined_and_predictions[combined_and_predictions['horizon'] == horizon]

        # Wahren Werte (observations)
        true_values = data_for_horizon['gesamt']

        # Geschätzte Quantile (q_dict)
        quantile_dict = {
            0.025: data_for_horizon['quantile_0.025'],
            0.5: data_for_horizon['quantile_0.5'],
            0.975: data_for_horizon['quantile_0.975']
        }

        # Interval Score berechnen
        total_score, sharpness_score, calibration_score = interval_score(true_values, alpha_test, q_dict=quantile_dict)

        # Ergebnisse speichern
        interval_scores_by_horizon.append((horizon, df_index, total_score, sharpness_score, calibration_score))

# Ergebnisse anzeigen
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    print(f"DataFrame {df_index}, Horizon {horizon}:")
    print(f"Total Interval Score: {total_score}")
    print(f"Sharpness Score: {sharpness_score}")
    print(f"Calibration Score: {calibration_score}")

#%%

from collections import defaultdict

# Dictionary zum Speichern der Interval Scores für jedes Horizon über alle DataFrames
average_scores_by_horizon = defaultdict(list)

# Durchschnitt der Interval Scores für jedes Horizon über alle DataFrames berechnen
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    average_scores_by_horizon[horizon].append(total_score.values[0])

# Ergebnisse anzeigen
print("Durchschnittliche Scores für jedes Horizon über alle DataFrames:")
for horizon, scores in average_scores_by_horizon.items():
    average_score = sum(scores) / len(scores)
    print(f"Horizon {horizon}: {average_score}")

#%%
import numpy as np

# Liste zum Speichern der Sharpness Scores für jedes Horizon und DataFrame
sharpness_scores_by_horizon = []

# Iteration über alle Einträge in der Liste der Interval Scores
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    sharpness_scores_by_horizon.append((horizon, df_index, sharpness_score.values[0]))

# Umrechnung der Liste in einen DataFrame für die einfache Berechnung des Durchschnitts
sharpness_scores_df = pd.DataFrame(sharpness_scores_by_horizon, columns=['horizon', 'df_index', 'sharpness_score'])

# Berechnung des Durchschnitts der Sharpness-Scores für jedes Horizon über alle DataFrames
average_sharpness_scores_by_horizon = sharpness_scores_df.groupby('horizon')['sharpness_score'].mean()

# Ergebnisse ausgeben
print("Durchschnittliche Sharpness-Scores für jedes Horizon über alle DataFrames:")
print(average_sharpness_scores_by_horizon)











# %%

'''
=======================================================================
QRF with a lagged Variable
=======================================================================
'''
start_date1 = '2021-01-01'
end_date1 = '2024-2-7'

data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date1)]
print(data_q1)
#%%

X = (data_q1[[
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
       'weekday_5', 'weekday_6',
        'month_2', 'month_3', 'month_4', 'month_5',
       'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11',
       'month_12',
         'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
       'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
       'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
       'DaylightHours_9.0', 'DaylightHours_10.0', 'DaylightHours_11.0',
       'DaylightHours_12.0', 'DaylightHours_13.0', 'DaylightHours_14.0',
       'DaylightHours_15.0', 'DaylightHours_16.0', 'DaylightHours_17.0',
       'season_spring', 'season_summer', 
       'season_winter', 
       'daytime_night',
       'Konstante']])

y = data_q1['gesamt']

#%%
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
rf_model = RandomForestQuantileRegressor(random_state=42)

X['gesamt_lag_1'] = y.shift(1)  # yt-1
X['gesamt_lag_24'] = y.shift(24)  # yt-24
X['gesamt_lag_247'] = y.shift(247)  # yt-247

X = X.dropna()
y = y[X.index] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Trainiere das Modell
rf_model.fit(X_train, y_train)

# Mache Vorhersagen auf den Testdaten
predictions = rf_model.predict(X_test, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975])
predictions_df = pd.DataFrame(predictions, columns=[f'quantile_{q}' for q in quantiles])
predictions_df['date_time'] = X_test.index

predictions_df = predictions_df.set_index('date_time')
predictions_df


#%%

'''
Die Lag Werte Schätzen
'''

submission_deadline = datetime(2024, 2, 7, 00, 00)

#%%
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])
#%%
df_horizon = X_full
df_horizon = df_horizon.reset_index()
#%%
df_horizon['date_time'] = pd.to_datetime(df_horizon['date_time'])
df_horizon['weekday'] = df_horizon['date_time'].dt.dayofweek
#%%
df_horizon.set_index("date_time", inplace=True)
df_horizon.head()
#%%
horizons = list(range(1,94))
LAST_IDX = -1
LAST_DATE = df_horizon.iloc[LAST_IDX].name


def get_date_from_horizon(last_ts, horizon):
    return last_ts + pd.DateOffset(hours=horizon)


#%%
horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
horizon_date
#%%
tau = [0.025, 0.25, 0.5, 0.75, 0.975]
#%%
future_X = pd.DataFrame(index=range(len(horizon_date)), columns=X_full.columns)
future_X
#%%
#X_linear_model = X_full.drop(columns=[f'gesamt_lag_{lag}' for lag in range(1, lag_periods + 1)])
#X_linear_model
lag_columns = ['gesamt_lag_1', 'gesamt_lag_24', 'gesamt_lag_247']

# Entferne die Lag-Spalten aus dem DataFrame X_full
X_linear_model = X_full.drop(columns=lag_columns)

# Zeige das aktualisierte DataFrame X_linear_model an
X_linear_model.head()

#%%
linear_model = sm.OLS(y_full, X_linear_model)
results_linear_model = linear_model.fit()

#%%
for i, target_datetime in enumerate(horizon_date):
    # Extrahieren Sie die relevanten Features für den Zielzeitpunkt
    weekday = target_datetime.weekday()
    month = target_datetime.month
    hour = target_datetime.hour
    duration_sunlight = df_rise_set.loc[target_datetime, 'DaylightHours']

    is_9_sunlight = 1 if duration_sunlight in [9] else 0
    is_10_sunlight = 1 if duration_sunlight in [10] else 0
    is_11_sunlight = 1 if duration_sunlight in [11] else 0
    is_12_sunlight = 1 if duration_sunlight in [12] else 0
    is_13_sunlight = 1 if duration_sunlight in [13] else 0
    is_14_sunlight = 1 if duration_sunlight in [14] else 0
    is_15_sunlight = 1 if duration_sunlight in [15] else 0
    is_16_sunlight = 1 if duration_sunlight in [16] else 0
    is_17_sunlight = 1 if duration_sunlight in [17] else 0
    is_spring = 1 if month in [3, 4, 5] else 0
    is_summer = 1 if month in [6, 7, 8] else 0
    is_winter = 1 if month in [12, 1, 2] else 0
    is_daytime_night = 1 if hour in [22,23,1,2,3,4,5,6] else 0

    lag_variables = [X.loc[target_datetime, column] for column in lag_columns]


    # Erstellen Sie den Feature-Vektor für die Vorhersage
    input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight] + [is_spring, is_summer, is_winter,is_daytime_night ] + [1] + lag_variables
    print(input_data)
    # Fügen Sie die Features zum DataFrame future_X hinzu
    future_X.loc[i, :] = input_data

#%%
future_X
future_X.index = horizon_date
future_X.index.name = 'date_time'
future_X
#%%
predictions_linear = results_linear_model.predict(future_X)
predictions_linear
#%%
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])
#%%
combined_y = pd.concat([y_full, predictions_linear], axis=0, ignore_index=False)
combined_y
#%%
rf_model.fit(X_full, y_full)
#%%
submission_deadline = datetime(2024, 2, 7, 00, 00)

df_horizon = X_full
df_horizon = df_horizon.reset_index()

df_horizon['date_time'] = pd.to_datetime(df_horizon['date_time'])
df_horizon['weekday'] = df_horizon['date_time'].dt.dayofweek

#%%
df_horizon.set_index("date_time", inplace=True)
df_horizon.head()
#%%
#horizons = [42, 46, 50, 66, 70, 74]
horizons = [61, 65, 69, 85, 89, 93]
LAST_IDX = -1
LAST_DATE = df_horizon.iloc[LAST_IDX].name


def get_date_from_horizon(last_ts, horizon):
    return last_ts + pd.DateOffset(hours=horizon)

#%%    

horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
horizon_date

#%%
horizons = [61, 65, 69, 85, 89, 93]
#%%
tau = [0.025, 0.25, 0.5, 0.75, 0.975]
#%%
future_X = pd.DataFrame(index=range(len(horizon_date)), columns=X_full.columns)
future_X
#%%
lag_periods = 3

# future_timestamps erstellen, eine Liste von zukünftigen Zeitstempeln basierend auf den Lag-Werten
future_timestamps = [target_datetime - pd.DateOffset(hours=lag) for lag in [1, 24, 247]]

# Ein leeres DataFrame df_future erstellen, das die gleichen Spalten wie df_horizon hat, aber Zeilen für die zukünftigen Zeitpunkte enthält
df_future = pd.DataFrame(index=future_timestamps, columns=df_horizon.columns)

# df_future anzeigen
df_future

#%%
for i, target_datetime in enumerate(horizon_date):
    # Extrahieren Sie die relevanten Features für den Zielzeitpunkt
    month = target_datetime.month
    hour = target_datetime.hour
    weekday = target_datetime.weekday()
    duration_sunlight = df_rise_set.loc[target_datetime, 'DaylightHours']

    is_9_sunlight = 1 if duration_sunlight in [9] else 0
    is_10_sunlight = 1 if duration_sunlight in [10] else 0
    is_11_sunlight = 1 if duration_sunlight in [11] else 0
    is_12_sunlight = 1 if duration_sunlight in [12] else 0
    is_13_sunlight = 1 if duration_sunlight in [13] else 0
    is_14_sunlight = 1 if duration_sunlight in [14] else 0
    is_15_sunlight = 1 if duration_sunlight in [15] else 0
    is_16_sunlight = 1 if duration_sunlight in [16] else 0
    is_17_sunlight = 1 if duration_sunlight in [17] else 0
    is_spring = 1 if month in [3, 4, 5] else 0
    is_summer = 1 if month in [6, 7, 8] else 0
    is_winter = 1 if month in [12, 1, 2] else 0
    is_daytime_night = 1 if hour in [22,23,1,2,3,4,5,6] else 0


    lag_variables = [X.loc[target_datetime, column] for column in lag_columns]



    # Erstellen Sie den Feature-Vektor für die Vorhersage
    input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight] + [is_spring, is_summer, is_winter,is_daytime_night ] + [1] + lag_variables
    print(input_data)
    # Fügen Sie die Features zum DataFrame future_X hinzu
    future_X.loc[i, :] = input_data
#%%
predictions = rf_model.predict(future_X, quantiles = [0.025, 0.25, 0.5, 0.75, 0.975])
predictions


#%%

from sklearn.ensemble import GradientBoostingRegressor
'''
----------------------------------
Gradient Boosting Regressor
=> optimized with the quantile Loss
-----------------------------------
'''

common_params = {
    'learning_rate': 0.1,
    'n_estimators': 300,
    'max_depth': 5,
    'min_samples_leaf': 9,
    'min_samples_split': 13,
    'random_state': 42
}


start_date1 = '2021-01-01'
end_date1 = '2023-11-15'

data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date1)]

submission_deadline = datetime(2023, 11, 15, 0, 0)

num_weeks = 9
all_forecasts_boosting = []

# Looping durch die Wochen
for week_num in range(0, num_weeks + 1):
    end_date = pd.to_datetime(submission_deadline) + pd.DateOffset(weeks=week_num)
    new_submission_deadline = end_date + pd.DateOffset(hours=0)  # Annahme: 0 Stunden nach dem Enddatum
    print(f"\nIteration {week_num}:")
    print("Enddatum:", end_date)
    print("Abgabetermin:", new_submission_deadline)

    # Daten für die aktuelle Woche auswählen
    data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date)]
 
    # Trainieren der Modelle mit data_q1
    X = data_q1[['weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
                 'weekday_5', 'weekday_6',
                 'month_2', 'month_3', 'month_4', 'month_5',
                 'month_6', 'month_7', 'month_8',
                 'month_9', 'month_10', 'month_11',
                 'month_12',
                 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
                 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
                 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
                 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                # 'DaylightHours_9.0', 'DaylightHours_10.0', 'DaylightHours_11.0',
                #'DaylightHours_12.0', 'DaylightHours_13.0', 'DaylightHours_14.0',
                #'DaylightHours_15.0', 'DaylightHours_16.0', 'DaylightHours_17.0',
                'Konstante']]
    y = data_q1['gesamt']

    # Trainieren der Modelle für verschiedene Quantile
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
    models = {}
    for quantile in quantiles:
        gbr_quantile = GradientBoostingRegressor(loss='quantile', alpha=quantile, **common_params)
        gbr_quantile.fit(X, y)
        models[f'Quantil_{quantile}'] = gbr_quantile

    # Definieren der Horizonte für die Vorhersagen
    horizons = [61, 65, 69, 85, 89, 93]
    LAST_IDX = -1
    df_horizon = data_energy.copy()
    df_horizon['weekday'] = df_horizon.index.weekday
    df_horizon = df_horizon.loc[df_horizon.index < end_date]
    LAST_DATE = df_horizon.iloc[LAST_IDX].name

    # Funktion zur Berechnung der Zeitpunkte basierend auf den Horizonten
    def get_date_from_horizon(last_ts, horizon):
        return last_ts + pd.DateOffset(hours=horizon)

    # Generieren der Vorhersagen für die Woche
    horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
    print(horizon_date)

    future_X = pd.DataFrame(index=range(len(horizon_date)), columns=X.columns)
    for i, target_datetime in enumerate(horizon_date):
        weekday = target_datetime.weekday()
        month = target_datetime.month
        hour = target_datetime.hour
       # duration_sunlight = df_rise_set.loc[target_datetime, 'DaylightHours']

        #is_9_sunlight = 1 if duration_sunlight in [9] else 0
        #is_10_sunlight = 1 if duration_sunlight in [10] else 0
        #is_11_sunlight = 1 if duration_sunlight in [11] else 0
        #is_12_sunlight = 1 if duration_sunlight in [12] else 0
        #is_13_sunlight = 1 if duration_sunlight in [13] else 0
        #is_14_sunlight = 1 if duration_sunlight in [14] else 0
        #is_15_sunlight = 1 if duration_sunlight in [15] else 0
        #is_16_sunlight = 1 if duration_sunlight in [16] else 0
        #is_17_sunlight = 1 if duration_sunlight in [17] else 0
        input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)]  + [1]

        #input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight]  + [1]
        future_X.loc[i, :] = input_data
        print(input_data)

    # Generieren der Vorhersagen für die Woche
    quantile_predictions = {}
    for quantile, model in models.items():
        predictions = model.predict(future_X)
        quantile_predictions[quantile] = predictions

    # Umwandeln der Vorhersagen in ein DataFrame
    quantile_predictions_df = pd.DataFrame(quantile_predictions, index=future_X.index)
    quantile_predictions_df['forecast_date'] = horizon_date
    quantile_predictions_df.set_index('forecast_date', inplace = True)

    # Anzeigen der Vorhersagen
    print(quantile_predictions_df)  
    print(f"Quantile Predictions for Week {week_num}:")
    print(quantile_predictions_df)
    all_forecasts_boosting.append(quantile_predictions_df)

#%%

'''
-----------
Quantile Loss
-----------
'''
# Liste zum Speichern der Durchschnittslosses für jeden Quantil-Level
average_losses = []

# Liste zum Speichern der Zwischenergebnisse pro DataFrame
intermediate_results = []

for df_index, quantile_predictions_df in enumerate(all_forecasts_boosting):
    # Merge basierend auf dem Datetime-Index
    true_predicted_boosting = pd.merge(
        combined_df_sorted,
        quantile_predictions_df,
        left_index=True,
        right_index=True,
        how='inner'
    )
    print(true_predicted_boosting)
    # Iteriere durch die Quantile und Berechnung des linearen Quantile Scores
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'Quantil_{quantile_level}'  # Anpassung der Genauigkeit auf 3 Dezimalstellen
        true_predicted_boosting[f'quantile_score_{quantile_level}_{df_index}'] = true_predicted_boosting.apply(
            lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
            axis=1
        )

    # Berechne den Loss für jedes Quantil-Level für diesen DataFrame und speichere es in der Liste
    df_average_losses = []
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        df_average_loss = true_predicted_boosting[f'quantile_score_{quantile_level}_{df_index}'].mean()
        df_average_losses.append(df_average_loss)

    # Füge die Liste der Durchschnittslosses für diesen DataFrame zu der Gesamtliste hinzu
    average_losses.append(df_average_losses)

    # Speichere die Zwischenergebnisse pro DataFrame
    intermediate_results.append(true_predicted_boosting)

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
Quantile Loss

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

for df_index, quantile_predictions_df in enumerate(all_forecasts_boosting):
    # Merge basierend auf dem Datetime-Index
    true_predicted_boosting = pd.merge(
        combined_df_sorted,
        quantile_predictions_df,
        left_index=True,
        right_index=True,
        how='inner'
    )
    print(true_predicted_boosting)

    # Iteriere durch die Quantile und Berechnung des linearen Quantile Scores
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'Quantil_{quantile_level}'  # Anpassung der Genauigkeit auf 3 Dezimalstellen
        true_predicted_boosting[f'quantile_score_{quantile_level}_{df_index}'] = true_predicted_boosting.apply(
            lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
            axis=1
        )

    # Berechne den Loss für jedes Quantil-Level für diesen DataFrame und speichere es in der Liste
    average_losses = []
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'Quantil_{quantile_level}'  # Anpassung der Genauigkeit auf 3 Dezimalstellen
        true_predicted_boosting[f'quantile_score_{quantile_level}_{df_index}'] = true_predicted_boosting.apply(
            lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
            axis=1
        )

        # Berechne den Loss für jedes Quantil-Level und speichere es in der Liste
        average_loss = true_predicted_boosting[f'quantile_score_{quantile_level}_{df_index}'].mean()
        average_losses.append(average_loss)

    # Berechne den Durchschnittsloss über alle Quantile
    overall_average_loss = sum(average_losses) / len(average_losses)

    # Gib die Ergebnisse aus
    print(f'Durchschnittlicher Loss für alle Quantile (DataFrame {df_index}): {overall_average_loss}')

    # Gib die Durchschnittslosses für jedes Quantil-Level aus
    for quantile_level, avg_loss in zip([0.025, 0.25, 0.5, 0.75, 0.975], average_losses):
        print(f'Durchschnittlicher Loss für Quantil {quantile_level} (DataFrame {df_index}): {avg_loss}')    
# %%


'''
Coverage Probability
95%
50%
'''

import pandas as pd
import numpy as np

def calculate_coverage_probability(true_values, quantiles):
    # Überprüfen, ob die wahren Werte im Quantilintervall liegen
    return np.mean((quantiles[:, 0] <= true_values) & (true_values <= quantiles[:, 1]))

all_coverage_probabilities = []

# Iterieren Sie über jeden Datensatz
for df_index, quantile_predictions_df in enumerate(all_forecasts_boosting):
    # Merge basierend auf dem Datetime-Index
    true_predicted_boosting = pd.merge(
        combined_df_sorted,
        quantile_predictions_df,
        left_index=True,
        right_index=True,
        how='inner'
    )
    print(true_predicted_boosting)

    # Wahre Werte und Quantile für einen bestimmten Datensatz
    true_values =true_predicted_boosting['gesamt'].values  # Ersetzen Sie 'true_values_column' durch den tatsächlichen Namen Ihrer Spalte
    quantiles = true_predicted_boosting[['Quantil_0.025', 'Quantil_0.975']].values  # Ersetzen Sie 'q0.025' und 'q0.975' durch die entsprechenden Spaltennamen

    # Coverage Probability berechnen
    coverage_probabilities = [calculate_coverage_probability(true_value, quantiles) for true_value in true_values]

    # Durchschnitt der Coverage Probabilities berechnen
    average_coverage = np.mean(coverage_probabilities)

    # Speichern Sie den Durchschnitt für diesen Datensatz
    all_coverage_probabilities.append(average_coverage)

    # Ausgabe der Coverage Probabilities und des Durchschnitts
    print(f'Coverage Probabilities für Datensatz {df_index}:')
    for i in range(len(coverage_probabilities)):
        print(f'  Wahrer Wert {true_values[i]:.5f}: {coverage_probabilities[i]:.2%}')
    print(f'Durchschnittliche Coverage Probability für Datensatz {df_index}: {average_coverage:.2%}\n')

# Durchschnitt über alle Datensätze berechnen
overall_average_coverage = np.mean(all_coverage_probabilities)

# Ausgabe des Gesamtdurchschnitts
print(f'Gesamtdurchschnittliche Coverage Probability über alle Datensätze: {overall_average_coverage:.2%}')
# %%


'''
--------------
Sharpness
95%
50%
------------
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
for df_index, quantile_predictions_df in enumerate(all_forecasts_boosting):
    # Merge basierend auf dem Datetime-Index
    true_predicted_boosting = pd.merge(
        combined_df_sorted,
        quantile_predictions_df,
        left_index=True,
        right_index=True,
        how='inner'
    )
    print(true_predicted_boosting)

    # Quantile für einen bestimmten Datensatz
    quantiles = true_predicted_boosting[['Quantil_0.25', 'Quantil_0.75']].values  # Ersetzen Sie 'q0.025' und 'q0.975' durch die entsprechenden Spaltennamen

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
# %%


'''
Interval Score 
50%
----
95%
'''
def interval_score(lower, upper, y, alpha):
    if (y >= lower) and (y <= upper):
        return 2 * alpha * np.abs(y - (lower + upper) / 2)
    elif y < lower:
        return 2 * (1 - alpha) * (lower - y)
    elif y > upper:
        return 2 * (1 - alpha) * (y - upper)

# List zum Speichern der Durchschnitts Interval Scores für jeden Datensatz
all_interval_scores = []
alpha = 0.05

# Iteriere über jeden Datensatz
for df_index, quantile_predictions_df in enumerate(all_forecasts_boosting):
    # Merge basierend auf dem Datetime-Index
    true_predicted_boosting = pd.merge(
        combined_df_sorted,
        quantile_predictions_df,
        left_index=True,
        right_index=True,
        how='inner'
    )
    print(true_predicted_boosting)

    # Iteriere über jeden Zeitstempel in diesem Datensatz
    interval_scores = []
    for timestamp in true_predicted_boosting.index:
        y = true_predicted_boosting.loc[timestamp, 'gesamt']
        l = true_predicted_boosting.loc[timestamp, 'Quantil_0.025']
        r = true_predicted_boosting.loc[timestamp, 'Quantil_0.975']

        score = interval_score(l, r, y, alpha)
        interval_scores.append(score)

    # Durchschnittlicher Interval Score über alle Zeitstempel dieses Datensatzes berechnen
    average_interval_score = np.mean(interval_scores)
    all_interval_scores.append(average_interval_score)

    # Ausgabe des Durchschnitts für diesen Datensatz
    print(f'Durchschnittlicher Interval Score für Datensatz {df_index}: {average_interval_score:.5f}\n')

# Durchschnittlicher Interval Score über alle Datensätze berechnen
overall_average_interval_score = np.mean(all_interval_scores)

# Ausgabe des gesamten durchschnittlichen Interval Scores
print(f'Gesamtdurchschnittlicher Interval Score über alle Datensätze: {overall_average_interval_score:.5f}')


# %%
'''
Quantile Score
=> Gradient Boosting Model 
'''


'''
Quantile Loss
'''

import pandas as pd

def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        return 0

# Liste zum Speichern des Durchschnittsverlusts pro Quantil und pro Horizon
average_losses_by_quantile_and_horizon = []

# Iteration durch jeden DataFrame
for df_index, quantile_predictions_df in enumerate(all_forecasts_boosting):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions= pd.merge(
        combined_df_sorted,
        quantile_predictions_df,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Iteration durch jeden Quantil-Level
    average_losses_by_horizon = []
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'Quantil_{quantile_level}'
        quantile_losses_by_horizon = []
        for horizon in combined_and_predictions['horizon'].unique():
            horizon_quantile_losses = combined_and_predictions.loc[combined_and_predictions['horizon'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()
            quantile_losses_by_horizon.append(horizon_quantile_losses)
        average_losses_by_horizon.append(quantile_losses_by_horizon)
    average_losses_by_quantile_and_horizon.append(average_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon für jeden DataFrame
for df_index, average_losses_by_horizon in enumerate(average_losses_by_quantile_and_horizon):
    for quantile_index, quantile_losses_by_horizon in enumerate(average_losses_by_horizon):
        for horizon_index, horizon_loss in enumerate(quantile_losses_by_horizon):
            print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1} (DataFrame {df_index}): {horizon_loss}')


#%%


import numpy as np

# Liste zum Speichern der Durchschnittsverluste pro Quantil und pro Horizon über alle DataFrames
overall_average_losses_by_quantile_and_horizon = []

# Iteration durch jeden Quantil-Level
for quantile_index in range(len(average_losses_by_quantile_and_horizon[0])):
    overall_losses_by_horizon = []
    # Iteration durch jeden Horizon
    for horizon_index in range(len(average_losses_by_quantile_and_horizon[0][0])):
        horizon_losses = []
        # Iteration durch jeden DataFrame
        for df_losses in average_losses_by_quantile_and_horizon:
            horizon_losses.append(df_losses[quantile_index][horizon_index])
        # Berechnung des Durchschnitts für diesen Horizon über alle DataFrames
        overall_horizon_loss = np.mean(horizon_losses)
        overall_losses_by_horizon.append(overall_horizon_loss)
    overall_average_losses_by_quantile_and_horizon.append(overall_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon über alle DataFrames
for quantile_index, losses_by_horizon in enumerate(overall_average_losses_by_quantile_and_horizon):
    for horizon_index, loss in enumerate(losses_by_horizon):
        print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1}: {loss}')





#%%

average_losses_by_dataframe = []

# Iteration durch jeden DataFrame
for df_index, quantile_predictions_df in enumerate(all_forecasts_boosting):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions= pd.merge(
        combined_df_sorted,
        quantile_predictions_df,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Gesamtverlust pro DataFrame initialisieren
    total_loss = 0

    # Iteration durch jeden Quantil-Level
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'Quantil_{quantile_level}'
        
        # Iteration durch jeden Horizont
        for horizon in combined_and_predictions['horizon'].unique():
            # Verlust pro Quantil und Horizont berechnen und zum Gesamtverlust addieren
            total_loss += combined_and_predictions.loc[combined_and_predictions['horizon'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()

    # Durchschnittsverlust pro DataFrame berechnen und zur Liste hinzufügen
    average_loss = total_loss / (len([0.025, 0.25, 0.5, 0.75, 0.975]) * len(combined_and_predictions['horizon'].unique()))
    average_losses_by_dataframe.append(average_loss)

# Ausgabe des Durchschnittsverlusts pro DataFrame
for df_index, average_loss in enumerate(average_losses_by_dataframe):
    print(f'Durchschnittlicher Loss für DataFrame {df_index}: {average_loss}')


#%%

'''
    Interval Score
'''
alpha_test = 0.05

# Liste zum Speichern der Interval Scores für jeden Horizon und DataFrame
interval_scores_by_horizon = []

for df_index, quantile_predictions_df in enumerate(all_forecasts_boosting):
    # Merge basierend auf dem Datetime-Index
    true_predicted_2= pd.merge(
        combined_df_sorted,
        quantile_predictions_df,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(true_predicted_2)

    # Iteriere durch die Horizonte
    for horizon in true_predicted2['horizon'].unique():
        # Filtere die Daten für den aktuellen Horizont
        data_for_horizon = true_predicted2[true_predicted2['horizon'] == horizon]

        # Wahren Werte (observations)
        true_values = data_for_horizon['gesamt']

        # Geschätzte Quantile (q_dict)
        quantile_dict = {
            0.025: data_for_horizon['Quantil_0.025'],
            0.5: data_for_horizon['Quantil_0.5'],
            0.975: data_for_horizon['Quantil_0.975']
        }

        # Interval Score berechnen
        total_score, sharpness_score, calibration_score = interval_score(true_values, alpha_test, q_dict=quantile_dict)

        # Ergebnisse speichern
        interval_scores_by_horizon.append((horizon, df_index, total_score, sharpness_score, calibration_score))

# Ergebnisse anzeigen
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    print(f"DataFrame {df_index}, Horizon {horizon}:")
    print(f"Total Interval Score: {total_score}")
    print(f"Sharpness Score: {sharpness_score}")
    print(f"Calibration Score: {calibration_score}") 



#%%

from collections import defaultdict

# Dictionary zum Speichern der Interval Scores für jedes Horizon über alle DataFrames
average_scores_by_horizon = defaultdict(list)

# Durchschnitt der Interval Scores für jedes Horizon über alle DataFrames berechnen
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    average_scores_by_horizon[horizon].append(total_score.values[0])

# Ergebnisse anzeigen
print("Durchschnittliche Scores für jedes Horizon über alle DataFrames:")
for horizon, scores in average_scores_by_horizon.items():
    average_score = sum(scores) / len(scores)
    print(f"Horizon {horizon}: {average_score}")

#%%
import numpy as np

# Liste zum Speichern der Sharpness Scores für jedes Horizon und DataFrame
sharpness_scores_by_horizon = []

# Iteration über alle Einträge in der Liste der Interval Scores
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    sharpness_scores_by_horizon.append((horizon, df_index, sharpness_score.values[0]))

# Umrechnung der Liste in einen DataFrame für die einfache Berechnung des Durchschnitts
sharpness_scores_df = pd.DataFrame(sharpness_scores_by_horizon, columns=['horizon', 'df_index', 'sharpness_score'])

# Berechnung des Durchschnitts der Sharpness-Scores für jedes Horizon über alle DataFrames
average_sharpness_scores_by_horizon = sharpness_scores_df.groupby('horizon')['sharpness_score'].mean()

# Ergebnisse ausgeben
print("Durchschnittliche Sharpness-Scores für jedes Horizon über alle DataFrames:")
print(average_sharpness_scores_by_horizon)



#%%
'''
--------
Model 1
--------
- Konstante
- Weekdays
- Months
- Hours
'''

start_date1 = '2021-01-01'
end_date1 = '2023-11-15'
submission_deadline = datetime(2023, 11, 15, 0, 0)

# Dummy data selection for the initial period
data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date1)]

# Define the number of weeks for forecasting
num_weeks = 9
all_estimates = []

# Model Training (outside the loop)
X = data_q1[['weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
             'weekday_5', 'weekday_6',
             'month_2', 'month_3', 'month_4', 'month_5',
             'month_6', 'month_7', 'month_8',
             'month_9', 'month_10', 'month_11',
             'month_12',
             'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
             'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
             'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
             'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
             'Konstante']]
y = data_q1['gesamt']

linear_model_horizon = sm.OLS(y, X)
results_linear_model_horizon = linear_model_horizon.fit()

# Iterate over each week
for week_num in range(0,num_weeks + 1):
    # Calculate new end date and submission deadline for each iteration
    end_date = pd.to_datetime(submission_deadline) + pd.DateOffset(weeks=week_num)
    new_submission_deadline = end_date + pd.DateOffset(hours=0)  # Assuming 0 hour after end date
    print(f"\nIteration {week_num}:")
    print("End Date:", end_date)
    print("Submission Deadline:", new_submission_deadline)

    # Update the data selection based on the new end date
    data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date)]

    # Update model with new data
    X = data_q1[['weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
                 'weekday_5', 'weekday_6',
                 'month_2', 'month_3', 'month_4', 'month_5',
                 'month_6', 'month_7', 'month_8',
                 'month_9', 'month_10', 'month_11',
                 'month_12',
                 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
                 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
                 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
                 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                 'Konstante']]
    y = data_q1['gesamt']

    # Quantile Prediction
    horizons = [61, 65, 69, 85, 89, 93]
    LAST_IDX = -1
    df_horizon = data_energy.copy()
    df_horizon['weekday'] = df_horizon.index.weekday
    df_horizon = df_horizon.loc[df_horizon.index < end_date]
    LAST_DATE = df_horizon.iloc[LAST_IDX].name

    def get_date_from_horizon(last_ts, horizon):
        return last_ts + pd.DateOffset(hours=horizon)   

    horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]

    future_X = pd.DataFrame(index=range(len(horizon_date)), columns=X.columns)

    for i, target_datetime in enumerate(horizon_date):
        # Extrahieren Sie die relevanten Features für den Zielzeitpunkt
        weekday = target_datetime.weekday()
        month = target_datetime.month
        hour = target_datetime.hour

        input_data = [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+1)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [1]

        # Fügen Sie die Features zum DataFrame future_X hinzu
        future_X.loc[i, :] = input_data

    future_X.index = horizon_date
    future_X.index.name = 'date_time'
    future_X = future_X.astype(float)

    f_linear = results_linear_model_horizon.predict(future_X)
    f_linear = f_linear.astype(float)

    model = sm.OLS(y, X)
    r = model.fit()
    residuals = r.resid

    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
    quantile_values = residuals.quantile(quantiles)

    quantiles_predictions = [0.025, 0.25, 0.5, 0.75, 0.975]

    quantile_predictions = pd.DataFrame()

    for quantile in quantiles_predictions:
        # Schätze das Quantil der Punktvorhersagen für jedes Zeitfenster
        quantile_values = f_linear + np.percentile(residuals, quantile * 100)

        # Füge die geschätzten Quantile dem DataFrame hinzu
        quantile_predictions[f'Quantile_{quantile}'] = quantile_values
        
    quantile_predictions['horizon'] = [36, 40, 44, 60, 64, 68]  

        

    # Der DataFrame "quantile_predictions" enthält jetzt die geschätzten Quantile der Punktvorhersagen für jedes Zeitfenster
    print(f"Quantile Predictions for Week {week_num}:")
    print(quantile_predictions)
    all_estimates.append(quantile_predictions)


#%%
'''
Quantile Loss
'''

import pandas as pd

def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        return 0

# Liste zum Speichern des Durchschnittsverlusts pro Quantil und pro Horizon
average_losses_by_quantile_and_horizon = []

# Iteration durch jeden DataFrame
for df_index, quantile_predictions in enumerate(all_estimates):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions= pd.merge(
        combined_df_sorted,
        quantile_predictions,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Iteration durch jeden Quantil-Level
    average_losses_by_horizon = []
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'Quantile_{quantile_level}'
        quantile_losses_by_horizon = []
        for horizon in combined_and_predictions['horizon_x'].unique():
            horizon_quantile_losses = combined_and_predictions.loc[combined_and_predictions['horizon_x'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()
            quantile_losses_by_horizon.append(horizon_quantile_losses)
        average_losses_by_horizon.append(quantile_losses_by_horizon)
    average_losses_by_quantile_and_horizon.append(average_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon für jeden DataFrame
for df_index, average_losses_by_horizon in enumerate(average_losses_by_quantile_and_horizon):
    for quantile_index, quantile_losses_by_horizon in enumerate(average_losses_by_horizon):
        for horizon_index, horizon_loss in enumerate(quantile_losses_by_horizon):
            print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1} (DataFrame {df_index}): {horizon_loss}')


#%%
# Liste zum Speichern der Durchschnittsverluste pro Quantil und pro Horizon über alle DataFrames
overall_average_losses_by_quantile_and_horizon = []

# Iteration durch jeden Quantil-Level
for quantile_index in range(len(average_losses_by_quantile_and_horizon[0])):
    overall_losses_by_horizon = []
    # Iteration durch jeden Horizon
    for horizon_index in range(len(average_losses_by_quantile_and_horizon[0][0])):
        horizon_losses = []
        # Iteration durch jeden DataFrame
        for df_losses in average_losses_by_quantile_and_horizon:
            horizon_losses.append(df_losses[quantile_index][horizon_index])
        # Berechnung des Durchschnitts für diesen Horizon über alle DataFrames
        overall_horizon_loss = np.mean(horizon_losses)
        overall_losses_by_horizon.append(overall_horizon_loss)
    overall_average_losses_by_quantile_and_horizon.append(overall_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon über alle DataFrames
for quantile_index, losses_by_horizon in enumerate(overall_average_losses_by_quantile_and_horizon):
    for horizon_index, loss in enumerate(losses_by_horizon):
        print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1}: {loss}')
#%%
average_losses_by_dataframe = []

# Iteration durch jeden DataFrame
for df_index, quantile_predictions in enumerate(all_estimates):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions= pd.merge(
        combined_df_sorted,
        quantile_predictions,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Gesamtverlust pro DataFrame initialisieren
    total_loss = 0

    # Iteration durch jeden Quantil-Level
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'Quantile_{quantile_level}'
        
        # Iteration durch jeden Horizont
        for horizon in combined_and_predictions['horizon_x'].unique():
            # Verlust pro Quantil und Horizont berechnen und zum Gesamtverlust addieren
            total_loss += combined_and_predictions.loc[combined_and_predictions['horizon_x'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()

    # Durchschnittsverlust pro DataFrame berechnen und zur Liste hinzufügen
    average_loss = total_loss / (len([0.025, 0.25, 0.5, 0.75, 0.975]) * len(combined_and_predictions['horizon_x'].unique()))
    average_losses_by_dataframe.append(average_loss)

# Ausgabe des Durchschnittsverlusts pro DataFrame
for df_index, average_loss in enumerate(average_losses_by_dataframe):
    print(f'Durchschnittlicher Loss für DataFrame {df_index}: {average_loss}')

#%%
'''
    Interval Score
    Quantil Model 2
'''
alpha_test = 0.5

# Liste zum Speichern der Interval Scores für jeden Horizon und DataFrame
interval_scores_by_horizon = []

for df_index, quantile_predictions in enumerate(all_estimates):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions= pd.merge(
        combined_df_sorted,
        quantile_predictions,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)


    # Iteriere durch die Horizonte
    for horizon in combined_and_predictions['horizon_x'].unique():
        # Filtere die Daten für den aktuellen Horizont
        data_for_horizon = combined_and_predictions[combined_and_predictions['horizon_x'] == horizon]

        # Wahren Werte (observations)
        true_values = data_for_horizon['gesamt']

        # Geschätzte Quantile (q_dict)
        quantile_dict = {
            0.25: data_for_horizon['Quantile_0.25'],
            0.5: data_for_horizon['Quantile_0.5'],
            0.75: data_for_horizon['Quantile_0.75']
        }

        # Interval Score berechnen
        total_score, sharpness_score, calibration_score = interval_score(true_values, alpha_test, q_dict=quantile_dict)

        # Ergebnisse speichern
        interval_scores_by_horizon.append((horizon, df_index, total_score, sharpness_score, calibration_score))

# Ergebnisse anzeigen
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    print(f"DataFrame {df_index}, Horizon {horizon}:")
    print(f"Total Interval Score: {total_score}")
    print(f"Sharpness Score: {sharpness_score}")
    print(f"Calibration Score: {calibration_score}") 




#%%
from collections import defaultdict

# Dictionary zum Speichern der Interval Scores für jedes Horizon über alle DataFrames
average_scores_by_horizon = defaultdict(list)

# Durchschnitt der Interval Scores für jedes Horizon über alle DataFrames berechnen
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    average_scores_by_horizon[horizon].append(total_score.values[0])

# Ergebnisse anzeigen
print("Durchschnittliche Scores für jedes Horizon über alle DataFrames:")
for horizon, scores in average_scores_by_horizon.items():
    average_score = sum(scores) / len(scores)
    print(f"horizon {horizon}: {average_score}")

#%%
import numpy as np

# Liste zum Speichern der Sharpness Scores für jedes Horizon und DataFrame
sharpness_scores_by_horizon = []

# Iteration über alle Einträge in der Liste der Interval Scores
for horizon, df_index, total_score, sharpness_score, calibration_score in interval_scores_by_horizon:
    sharpness_scores_by_horizon.append((horizon, df_index, sharpness_score.values[0]))

# Umrechnung der Liste in einen DataFrame für die einfache Berechnung des Durchschnitts
sharpness_scores_df = pd.DataFrame(sharpness_scores_by_horizon, columns=['horizon', 'df_index', 'sharpness_score'])

# Berechnung des Durchschnitts der Sharpness-Scores für jedes Horizon über alle DataFrames
average_sharpness_scores_by_horizon = sharpness_scores_df.groupby('horizon')['sharpness_score'].mean()

# Ergebnisse ausgeben
print("Durchschnittliche Sharpness-Scores für jedes Horizon über alle DataFrames:")
print(average_sharpness_scores_by_horizon)



























# %%


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



# %%

import pandas as pd

def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        return 0

# Liste zum Speichern des Durchschnittsverlusts pro Quantil und pro Horizon
average_losses_by_quantile_and_horizon = []


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

    average_losses_by_horizon = []
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'q{quantile_level}'
        quantile_losses_by_horizon = []
        for horizon in combined_and_predictions['horizon_x'].unique():
            horizon_quantile_losses = combined_and_predictions.loc[combined_and_predictions['horizon_x'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()
            quantile_losses_by_horizon.append(horizon_quantile_losses)
        average_losses_by_horizon.append(quantile_losses_by_horizon)
    average_losses_by_quantile_and_horizon.append(average_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon für jeden DataFrame
for df_index, average_losses_by_horizon in enumerate(average_losses_by_quantile_and_horizon):
    for quantile_index, quantile_losses_by_horizon in enumerate(average_losses_by_horizon):
        for horizon_index, horizon_loss in enumerate(quantile_losses_by_horizon):
            print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1} (DataFrame {df_index}): {horizon_loss}')
# %%
overall_average_losses_by_quantile_and_horizon = []

# Iteration durch jeden Quantil-Level
for quantile_index in range(len(average_losses_by_quantile_and_horizon[0])):
    overall_losses_by_horizon = []
    # Iteration durch jeden Horizon
    for horizon_index in range(len(average_losses_by_quantile_and_horizon[0][0])):
        horizon_losses = []
        # Iteration durch jeden DataFrame
        for df_losses in average_losses_by_quantile_and_horizon:
            horizon_losses.append(df_losses[quantile_index][horizon_index])
        # Berechnung des Durchschnitts für diesen Horizon über alle DataFrames
        overall_horizon_loss = np.mean(horizon_losses)
        overall_losses_by_horizon.append(overall_horizon_loss)
    overall_average_losses_by_quantile_and_horizon.append(overall_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon über alle DataFrames
for quantile_index, losses_by_horizon in enumerate(overall_average_losses_by_quantile_and_horizon):
    for horizon_index, loss in enumerate(losses_by_horizon):
        print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1}: {loss}')
# %%
def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        return 0

# Liste zum Speichern des Durchschnittsverlusts pro Quantil und pro Horizon
average_losses_by_quantile_and_horizon = []


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

    average_losses_by_horizon = []
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'q{quantile_level}'
        quantile_losses_by_horizon = []
        for horizon in combined_and_predictions['horizon_x'].unique():
            horizon_quantile_losses = combined_and_predictions.loc[combined_and_predictions['horizon_x'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()
            quantile_losses_by_horizon.append(horizon_quantile_losses)
        average_losses_by_horizon.append(quantile_losses_by_horizon)
    average_losses_by_quantile_and_horizon.append(average_losses_by_horizon)

# Ausgabe des Durchschnittsverlusts pro Quantil und pro Horizon für jeden DataFrame
for df_index, average_losses_by_horizon in enumerate(average_losses_by_quantile_and_horizon):
    for quantile_index, quantile_losses_by_horizon in enumerate(average_losses_by_horizon):
        for horizon_index, horizon_loss in enumerate(quantile_losses_by_horizon):
            print(f'Durchschnittlicher Loss für Quantil {quantile_index} und Horizon {horizon_index + 1} (DataFrame {df_index}): {horizon_loss}')



#%%
average_losses_by_dataframe = []

# Iteration durch jeden DataFrame
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

    # Gesamtverlust pro DataFrame initialisieren
    total_loss = 0

    # Iteration durch jeden Quantil-Level
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'q{quantile_level}'
        
        # Iteration durch jeden Horizont
        for horizon in combined_and_predictions['horizon_x'].unique():
            # Verlust pro Quantil und Horizont berechnen und zum Gesamtverlust addieren
            total_loss += combined_and_predictions.loc[combined_and_predictions['horizon_x'] == horizon].apply(
                lambda row: quantile_score(row[quantile_column], row['gesamt'], quantile_level),
                axis=1
            ).mean()

    # Durchschnittsverlust pro DataFrame berechnen und zur Liste hinzufügen
    average_loss = total_loss / (len([0.025, 0.25, 0.5, 0.75, 0.975]) * len(combined_and_predictions['horizon_x'].unique()))
    average_losses_by_dataframe.append(average_loss)

# Ausgabe des Durchschnittsverlusts pro DataFrame
for df_index, average_loss in enumerate(average_losses_by_dataframe):
    print(f'Durchschnittlicher Loss für DataFrame {df_index}: {average_loss}')                    
# %%
