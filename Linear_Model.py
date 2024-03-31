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
'''
--------------------------------------------
Astral Sun => Sunset und Sunrise calculation.
---------------------------------------------
'''
from astral import LocationInfo
from astral.sun import sun

def get_sunrise_sunset(latitude, longitude, date):
    city = LocationInfo(latitude, longitude)
    s = sun(city.observer, date=date, tzinfo=city.timezone)
    return s['sunrise'], s['sunset']

# Beispiel für Edermünde => Geographischer Schwerpunkt von Deutschland
latitude = 51.2163421
longitude = 9.3989666

# Startdatum
start_date = datetime(2018, 12, 24)

# Enddatum
end_date = datetime(2024, 12, 31)

# Liste zum Speichern der Ergebnisse
astral_data = []

# Schleife durch alle Tage im Jahr
current_date = start_date
while current_date <= end_date:
    sunrise, sunset = get_sunrise_sunset(latitude, longitude, current_date)
    sunrise_sunset_data.append({
        'Date': current_date,
        'Sunrise': sunrise,
        'Sunset': sunset
    })
    current_date += timedelta(days=1)

# DataFrame erstellen
astral_data = pd.DataFrame(astral_data)

# Ergebnisse anzeigen
print(astral_data)









# %%

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
# Füge eine Spalte 'IstFeiertag' zum DataFrame hinzu
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
start_date1 = '2023-01-01'
end_date1 = '2024-1-20'

data_q1 = dummy_data[(dummy_data.index >= start_date1) & (dummy_data.index < end_date1)]
print(data_q1)
#%%
data_q1.columns
# %%
X = (data_q1[[
   #'weekday' ,
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
       'weekday_5', 'weekday_6',
        'month_2', 'month_3', 'month_4', 'month_5',
       'month_6', 'month_7', 'month_8', 
       'month_9', 'month_10', 'month_11',
       'month_12',
         'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
       'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
       'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
       #'DaylightHours_9.0', 'DaylightHours_10.0', 'DaylightHours_11.0',
       #'DaylightHours_12.0',
        # 'DaylightHours_13.0', 'DaylightHours_14.0',
       #'DaylightHours_15.0', 'DaylightHours_16.0', 'DaylightHours_17.0',
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
# %%
linear_model = sm.OLS(y_train, X_train)
# %%
results_linear_model = linear_model.fit()
# %%
print(results_linear_model.summary())
#%%
predictions = results_linear_model.predict(X_test)
#%%
predictions
#%%
residues = [y_test - predictions] 
#%%
quantiles_residues = np.percentile(residues, [2.5, 25, 50, 75, 97.5])
#%%
quantile_residues = pd.DataFrame({'Quantile': [2.5, 25, 50, 75, 97.5], 'Residuen': quantiles_residues})
quantile_residues

#%%
quantiles_predictions = [2.5, 25, 50, 75, 97.5]
quantile_values = [predictions + np.percentile(residues, q) for q in quantiles_predictions]

# DataFrame für die Quantile der Punktvorhersagen erstellen
quantile_predictions = pd.DataFrame(dict(zip([f'Quantile_{q}' for q in quantiles_predictions], quantile_values)))

# Der DataFrame "quantile_predictions" enthält jetzt die geschätzten Quantile der Punktvorhersagen
print(quantile_predictions)
#%%

'''
------------------------------
Predictions for the 6 horizons
- need to make predictions for the future
- we use the entiere available data set from X and Y to make the forecasts
---------------------------
'''
np.random.seed(42)
linear_model_horizon = sm.OLS(y, X)
results_linear_model_horizon = linear_model_horizon.fit()
#%%
submission_deadline = datetime(2024, 1, 10, 00, 00)

#%%
df_horizon = data_energy.copy()
df_horizon
#%%
# Fügen Sie eine neuSpalte 'Wochentag' hinzu
df_horizon['weekday'] = df_horizon.index.weekday

df_horizon.head()
#%%
end_date= pd.to_datetime('2024-1-10 00:00:00')
df_horizon = df_horizon.loc[df_horizon.index < end_date]
df_horizon
#%%
horizons = [61, 65, 69, 85, 89, 93]
LAST_IDX = -1
LAST_DATE = df_horizon.iloc[LAST_IDX].name


def get_date_from_horizon(last_ts, horizon):
    return last_ts + pd.DateOffset(hours=horizon)

#%%
horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
horizon_date

#%%
hours_offset = [61, 65, 69, 85, 89, 93]
#%%
tau = [0.025, 0.25, 0.5, 0.75, 0.975]
#%%
'''
Punktvorhersagen berechnen für die sechs Zeitfenster
'''
predictions_model_horizon = []
point_predictions_model_horizon = []

for offset in hours_offset:
    target_datetime = LAST_DATE + pd.DateOffset(hours=offset)
    print(target_datetime)
    # Extrahieren Sie den Monat, die Uhrzeit und den Wochentag für den Zielzeitpunkt
    month = target_datetime.month
    hour = target_datetime.hour
    weekday = target_datetime.weekday()
    #print(month)
    print(hour)
    print(weekday)

    # Erstellen Sie die Eingabedaten entsprechend Ihrem Modell für den aktuellen Zielzeitpunkt
    #input_data = [weekday == (i+1) for i in range(6)] + [hour == (i) for i in range(1,24)] + [1]
    input_data = [weekday == (i+1) for i in range(6)] + [month == (i) for i in range(11)] + [hour == (i) for i in range(1,24)] + [1]

    print(input_data)

    # Führen Sie die Vorhersage mit dem Modell für den aktuellen Zielzeitpunkt durch
    predicted_value = results_linear_model_horizon.predict(input_data)
    print(predicted_value)

    # Fügen Sie die Vorhersage zu quantile_predictions hinzu
    predictions_model_horizon.append((target_datetime, predicted_value))
    point_predictions_model_horizon.append((predicted_value))


#%%
future_X = pd.DataFrame(index=range(len(horizon_date)), columns=X.columns)
future_X
#%%
for i, target_datetime in enumerate(horizon_date):
    # Extrahieren Sie die relevanten Features für den Zielzeitpunkt

    weekday = target_datetime.weekday()
    month = target_datetime.month
    hour = target_datetime.hour
    #duration_sunlight = df_rise_set.loc[target_datetime, 'DaylightHours']

    #is_9_sunlight = 1 if duration_sunlight in [9] else 0
    #is_10_sunlight = 1 if duration_sunlight in [10] else 0
    #is_11_sunlight = 1 if duration_sunlight in [11] else 0
    #is_12_sunlight = 1 if duration_sunlight in [12] else 0
    #is_13_sunlight = 1 if duration_sunlight in [13] else 0
    #is_14_sunlight = 1 if duration_sunlight in [14] else 0
    #is_15_sunlight = 1 if duration_sunlight in [15] else 0
    #is_16_sunlight = 1 if duration_sunlight in [16] else 0
    #is_17_sunlight = 1 if duration_sunlight in [17] else 0
    #is_spring = 1 if month in [3, 4, 5] else 0
    #is_summer = 1 if month in [6, 7, 8] else 0
    #is_winter = 1 if month in [12, 1, 2] else 0
    #is_weekday = 1 if weekday in [0,1,2,3,4] else 0
    #is_daytime_night = 1 if hour in [22,23,1,2,3,4,5,6] else 0



    # Erstellen Sie den Feature-Vektor für die Vorhersage
    #input_data =[is_weekday] + [is_daytime_night] + [is_spring, is_summer, is_winter] + [1]
    #input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight, is_15_sunlight, is_16_sunlight, is_17_sunlight] + [is_spring, is_summer, is_winter,is_daytime_night ] + [1]
    input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+1)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] + [1]

    print(input_data)
    # Fügen Sie die Features zum DataFrame future_X hinzu
    future_X.loc[i, :] = input_data

#%%
future_X
future_X.index = horizon_date
future_X.index.name = 'date_time'
future_X
#%%
f_linear = results_linear_model_horizon.predict(future_X)
f_linear
#%%
f_linear = f_linear.astype(float)
f_linear

#%%
future_X = future_X.astype(float)
#%%
model = sm.OLS(y, X)
r = model.fit()
residuals = r.resid

#%%
'''
Quantile der Residuen berechnen
'''
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
quantile_values = residuals.quantile(quantiles)
quantile_values
#%%
'''
Quantile der Punktvorhersagen schätzen
'''
quantiles_predictions = [0.025, 0.25, 0.5, 0.75, 0.975]

quantile_predictions = pd.DataFrame()

for quantile in quantiles_predictions:
    # Schätze das Quantil der Punktvorhersagen für jedes Zeitfenster
    quantile_values = f_linear + np.percentile(residuals, quantile * 100)

    # Füge die geschätzten Quantile dem DataFrame hinzu
    quantile_predictions[f'Quantile_{quantile}'] = quantile_values

# Der DataFrame "quantile_predictions" enthält jetzt die geschätzten Quantile der Punktvorhersagen für jedes Zeitfenster
print(quantile_predictions)
#%%
quantile_predictions = pd.DataFrame(quantile_predictions)
quantile_predictions

#%%
horizons_abgabe = [36, 40, 44, 60, 64, 68]
forecastdate = datetime(2024, 1, 10, 00, 00)

df_linear = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "energy",
    "horizon": [str(h) + " hour" for h in horizons_abgabe],
    "q0.025": quantile_predictions['Quantile_0.025'],
    "q0.25": quantile_predictions['Quantile_0.25'],
    "q0.5": quantile_predictions['Quantile_0.5'],
    "q0.75": quantile_predictions['Quantile_0.75'],
    "q0.975": quantile_predictions['Quantile_0.975']
})
df_linear
#%%
print(df_linear)

#%%
'''
Funktion die für die Forecast Evaluation
=> Gibt die wahren y basierend auf den horizon h und dem forecast_date heraus
'''
forecast_date = pd.to_datetime('2024-1-10 00:00:00')  # Hier dein tatsächliches Prognosedatum einfügen
horizon = [36, 40, 44, 60, 64, 68]

# Extrahiere den Wert für jedes Zeitfenster nach dem Prognosedatum
stromverbrauch_true = {}

for h in horizon:
    target_datetime = forecast_date + pd.Timedelta(hours=h)
    stromverbrauch_true[h] = data_energy.loc[target_datetime, 'gesamt']
    stromverbrauch_true_df = pd.DataFrame(list(stromverbrauch_true.items()), columns=['horizon', 'gesamt'])
    stromverbrauch_true_df.set_index('horizon', inplace=True)
print(stromverbrauch_true)

# %%

#%%
'''
------------------
Pinball Loss
-----------------
'''
tau_values = [0.025, 0.25, 0.5, 0.75, 0.975]
#%%
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
    predictions = df_linear[f'q{quantile}']
    loss = np.mean([quantile_score(q_hat, y, quantile) for q_hat, y in zip(predictions, y_true)])
    pinball_losses[quantile] = loss
    print(f"Pinball Loss for Quantile {quantile}: {loss}")

# Berechne den durchschnittlichen Pinball Loss über alle Quantile
average_pinball_loss = np.mean(list(pinball_losses.values()))
print(f"Average Pinball Loss: {average_pinball_loss}")

#%%

'''
---------------------
Diebold Mariano Test
to compare the Baseline Model with the Linear Model
--------------------
'''
S_q1_linear = pinball_losses[0.025]
S_q2_linear = pinball_losses[0.25]
S_q3_linear = pinball_losses[0.5]
S_q4_linear = pinball_losses[0.75]
S_q5_linear = pinball_losses[0.975]

S_mean_linear = np.mean([S_q1_linear, S_q2_linear, S_q3_linear, S_q4_linear, S_q5_linear], axis = 0)
S_mean_linear
# %%


'''
--------------------------
Mincer-Zarnowitz Regression
=> Checking for Autocalibration 
---------------------------
'''

'''h1 und q0.025'''


#%%
quantile_model_h1_q1 = sm.QuantReg(Y_tau_h1_q1, X_tau_h1_q1)
q_h1_q1= quantile_model_h1_q1.fit(q=0.025)
print(q_h1_q1.summary())



# %%
predictions_dict = {}
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

for quantile in quantiles:
    m = sm.QuantReg(merged_data['y_true'],sm.add_constant(merged_data['forecast']) , random_seed = 42 ).fit(q=quantile)
    print(f"Quantile {quantile}:")
    print(m.summary())
    
# %%

'''
--------------------------------------------------------------------------
Für die Evaluation:
-------------------------
Jetzt die 9 Vorhersagen automatisch genervieren
=> Lineares Model 1 
=> mit den Erklärenden Variablen wie in den Folien nur mit Weekday zusätzlich
Weekdays
Months
Hours
Konstante
----------------------------------------------------------------------------
15/11/23 => Schätzungen 17/11 und 18/11
22/11/23 => Schätzungen 24/11 und 25/11
29/11/23 => Schätzungen 01/12 und 02/12
06/12/23 => Schätzungen 08/12 und 09/12
13/12/23 => Schätzungen 15/12 und 16/12
20/12/23 => Schätzungen 22/12 und 23/12
27/12/23 => Schätzungen 29/12 und 30/12
03/01/24 => Schätzungen 05/01 und 06/01
10/01/24 => Schätzungen 12/01 und 13/01
17/01/24 => Schätzungen 19/01 und 20/01
----------------------------------------------------------------------------
'''

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
all_predictions = all_estimates.copy()
all_predictions

#%%
'''
---------------
Quantile Score 
für das Lineare Model 1
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
for df_index, quantile_predictions in enumerate(all_predictions):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        quantile_predictions,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)



    # Liste zum Speichern der Durchschnittslosses für jeden Quantil-Level
    average_losses = []

    # Iteriere durch die Quantile und Berechnung des linearen Quantile Scores
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'Quantile_{quantile_level}'  # Anpassung der Genauigkeit auf 3 Dezimalstellen
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

# %%
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
for df_index, quantile_predictions in enumerate(all_predictions):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        quantile_predictions,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Iteriere durch die Quantile und Berechnung des linearen Quantile Scores
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'Quantile_{quantile_level}'  # Anpassung der Genauigkeit auf 3 Dezimalstellen
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


# %%
'''
Coverage Probability
95%
[0.025 und 0.975]
'''

import pandas as pd
import numpy as np

def calculate_coverage_probability(true_values, quantiles):
    # Überprüfen, ob die wahren Werte im Quantilintervall liegen
    return np.mean((quantiles[:, 0] <= true_values) & (true_values <= quantiles[:, 1]))

# Beispiel: Annahme, dass Sie eine Liste von Datensätzen (predictions_dfs) haben

all_coverage_probabilities = []

# Iterieren Sie über jeden Datensatz
for df_index, quantile_predictions in enumerate(all_predictions):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        quantile_predictions,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Wahre Werte und Quantile für einen bestimmten Datensatz
    true_values = combined_and_predictions['gesamt'].values  # Ersetzen Sie 'true_values_column' durch den tatsächlichen Namen Ihrer Spalte
    quantiles = combined_and_predictions[['Quantile_0.025', 'Quantile_0.975']].values  # Ersetzen Sie 'q0.025' und 'q0.975' durch die entsprechenden Spaltennamen

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
Coverage Probability
50%
[25% und 75%]
'''


import pandas as pd
import numpy as np

def calculate_coverage_probability(true_values, quantiles):
    # Überprüfen, ob die wahren Werte im Quantilintervall liegen
    return np.mean((quantiles[:, 0] <= true_values) & (true_values <= quantiles[:, 1]))

# Beispiel: Annahme, dass Sie eine Liste von Datensätzen (predictions_dfs) haben

all_coverage_probabilities = []

# Iterieren Sie über jeden Datensatz
for df_index, quantile_predictions in enumerate(all_predictions):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        quantile_predictions,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Wahre Werte und Quantile für einen bestimmten Datensatz
    true_values = combined_and_predictions['gesamt'].values  # Ersetzen Sie 'true_values_column' durch den tatsächlichen Namen Ihrer Spalte
    quantiles = combined_and_predictions[['Quantile_0.25', 'Quantile_0.75']].values  # Ersetzen Sie 'q0.025' und 'q0.975' durch die entsprechenden Spaltennamen

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
for df_index, quantile_predictions in enumerate(all_predictions):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        quantile_predictions,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Quantile für einen bestimmten Datensatz
    quantiles = combined_and_predictions[['Quantile_0.025', 'Quantile_0.975']].values  # Ersetzen Sie 'q0.025' und 'q0.975' durch die entsprechenden Spaltennamen

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
for df_index, quantile_predictions in enumerate(all_predictions):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        quantile_predictions,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Quantile für einen bestimmten Datensatz
    quantiles = combined_and_predictions[['Quantile_0.25', 'Quantile_0.75']].values  # Ersetzen Sie 'q0.025' und 'q0.975' durch die entsprechenden Spaltennamen

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
def interval_score(lower, upper, y, alpha):
    if (y >= lower) and (y <= upper):
        return 2 * alpha * np.abs(y - (lower + upper) / 2)
    elif y < lower:
        return 2 * (1 - alpha) * (lower - y)
    elif y > upper:
        return 2 * (1 - alpha) * (y - upper)

# List zum Speichern der Durchschnitts Interval Scores für jeden Datensatz
all_interval_scores = []
alpha = 0.5

# Iteriere über jeden Datensatz
for df_index, quantile_predictions in enumerate(all_predictions):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        quantile_predictions,
        left_index=True,
        right_index=True,
        how='inner'
    )

    # Iteriere über jeden Zeitstempel in diesem Datensatz
    interval_scores = []
    for timestamp in combined_and_predictions.index:
        y = combined_and_predictions.loc[timestamp, 'gesamt']
        l = combined_and_predictions.loc[timestamp, 'Quantile_0.25']
        r = combined_and_predictions.loc[timestamp, 'Quantile_0.75']

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



#%%
'''
Interval Score
[95%]
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
for df_index, quantile_predictions in enumerate(all_predictions):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        quantile_predictions,
        left_index=True,
        right_index=True,
        how='inner'
    )

    # Iteriere über jeden Zeitstempel in diesem Datensatz
    interval_scores = []
    for timestamp in combined_and_predictions.index:
        y = combined_and_predictions.loc[timestamp, 'gesamt']
        l = combined_and_predictions.loc[timestamp, 'Quantile_0.025']
        r = combined_and_predictions.loc[timestamp, 'Quantile_0.975']

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
---------
MODEL 2
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
             #'season_spring', 'season_summer', 'season_winter', 
              #  'daytime_night',
               # 'weekday',
             'DaylightHours_9.0', 'DaylightHours_10.0', 'DaylightHours_11.0',
                'DaylightHours_12.0',
                'DaylightHours_13.0', 'DaylightHours_14.0',
                'DaylightHours_15.0', 'DaylightHours_16.0', 'DaylightHours_17.0',
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
                 'DaylightHours_9.0', 'DaylightHours_10.0', 'DaylightHours_11.0',
                'DaylightHours_12.0',
                'DaylightHours_13.0', 'DaylightHours_14.0',
                'DaylightHours_15.0', 'DaylightHours_16.0', 'DaylightHours_17.0',
                #'season_spring', 'season_summer', 'season_winter', 
                #'daytime_night',
                #'weekday',
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

        #is_spring = 1 if month in [3, 4, 5] else 0
        #is_summer = 1 if month in [6, 7, 8] else 0
        #is_winter = 1 if month in [12, 1, 2] else 0

        #is_weekday = 1 if weekday in [0,1,2,3,4] else 0

        #is_daytime_night = 1 if hour in [22,23,1,2,3,4,5,6] else 0
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

        #input_data =  [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+2)) for i in range(11)] +  [int(hour == (i+1)) for i in range(23)] +  [is_spring, is_summer, is_winter, is_daytime_night , is_weekday ] + [1]

        input_data = [int(weekday == (i+1)) for i in range(6)] + [int(month == (i+1)) for i in range(11)] + [int(hour == (i+1)) for i in range(23)]  + [is_9_sunlight, is_10_sunlight, is_11_sunlight, is_12_sunlight, is_13_sunlight, is_14_sunlight ,is_15_sunlight, is_16_sunlight, is_17_sunlight] + [1]

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
# %%


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
for df_index, quantile_predictions in enumerate(all_predictions):
    # Merge basierend auf dem Datetime-Index
    combined_and_predictions = pd.merge(
        combined_df_sorted,
        quantile_predictions,
        left_index=True,
        right_index=True,  # Hier auf den Index des zweiten DataFrame mergen
        how='inner'
    )
    print(combined_and_predictions)

    # Iteriere durch die Quantile und Berechnung des linearen Quantile Scores
    for quantile_level in [0.025, 0.25, 0.5, 0.75, 0.975]:
        quantile_column = f'quantile_{quantile_level}'  # Anpassung der Genauigkeit auf 3 Dezimalstellen
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

# %%
