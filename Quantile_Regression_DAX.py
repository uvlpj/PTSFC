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
#data = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission7/^GDAXI.csv')
#data = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission8/^GDAXI.csv')
hist = pd.read_csv('/Users/sophiasiefert/Downloads/^GDAXI.csv')



#%%
#hist2 = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/22_11_23^GDAXI.csv')
#hist2 = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission5/^GDAXI.csv')
#hist2 = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission6/^GDAXI Kopie.csv')
#hist2 = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission7/^GDAXI Kopie.csv')
#hist2 = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/DAX/Submissions/Submission8/^GDAXI Kopie 3.csv')

#%%

#%%
#for i in range(5):
 #   data["ret"+str(i+1)] = compute_return(data["Close"].values, h=i+1)

#%%
for i in range(5):
    hist["ret"+str(i+1)] = compute_return(hist["Close"].values, h=i+1)    
#%%
hist

#%%
hist = hist.dropna()
# %%
#del data['Open']
#del data['High']
#del data['Low']
#del data['Volume']
#del data['Adj Close']
# %%

# %%
hist['Rt'] = 100 * (np.log(hist['Close']) - np.log(hist['Close'].shift(1)))
#%%
hist['price_up'] = np.where(hist['Close'] > hist['Close'].shift(1), 1, 0)

# Variable für Renditenänderung erstellen (1: gestiegen, 0: gleich geblieben oder gefallen)
hist['ret1_up'] = np.where(hist['ret1'] > 0, 1, 0)
hist['ret2_up'] = np.where(hist['ret2'] > 0, 1, 0)
hist['ret3_up'] = np.where(hist['ret3'] > 0, 1, 0)
hist['ret4_up'] = np.where(hist['ret4'] > 0, 1, 0)
hist['ret5_up'] = np.where(hist['ret5'] > 0, 1, 0)
#%%
hist = hist.dropna()

#%%
hist.set_index('Date', inplace=True)
#%%

train_end_date = '2023-11-15'
test_start_date = '2023-11-16'
test_end_date = '2024-01-17'

train_data = hist.loc[:train_end_date]
test_data = hist.loc[test_start_date:test_end_date]







#%%
'''
Quantile Regression for the DAX returns (ret1, ret2, ..., ret5)
Y = Rt+1:t+h and X = (1, |Rt|)
for h = 1: Y = Rt+1:t+1 = Rt+1 and X = (1, |Rt|)
for h = 2: Y = Rt+1:t+2 = Rt+1 + Rt+2 and X = (1, |Rt|)
for h = 3: Y = Rt+1:t+3 = Rt+1 + Rt+2 + Rt+3 and X = (1, |Rt|)
for h = 4: Y = Rt+1:t+4 = Rt+1 + Rt+2 + Rt+3 + Rt+4 and X = (1, |Rt|)
for h = 5: Y = Rt+1:t+5 = Rt+1 + Rt+2 + Rt+3 + Rt+4 + Rt+5 and X = (1, |Rt|)
'''

# %%
'''
For h = 1 
Quantile Predictions for the log returns ret1
'''
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles = pd.DataFrame(index=quantiles, columns=['predicted_ret1'])

# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = hist['Rt'].shift(-1).iloc[:-1]  # Rt+1:t+1
    X = sm.add_constant(hist['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    # Quantilregression
    quant_reg = sm.QuantReg(Y, X).fit(q=tau)

    # Vorhersagen für h=1
    recent_Rt = hist['Rt'].iloc[-1]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles.loc[tau, 'predicted_ret1'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles)

#%%
'''
Mit einem Rolling Window Ansatz:
h = 1
'''

import statsmodels.api as sm

window_size = 7  # Eine Woche
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# Liste zur Speicherung der Vorhersagen für jede Woche und jedes Quantil
pred_quantiles_list_1 = []

# Schleife über die Zeitpunkte im Testdatensatz
for i in range(len(test_data) - window_size + 1):
    print(test_data)
    print(len(test_data) - window_size + 1)
    # DataFrame zur Speicherung der Vorhersagen für die aktuelle Woche
    pred_quantiles_week_1 = pd.DataFrame(index=quantiles, columns=['predicted_ret1', 'start_date'])

    # Aktueller Zeitpunkt im Rolling-Window
    current_window = test_data.iloc[i:i + window_size]
    window_start_date = current_window.index[0]
    print(window_start_date)
    print(current_window)

    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = current_window['ret4'].shift(-1).iloc[:-1]  # Rt+1:t+2
    X = sm.add_constant(current_window['Rt'].abs().iloc[:-1])  # (1, |Rt|)
    print(Y)
    print(X)

    # Schleife über Quantile
    for tau in quantiles:
        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=2
        recent_Rt = current_window['Rt'].iloc[-1]
        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles_week_1.loc[tau, 'predicted_ret1'] = prediction.iloc[0]
        pred_quantiles_week_1.loc[tau, 'start_date'] = window_start_date

    # Vorhersagen für die aktuelle Woche der Liste hinzufügen
    pred_quantiles_list_1.append(pred_quantiles_week_1)

# Ausgabe der Vorhersagen für jede Woche und jedes Quantil
for idx, pred_quantiles_week_1 in enumerate(pred_quantiles_list_1, start=1):
    print(f"Vorhersagen für Woche {idx}:")
    print(pred_quantiles_week_1)

#%%

'''
    Rolling Window Approach mit DATA SET und nicht LISTE
    h = 1
'''
import pandas as pd

window_size = 7  # Eine Woche
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung aller Vorhersagen für jede Woche und jedes Quantil
pred_quantiles_df_1 = pd.DataFrame(columns=['predicted_ret1', 'start_date', 'quantile'])

# Schleife über die Zeitpunkte im Testdatensatz
for i in range(len(test_data) - window_size + 1):
    # Aktueller Zeitpunkt im Rolling-Window
    current_window = test_data.iloc[i:i + window_size]
    window_start_date = current_window.index[0]

    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = current_window['ret1'].shift(-1).iloc[:-1]  # Rt+1:t+2
    X = sm.add_constant(current_window['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    # Schleife über Quantile
    for tau in quantiles:
        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=2
        recent_Rt = current_window['Rt'].iloc[-1]
        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Hinzufügen der Vorhersage zum DataFrame
        pred_quantiles_df_1 = pred_quantiles_df_1.append({'predicted_ret1': prediction.iloc[0],
                                                          'start_date': window_start_date,
                                                          'quantile': tau},
                                                         ignore_index=True)

# Ausgabe des DataFrame mit Vorhersagen für jede Woche und jedes Quantil
print("Vorhersagen für jede Woche und jedes Quantil:")
print(pred_quantiles_df_1)
  











#%%
'''
For h = 2
Quantile Predictions for the log returns ret2
'''
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles2 = pd.DataFrame(index=quantiles, columns=['predicted_ret2'])

# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = hist['ret2'].shift(-1).iloc[:-1]  # Rt+1:t+2
    X = sm.add_constant(hist['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    # Quantilregression
    quant_reg = sm.QuantReg(Y, X).fit(q=tau)

    # Vorhersagen für h=1
    recent_Rt = hist['Rt'].iloc[-1]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles2.loc[tau, 'predicted_ret2'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles2)

#%%
'''
Rolling Window Ansatz
h = 2
'''
window_size = 7  # Eine Woche
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# Liste zur Speicherung der Vorhersagen für jede Woche und jedes Quantil
pred_quantiles_list_2 = []

# Schleife über die Zeitpunkte im Testdatensatz
for i in range(len(test_data) - window_size + 1):
    print(test_data)
    print(len(test_data) - window_size + 1)
    # DataFrame zur Speicherung der Vorhersagen für die aktuelle Woche
    pred_quantiles_week_2 = pd.DataFrame(index=quantiles, columns=['predicted_ret2', 'start_date'])

    # Aktueller Zeitpunkt im Rolling-Window
    current_window = test_data.iloc[i:i + window_size]
    window_start_date = current_window.index[0]
    print(window_start_date)
    print(current_window)

    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = current_window['ret2'].shift(-1).iloc[:-1]  # Rt+1:t+2
    X = sm.add_constant(current_window['Rt'].abs().iloc[:-1])  # (1, |Rt|)
    print(Y)
    print(X)

    # Schleife über Quantile
    for tau in quantiles:
        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=2
        recent_Rt = current_window['Rt'].iloc[-1]
        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles_week_2.loc[tau, 'predicted_ret2'] = prediction.iloc[0]
        pred_quantiles_week_2.loc[tau, 'start_date'] = window_start_date

    # Vorhersagen für die aktuelle Woche der Liste hinzufügen
    pred_quantiles_list_2.append(pred_quantiles_week_2)

# Ausgabe der Vorhersagen für jede Woche und jedes Quantil
for idx, pred_quantiles_week_2 in enumerate(pred_quantiles_list_2, start=1):
    print(f"Vorhersagen für Woche {idx}:")
    print(pred_quantiles_week_2)

#%%

'''
    Rolling Window Approach mit DATA SET und nicht LISTE
    h = 2
'''
import pandas as pd

window_size = 7  # Eine Woche
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung aller Vorhersagen für jede Woche und jedes Quantil
pred_quantiles_df_2 = pd.DataFrame(columns=['predicted_ret2', 'start_date', 'quantile'])

# Schleife über die Zeitpunkte im Testdatensatz
for i in range(len(test_data) - window_size + 1):
    # Aktueller Zeitpunkt im Rolling-Window
    current_window = test_data.iloc[i:i + window_size]
    window_start_date = current_window.index[0]

    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = current_window['ret2'].shift(-1).iloc[:-1]  # Rt+1:t+2
    X = sm.add_constant(current_window['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    # Schleife über Quantile
    for tau in quantiles:
        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=2
        recent_Rt = current_window['Rt'].iloc[-1]
        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Hinzufügen der Vorhersage zum DataFrame
        pred_quantiles_df_2 = pred_quantiles_df_2.append({'predicted_ret2': prediction.iloc[0],
                                                          'start_date': window_start_date,
                                                          'quantile': tau},
                                                         ignore_index=True)

# Ausgabe des DataFrame mit Vorhersagen für jede Woche und jedes Quantil
print("Vorhersagen für jede Woche und jedes Quantil:")
print(pred_quantiles_df_2)
  






#%%
'''
For h = 3
Quantile Predictions for the log returns ret3
'''
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles3 = pd.DataFrame(index=quantiles, columns=['predicted_ret3'])

# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = hist['ret3'].shift(-1).iloc[:-1]  # Rt+1:t+3
    X = sm.add_constant(hist['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    # Quantilregression
    quant_reg = sm.QuantReg(Y, X).fit(q=tau)

    # Vorhersagen für h=1
    recent_Rt = hist['Rt'].iloc[-1]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles3.loc[tau, 'predicted_ret3'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles3)

#%%

'''
Rolling Window Ansatz
h = 3
'''
window_size = 7  # Eine Woche
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# Liste zur Speicherung der Vorhersagen für jede Woche und jedes Quantil
pred_quantiles_list_3 = []

# Schleife über die Zeitpunkte im Testdatensatz
for i in range(len(test_data) - window_size + 1):
    print(test_data)
    print(len(test_data) - window_size + 1)
    # DataFrame zur Speicherung der Vorhersagen für die aktuelle Woche
    pred_quantiles_week_3 = pd.DataFrame(index=quantiles, columns=['predicted_ret3', 'start_date'])

    # Aktueller Zeitpunkt im Rolling-Window
    current_window = test_data.iloc[i:i + window_size]
    window_start_date = current_window.index[0]
    print(window_start_date)
    print(current_window)

    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = current_window['ret3'].shift(-1).iloc[:-1]  # Rt+1:t+2
    X = sm.add_constant(current_window['Rt'].abs().iloc[:-1])  # (1, |Rt|)
    print(Y)
    print(X)

    # Schleife über Quantile
    for tau in quantiles:
        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=2
        recent_Rt = current_window['Rt'].iloc[-1]
        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles_week_3.loc[tau, 'predicted_ret3'] = prediction.iloc[0]
        pred_quantiles_week_3.loc[tau, 'start_date'] = window_start_date

    # Vorhersagen für die aktuelle Woche der Liste hinzufügen
    pred_quantiles_list_3.append(pred_quantiles_week_3)

# Ausgabe der Vorhersagen für jede Woche und jedes Quantil
for idx, pred_quantiles_week_3 in enumerate(pred_quantiles_list_3, start=1):
    print(f"Vorhersagen für Woche {idx}:")
    print(pred_quantiles_week_3)

#%%

'''
    Rolling Window Approach mit DATA SET und nicht LISTE
    h = 3
'''
import pandas as pd

window_size = 7  # Eine Woche
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung aller Vorhersagen für jede Woche und jedes Quantil
pred_quantiles_df_3 = pd.DataFrame(columns=['predicted_ret3', 'start_date', 'quantile'])

# Schleife über die Zeitpunkte im Testdatensatz
for i in range(len(test_data) - window_size + 1):
    # Aktueller Zeitpunkt im Rolling-Window
    current_window = test_data.iloc[i:i + window_size]
    window_start_date = current_window.index[0]

    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = current_window['ret3'].shift(-1).iloc[:-1]  # Rt+1:t+2
    X = sm.add_constant(current_window['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    # Schleife über Quantile
    for tau in quantiles:
        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=2
        recent_Rt = current_window['Rt'].iloc[-1]
        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Hinzufügen der Vorhersage zum DataFrame
        pred_quantiles_df_3 = pred_quantiles_df_3.append({'predicted_ret3': prediction.iloc[0],
                                                          'start_date': window_start_date,
                                                          'quantile': tau},
                                                         ignore_index=True)

# Ausgabe des DataFrame mit Vorhersagen für jede Woche und jedes Quantil
print("Vorhersagen für jede Woche und jedes Quantil:")
print(pred_quantiles_df_3)
  








#%%
'''
For h = 4
Quantile Predictions for the log returns ret4
'''
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles4 = pd.DataFrame(index=quantiles, columns=['predicted_ret4'])

# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = hist['ret4'].shift(-1).iloc[:-1]  # Rt+1:t+4
    X = sm.add_constant(hist['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    # Quantilregression
    quant_reg = sm.QuantReg(Y, X).fit(q=tau)

    # Vorhersagen für h=1
    recent_Rt = hist['Rt'].iloc[-1]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles4.loc[tau, 'predicted_ret4'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles4)


#%%
'''
Rolling Window Ansatz
h = 4
'''
window_size = 7  # Eine Woche
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# Liste zur Speicherung der Vorhersagen für jede Woche und jedes Quantil
pred_quantiles_list_4 = []

# Schleife über die Zeitpunkte im Testdatensatz
for i in range(len(test_data) - window_size + 1):
    print(test_data)
    print(len(test_data) - window_size + 1)
    # DataFrame zur Speicherung der Vorhersagen für die aktuelle Woche
    pred_quantiles_week_4 = pd.DataFrame(index=quantiles, columns=['predicted_ret4', 'start_date'])

    # Aktueller Zeitpunkt im Rolling-Window
    current_window = test_data.iloc[i:i + window_size]
    window_start_date = current_window.index[0]
    print(window_start_date)
    print(current_window)

    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = current_window['ret4'].shift(-1).iloc[:-1]  # Rt+1:t+2
    X = sm.add_constant(current_window['Rt'].abs().iloc[:-1])  # (1, |Rt|)
    print(Y)
    print(X)

    # Schleife über Quantile
    for tau in quantiles:
        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=2
        recent_Rt = current_window['Rt'].iloc[-1]
        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles_week_4.loc[tau, 'predicted_ret4'] = prediction.iloc[0]
        pred_quantiles_week_4.loc[tau, 'start_date'] = window_start_date

    # Vorhersagen für die aktuelle Woche der Liste hinzufügen
    pred_quantiles_list_4.append(pred_quantiles_week_4)

# Ausgabe der Vorhersagen für jede Woche und jedes Quantil
for idx, pred_quantiles_week_4 in enumerate(pred_quantiles_list_4, start=1):
    print(f"Vorhersagen für Woche {idx}:")
    print(pred_quantiles_week_4)


#%%
'''
    Rolling Window Approach mit DATA SET und nicht LISTE
    h = 4
'''
import pandas as pd

window_size = 7  # Eine Woche
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung aller Vorhersagen für jede Woche und jedes Quantil
pred_quantiles_df_4 = pd.DataFrame(columns=['predicted_ret4', 'start_date', 'quantile'])

# Schleife über die Zeitpunkte im Testdatensatz
for i in range(len(test_data) - window_size + 1):
    # Aktueller Zeitpunkt im Rolling-Window
    current_window = test_data.iloc[i:i + window_size]
    window_start_date = current_window.index[0]

    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = current_window['ret5'].shift(-1).iloc[:-1]  # Rt+1:t+2
    X = sm.add_constant(current_window['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    # Schleife über Quantile
    for tau in quantiles:
        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=2
        recent_Rt = current_window['Rt'].iloc[-1]
        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Hinzufügen der Vorhersage zum DataFrame
        pred_quantiles_df_4 = pred_quantiles_df_4.append({'predicted_ret4': prediction.iloc[0],
                                                          'start_date': window_start_date,
                                                          'quantile': tau},
                                                         ignore_index=True)

# Ausgabe des DataFrame mit Vorhersagen für jede Woche und jedes Quantil
print("Vorhersagen für jede Woche und jedes Quantil:")
print(pred_quantiles_df_4)
      

#%%
'''
For h = 5
Quantile Predictions for the log returns ret5
'''
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles5 = pd.DataFrame(index=quantiles, columns=['predicted_ret5'])

# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = hist['ret5'].shift(-1).iloc[:-1]  # Rt+1:t+5
    X = sm.add_constant(hist['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    # Quantilregression
    quant_reg = sm.QuantReg(Y, X).fit(q=tau)

    # Vorhersagen für h=1
    recent_Rt = hist['Rt'].iloc[-1]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles5.loc[tau, 'predicted_ret5'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles5)


#%%

'''
Rolling Window Ansatz
h = 5
'''
window_size = 7  # Eine Woche
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# Liste zur Speicherung der Vorhersagen für jede Woche und jedes Quantil
pred_quantiles_list_5 = []

# Schleife über die Zeitpunkte im Testdatensatz
for i in range(len(test_data) - window_size + 1):
    print(test_data)
    print(len(test_data) - window_size + 1)
    # DataFrame zur Speicherung der Vorhersagen für die aktuelle Woche
    pred_quantiles_week_5 = pd.DataFrame(index=quantiles, columns=['predicted_ret5', 'start_date'])

    # Aktueller Zeitpunkt im Rolling-Window
    current_window = test_data.iloc[i:i + window_size]
    window_start_date = current_window.index[0]
    print(window_start_date)
    print(current_window)

    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = current_window['ret5'].shift(-1).iloc[:-1]  # Rt+1:t+2
    X = sm.add_constant(current_window['Rt'].abs().iloc[:-1])  # (1, |Rt|)
    print(Y)
    print(X)

    # Schleife über Quantile
    for tau in quantiles:
        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=2
        recent_Rt = current_window['Rt'].iloc[-1]
        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles_week_5.loc[tau, 'predicted_ret5'] = prediction.iloc[0]
        pred_quantiles_week_5.loc[tau, 'start_date'] = window_start_date

    # Vorhersagen für die aktuelle Woche der Liste hinzufügen
    pred_quantiles_list_5.append(pred_quantiles_week_5)

# Ausgabe der Vorhersagen für jede Woche und jedes Quantil
for idx, pred_quantiles_week_5 in enumerate(pred_quantiles_list_5, start=1):
    print(f"Vorhersagen für Woche {idx}:")
    print(pred_quantiles_week_5)


#%%
'''
    Rolling Window Approach mit DATA SET und nicht LISTE
'''
import pandas as pd

window_size = 7  # Eine Woche
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung aller Vorhersagen für jede Woche und jedes Quantil
pred_quantiles_df_5 = pd.DataFrame(columns=['predicted_ret5', 'start_date', 'quantile'])

# Schleife über die Zeitpunkte im Testdatensatz
for i in range(len(test_data) - window_size + 1):
    # Aktueller Zeitpunkt im Rolling-Window
    current_window = test_data.iloc[i:i + window_size]
    window_start_date = current_window.index[0]

    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y = current_window['ret5'].shift(-1).iloc[:-1]  # Rt+1:t+2
    X = sm.add_constant(current_window['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    # Schleife über Quantile
    for tau in quantiles:
        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=2
        recent_Rt = current_window['Rt'].iloc[-1]
        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Hinzufügen der Vorhersage zum DataFrame
        pred_quantiles_df_5 = pred_quantiles_df_5.append({'predicted_ret5': prediction.iloc[0],
                                                          'start_date': window_start_date,
                                                          'quantile': tau},
                                                         ignore_index=True)

# Ausgabe des DataFrame mit Vorhersagen für jede Woche und jedes Quantil
print("Vorhersagen für jede Woche und jedes Quantil:")
print(pred_quantiles_df_5)
  

#%%
'''
Datensätze zusammenfügen
=> Rolling Window Spalten Format
------
da in den Testdaten 36 Tage sind und wir diese vorhersagen
'''
pred_quantiles_combined = []

for df1, df2, df3, df4, df5 in zip(pred_quantiles_list_1, pred_quantiles_list_2, pred_quantiles_list_3, pred_quantiles_list_4, pred_quantiles_list_5):
    df_combined = pd.concat([df1, df2, df3, df4, df5], axis=1)
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]  # Entfernen duplizierter Spalten
    pred_quantiles_combined.append(df_combined)

# Anzeigen des ersten kombinierten DataFrames
print(pred_quantiles_combined[0])

#%%
import pandas as pd

# Iteriere über jedes DataFrame in pred_quantiles_combined
for df in pred_quantiles_combined:
    # Ändere den Index
    df.reset_index(inplace=True)
    # Benenne die Spalte 'index' in 'quantile' um
    df.rename(columns={'index': 'quantile'}, inplace=True)
    # Setze 'start_date' als Index
    df.set_index('start_date', inplace=True)

    # Gib das aktualisierte DataFrame aus
    print(df)

#%%

'''
Datensätze zusammenfügen
=> Datensatz format
'''
forecast = [pred_quantiles_df_1, pred_quantiles_df_2, pred_quantiles_df_3, pred_quantiles_df_4, pred_quantiles_df_5]

# Zusammenführen der DataFrames entlang der Zeilenachse (axis=0)
c_df = pd.concat(forecast, axis=1)

# Ergebnis ausgeben
print(c_df)

#%%
'''
doppelte spalten entfernen
'''
duplicates = c_df.columns[c_df.columns.duplicated()]

# Entfernen doppelter Spalten
combined_df = c_df.loc[:, ~c_df.columns.duplicated()]

# Ergebnis ausgeben
print(combined_df)

#%%
combined_df.set_index('start_date', inplace=True)
combined_df.index.name = 'Date'

# Ergebnis ausgeben
print(combined_df)

#%%
dax_quantiles = pd.merge(test_data, combined_df, left_index=True, right_index=True)
dax_quantiles

del dax_quantiles['price_up']
del dax_quantiles['ret1_up']
del dax_quantiles['ret2_up']
del dax_quantiles['ret3_up']
del dax_quantiles['ret4_up']
del dax_quantiles['ret5_up']

#%%
subset_ret1 = dax_quantiles[['ret1', 'predicted_ret1', 'quantile']]
subset_ret2 = dax_quantiles[['ret2', 'predicted_ret2', 'quantile']]
subset_ret3 = dax_quantiles[['ret3', 'predicted_ret3', 'quantile']]
subset_ret4 = dax_quantiles[['ret4', 'predicted_ret4', 'quantile']]
subset_ret5 = dax_quantiles[['ret5', 'predicted_ret5', 'quantile']]

#%%
def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        # Fall, in dem q_hat gleich y ist (Randfall)
        return 0

print(quantile_score(0.932793,-0.168307, 0.975))
#%%
import pandas as pd

def quantile_score(row):
    q_hat = row['predicted_ret1']
    y = row['ret1']
    tau = row['quantile']
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        return 0

# Nehmen Sie an, Ihr DataFrame heißt 'dax_quantiles'
# Fügen Sie eine neue Spalte 'quantile_score' hinzu, indem Sie die Funktion quantile_score auf jeden Zeile anwenden
dax_quantiles['quantile_score_1'] = dax_quantiles.apply(quantile_score, axis=1)

# Ausgabe des DataFrame mit der neuen Spalte 'quantile_score'
print(dax_quantiles)
#%%
import pandas as pd

def quantile_score(row):
    q_hat = row['predicted_ret2']
    y = row['ret2']
    tau = row['quantile']
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        return 0

# Nehmen Sie an, Ihr DataFrame heißt 'dax_quantiles'
# Fügen Sie eine neue Spalte 'quantile_score' hinzu, indem Sie die Funktion quantile_score auf jeden Zeile anwenden
dax_quantiles['quantile_score_2'] = dax_quantiles.apply(quantile_score, axis=1)

# Ausgabe des DataFrame mit der neuen Spalte 'quantile_score'
print(dax_quantiles)


#%%
import pandas as pd

def quantile_score(row):
    q_hat = row['predicted_ret3']
    y = row['ret3']
    tau = row['quantile']
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        return 0

# Nehmen Sie an, Ihr DataFrame heißt 'dax_quantiles'
# Fügen Sie eine neue Spalte 'quantile_score' hinzu, indem Sie die Funktion quantile_score auf jeden Zeile anwenden
dax_quantiles['quantile_score_3'] = dax_quantiles.apply(quantile_score, axis=1)

# Ausgabe des DataFrame mit der neuen Spalte 'quantile_score'
print(dax_quantiles)
#%%

import pandas as pd

def quantile_score(row):
    q_hat = row['predicted_ret4']
    y = row['ret4']
    tau = row['quantile']
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        return 0

# Nehmen Sie an, Ihr DataFrame heißt 'dax_quantiles'
# Fügen Sie eine neue Spalte 'quantile_score' hinzu, indem Sie die Funktion quantile_score auf jeden Zeile anwenden
dax_quantiles['quantile_score_4'] = dax_quantiles.apply(quantile_score, axis=1)

# Ausgabe des DataFrame mit der neuen Spalte 'quantile_score'
print(dax_quantiles)



#%%

import pandas as pd

def quantile_score(row):
    q_hat = row['predicted_ret5']
    y = row['ret5']
    tau = row['quantile']
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        return 0

# Nehmen Sie an, Ihr DataFrame heißt 'dax_quantiles'
# Fügen Sie eine neue Spalte 'quantile_score' hinzu, indem Sie die Funktion quantile_score auf jeden Zeile anwenden
dax_quantiles['quantile_score_5'] = dax_quantiles.apply(quantile_score, axis=1)

# Ausgabe des DataFrame mit der neuen Spalte 'quantile_score'
print(dax_quantiles)
#%%


data_ret1 = dax_quantiles[['ret1', 'predicted_ret1', 'quantile', 'quantile_score_1']]
data_ret2 = dax_quantiles[['ret2', 'predicted_ret2', 'quantile',  'quantile_score_2']]
data_ret3 = dax_quantiles[['ret3', 'predicted_ret3', 'quantile',  'quantile_score_3']]
data_ret4 = dax_quantiles[['ret4', 'predicted_ret4', 'quantile',  'quantile_score_4']]
data_ret5 = dax_quantiles[['ret5', 'predicted_ret5', 'quantile',  'quantile_score_5']]

#%%
grouped1 = data_ret1.groupby('quantile').mean()
grouped2 = data_ret2.groupby('quantile').mean()
grouped3 = data_ret3.groupby('quantile').mean()
grouped4 = data_ret4.groupby('quantile').mean()
grouped5 = data_ret5.groupby('quantile').mean()

#%%
grouped1
#%%
grouped2
#%%
grouped3 
#%%
grouped4 
#%%
grouped5 
#%%

'''Coverage Probability'''
''' Ret 1'''

data = subset_ret1
# Initialisierung der leeren Listen für die Coverage-Wahrscheinlichkeiten
coverage_probabilities_50 = []
coverage_probabilities_95 = []

# Iteration über jedes eindeutige Datum
for date in data.index.unique():
    # Auswahl der Daten für das aktuelle Datum
    subset = data.loc[date]
    
    # Überprüfung der Abdeckung für 95% Konfidenzintervalle
    coverage_95 = ((subset['ret1'] >= subset[subset['quantile'] == 0.025]['predicted_ret1'].iloc[0]) & 
                   (subset['ret1'] <= subset[subset['quantile'] == 0.975]['predicted_ret1'].iloc[0])).mean()
    coverage_probabilities_95.append(coverage_95)
    
    # Überprüfung der Abdeckung für 50% Konfidenzintervalle
    coverage_50 = ((subset['ret1'] >= subset[subset['quantile'] == 0.25]['predicted_ret1'].iloc[0]) & 
                   (subset['ret1'] <= subset[subset['quantile'] == 0.75]['predicted_ret1'].iloc[0])).mean()
    coverage_probabilities_50.append(coverage_50)

# Ausgabe der Coverage-Wahrscheinlichkeiten für jedes Datum
for i, date in enumerate(data.index.unique()):
    print(f"Datum: {date}")
    print("Coverage-Wahrscheinlichkeit für 50% Konfidenzintervalle:", coverage_probabilities_50[i])
    print("Coverage-Wahrscheinlichkeit für 95% Konfidenzintervalle:", coverage_probabilities_95[i])
    print()
average_coverage_probability_50 = np.mean(coverage_probabilities_50)
average_coverage_probability_95 = np.mean(coverage_probabilities_95)

print("Durchschnittliche Coverage-Wahrscheinlichkeit für 50% Konfidenzintervalle:", average_coverage_probability_50)
print("Durchschnittliche Coverage-Wahrscheinlichkeit für 95% Konfidenzintervalle:", average_coverage_probability_95)

#%%

'''Coverage Probability'''
''' Ret 2'''

data = subset_ret2
# Initialisierung der leeren Listen für die Coverage-Wahrscheinlichkeiten
coverage_probabilities_50 = []
coverage_probabilities_95 = []

# Iteration über jedes eindeutige Datum
for date in data.index.unique():
    # Auswahl der Daten für das aktuelle Datum
    subset = data.loc[date]
    
    # Überprüfung der Abdeckung für 95% Konfidenzintervalle
    coverage_95 = ((subset['ret2'] >= subset[subset['quantile'] == 0.025]['predicted_ret2'].iloc[0]) & 
                   (subset['ret2'] <= subset[subset['quantile'] == 0.975]['predicted_ret2'].iloc[0])).mean()
    coverage_probabilities_95.append(coverage_95)
    
    # Überprüfung der Abdeckung für 50% Konfidenzintervalle
    coverage_50 = ((subset['ret2'] >= subset[subset['quantile'] == 0.25]['predicted_ret2'].iloc[0]) & 
                   (subset['ret2'] <= subset[subset['quantile'] == 0.75]['predicted_ret2'].iloc[0])).mean()
    coverage_probabilities_50.append(coverage_50)

# Ausgabe der Coverage-Wahrscheinlichkeiten für jedes Datum
for i, date in enumerate(data.index.unique()):
    print(f"Datum: {date}")
    print("Coverage-Wahrscheinlichkeit für 50% Konfidenzintervalle:", coverage_probabilities_50[i])
    print("Coverage-Wahrscheinlichkeit für 95% Konfidenzintervalle:", coverage_probabilities_95[i])
    print()
average_coverage_probability_50 = np.mean(coverage_probabilities_50)
average_coverage_probability_95 = np.mean(coverage_probabilities_95)

print("Durchschnittliche Coverage-Wahrscheinlichkeit für 50% Konfidenzintervalle:", average_coverage_probability_50)
print("Durchschnittliche Coverage-Wahrscheinlichkeit für 95% Konfidenzintervalle:", average_coverage_probability_95)



#%%

'''Coverage Probability'''
''' Ret 3'''

data = subset_ret3
# Initialisierung der leeren Listen für die Coverage-Wahrscheinlichkeiten
coverage_probabilities_50 = []
coverage_probabilities_95 = []

# Iteration über jedes eindeutige Datum
for date in data.index.unique():
    # Auswahl der Daten für das aktuelle Datum
    subset = data.loc[date]
    
    # Überprüfung der Abdeckung für 95% Konfidenzintervalle
    coverage_95 = ((subset['ret3'] >= subset[subset['quantile'] == 0.025]['predicted_ret3'].iloc[0]) & 
                   (subset['ret3'] <= subset[subset['quantile'] == 0.975]['predicted_ret3'].iloc[0])).mean()
    coverage_probabilities_95.append(coverage_95)
    
    # Überprüfung der Abdeckung für 50% Konfidenzintervalle
    coverage_50 = ((subset['ret3'] >= subset[subset['quantile'] == 0.25]['predicted_ret3'].iloc[0]) & 
                   (subset['ret3'] <= subset[subset['quantile'] == 0.75]['predicted_ret3'].iloc[0])).mean()
    coverage_probabilities_50.append(coverage_50)

# Ausgabe der Coverage-Wahrscheinlichkeiten für jedes Datum
for i, date in enumerate(data.index.unique()):
    print(f"Datum: {date}")
    print("Coverage-Wahrscheinlichkeit für 50% Konfidenzintervalle:", coverage_probabilities_50[i])
    print("Coverage-Wahrscheinlichkeit für 95% Konfidenzintervalle:", coverage_probabilities_95[i])
    print()
average_coverage_probability_50 = np.mean(coverage_probabilities_50)
average_coverage_probability_95 = np.mean(coverage_probabilities_95)

print("Durchschnittliche Coverage-Wahrscheinlichkeit für 50% Konfidenzintervalle:", average_coverage_probability_50)
print("Durchschnittliche Coverage-Wahrscheinlichkeit für 95% Konfidenzintervalle:", average_coverage_probability_95)
#%%

'''Coverage Probability'''
''' Ret 4'''

data = subset_ret4
# Initialisierung der leeren Listen für die Coverage-Wahrscheinlichkeiten
coverage_probabilities_50 = []
coverage_probabilities_95 = []

# Iteration über jedes eindeutige Datum
for date in data.index.unique():
    # Auswahl der Daten für das aktuelle Datum
    subset = data.loc[date]
    
    # Überprüfung der Abdeckung für 95% Konfidenzintervalle
    coverage_95 = ((subset['ret4'] >= subset[subset['quantile'] == 0.025]['predicted_ret4'].iloc[0]) & 
                   (subset['ret4'] <= subset[subset['quantile'] == 0.975]['predicted_ret4'].iloc[0])).mean()
    coverage_probabilities_95.append(coverage_95)
    
    # Überprüfung der Abdeckung für 50% Konfidenzintervalle
    coverage_50 = ((subset['ret4'] >= subset[subset['quantile'] == 0.25]['predicted_ret4'].iloc[0]) & 
                   (subset['ret4'] <= subset[subset['quantile'] == 0.75]['predicted_ret4'].iloc[0])).mean()
    coverage_probabilities_50.append(coverage_50)

# Ausgabe der Coverage-Wahrscheinlichkeiten für jedes Datum
for i, date in enumerate(data.index.unique()):
    print(f"Datum: {date}")
    print("Coverage-Wahrscheinlichkeit für 50% Konfidenzintervalle:", coverage_probabilities_50[i])
    print("Coverage-Wahrscheinlichkeit für 95% Konfidenzintervalle:", coverage_probabilities_95[i])
    print()
average_coverage_probability_50 = np.mean(coverage_probabilities_50)
average_coverage_probability_95 = np.mean(coverage_probabilities_95)

print("Durchschnittliche Coverage-Wahrscheinlichkeit für 50% Konfidenzintervalle:", average_coverage_probability_50)
print("Durchschnittliche Coverage-Wahrscheinlichkeit für 95% Konfidenzintervalle:", average_coverage_probability_95)

#%%
'''Coverage Probability'''
''' Ret 5'''

data = subset_ret5
# Initialisierung der leeren Listen für die Coverage-Wahrscheinlichkeiten
coverage_probabilities_50 = []
coverage_probabilities_95 = []

# Iteration über jedes eindeutige Datum
for date in data.index.unique():
    # Auswahl der Daten für das aktuelle Datum
    subset = data.loc[date]
    
    # Überprüfung der Abdeckung für 95% Konfidenzintervalle
    coverage_95 = ((subset['ret5'] >= subset[subset['quantile'] == 0.025]['predicted_ret5'].iloc[0]) & 
                   (subset['ret5'] <= subset[subset['quantile'] == 0.975]['predicted_ret5'].iloc[0])).mean()
    coverage_probabilities_95.append(coverage_95)
    
    # Überprüfung der Abdeckung für 50% Konfidenzintervalle
    coverage_50 = ((subset['ret5'] >= subset[subset['quantile'] == 0.25]['predicted_ret5'].iloc[0]) & 
                   (subset['ret5'] <= subset[subset['quantile'] == 0.75]['predicted_ret5'].iloc[0])).mean()
    coverage_probabilities_50.append(coverage_50)

# Ausgabe der Coverage-Wahrscheinlichkeiten für jedes Datum
for i, date in enumerate(data.index.unique()):
    print(f"Datum: {date}")
    print("Coverage-Wahrscheinlichkeit für 50% Konfidenzintervalle:", coverage_probabilities_50[i])
    print("Coverage-Wahrscheinlichkeit für 95% Konfidenzintervalle:", coverage_probabilities_95[i])
    print()
average_coverage_probability_50 = np.mean(coverage_probabilities_50)
average_coverage_probability_95 = np.mean(coverage_probabilities_95)

print("Durchschnittliche Coverage-Wahrscheinlichkeit für 50% Konfidenzintervalle:", average_coverage_probability_50)
print("Durchschnittliche Coverage-Wahrscheinlichkeit für 95% Konfidenzintervalle:", average_coverage_probability_95)



#%%

'''
----------
Sharpness
ret1
----------
'''
from scipy.stats import entropy
data = subset_ret1
# Berechnung der Schärfe der 50% und 95% Konfidenzintervalle
# Berechnung der Schärfe der 50% und 95% Konfidenzintervalle
sharpness_50 = ((data[data['quantile'] == 0.75]['predicted_ret1'] - data[data['quantile'] == 0.25]['predicted_ret1']) ** 2).mean()
sharpness_95 = ((data[data['quantile'] == 0.975]['predicted_ret1'] - data[data['quantile'] == 0.025]['predicted_ret1']) ** 2).mean()

print("Schärfe des 50% Konfidenzintervalls:", sharpness_50)
print("Schärfe des 95% Konfidenzintervalls:", sharpness_95)



#%%

'''
----------
Sharpness
ret2
----------
'''
from scipy.stats import entropy
data = subset_ret2
# Berechnung der Schärfe der 50% und 95% Konfidenzintervalle
# Berechnung der Schärfe der 50% und 95% Konfidenzintervalle
sharpness_50 = ((data[data['quantile'] == 0.75]['predicted_ret1'] - data[data['quantile'] == 0.25]['predicted_ret1']) ** 2).mean()
sharpness_95 = ((data[data['quantile'] == 0.975]['predicted_ret1'] - data[data['quantile'] == 0.025]['predicted_ret1']) ** 2).mean()

print("Schärfe des 50% Konfidenzintervalls:", sharpness_50)
print("Schärfe des 95% Konfidenzintervalls:", sharpness_95)

#%%
'''
----------
Sharpness
ret3
----------
'''
from scipy.stats import entropy
data = subset_ret3
# Berechnung der Schärfe der 50% und 95% Konfidenzintervalle
# Berechnung der Schärfe der 50% und 95% Konfidenzintervalle
sharpness_50 = ((data[data['quantile'] == 0.75]['predicted_ret1'] - data[data['quantile'] == 0.25]['predicted_ret1']) ** 2).mean()
sharpness_95 = ((data[data['quantile'] == 0.975]['predicted_ret1'] - data[data['quantile'] == 0.025]['predicted_ret1']) ** 2).mean()

print("Schärfe des 50% Konfidenzintervalls:", sharpness_50)
print("Schärfe des 95% Konfidenzintervalls:", sharpness_95)


#%%
'''
----------
Sharpness
ret4
----------
'''
from scipy.stats import entropy
data = subset_ret4
# Berechnung der Schärfe der 50% und 95% Konfidenzintervalle
# Berechnung der Schärfe der 50% und 95% Konfidenzintervalle
sharpness_50 = ((data[data['quantile'] == 0.75]['predicted_ret1'] - data[data['quantile'] == 0.25]['predicted_ret1']) ** 2).mean()
sharpness_95 = ((data[data['quantile'] == 0.975]['predicted_ret1'] - data[data['quantile'] == 0.025]['predicted_ret1']) ** 2).mean()

print("Schärfe des 50% Konfidenzintervalls:", sharpness_50)
print("Schärfe des 95% Konfidenzintervalls:", sharpness_95)

#%%
'''
----------
Sharpness
ret5
----------
'''
from scipy.stats import entropy
data = subset_ret5
# Berechnung der Schärfe der 50% und 95% Konfidenzintervalle
# Berechnung der Schärfe der 50% und 95% Konfidenzintervalle
sharpness_50 = ((data[data['quantile'] == 0.75]['predicted_ret1'] - data[data['quantile'] == 0.25]['predicted_ret1']) ** 2).mean()
sharpness_95 = ((data[data['quantile'] == 0.975]['predicted_ret1'] - data[data['quantile'] == 0.025]['predicted_ret1']) ** 2).mean()

print("Schärfe des 50% Konfidenzintervalls:", sharpness_50)
print("Schärfe des 95% Konfidenzintervalls:", sharpness_95)



#%%
import numpy as np

def quantile_score(q_hat, y, tau):
    loss = np.where(q_hat > y, 2 * (1 - tau) * (q_hat - y), 2 * tau * (y - q_hat))
    return loss

quantile_loss_list = []

# Iteration über jedes DataFrame in pred_quantiles_combined
for combined_df in pred_quantiles_combined:
    # Extrahiere das Startdatum der Woche aus dem Index des DataFrames
    week_start_date = combined_df.index[0]
    print(week_start_date)

    # Extrahiere die vorhergesagten Werte für diese Woche
    predicted_values = combined_df.drop('quantile', axis=1)
    print(predicted_values)

    # Extrahiere die wahren Werte für diese Woche aus den Testdaten
    true_values = test_data.loc[week_start_date, ['ret1', 'ret2', 'ret3', 'ret4', 'ret5']]
    print(true_values)

    quantiles = combined_df['quantile'].tolist()  # Holen Sie sich die Quantile aus dem DataFrame
    print(quantiles)

    # Berechne den Quantile Loss für jede Quantile und jede Rendite für diese Woche
    week_loss = []
    for i, quantile in enumerate(quantiles, start=1):
        # Extrahiere die vorhergesagten Werte für das aktuelle Quantil
        q_hat = predicted_values[f'predicted_ret{i}']
        print(q_hat)

        # Berechne den Quantile Loss für jede Rendite
        for j in range(1, 6):
            y = true_values[f'ret{j}']
            print(y)
            loss = quantile_score(q_hat, y, quantile)
            print(loss)
            week_loss.append(loss)

    # Füge die Verlustwerte dieser Woche zur Liste hinzu
    quantile_loss_list.append(week_loss)




# %%
'''
For h = 2
'''
pred_quantiles2 = pd.DataFrame(index=quantiles, columns=['predicted_ret2'])

#quantiles = [0.025, 0.250, 0.50, 0.750, 0.975]
for tau in quantiles:
    # Y = ret1 + ret2, X = (1, |Rt|)
    Y = hist['ret1'].shift(-1).iloc[:-1] + hist['ret2'].shift(-2).iloc[:-1]  # ret1 + ret2
    Y = Y.dropna()
    X = sm.add_constant(hist['Rt'].abs().iloc[:-1].loc[Y.index]) 

    # Quantilregression
    quant_reg = sm.QuantReg(Y, X).fit(q=tau)

    # Vorhersagen für h=2
    recent_Rt = hist['Rt'].iloc[-1]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles2.loc[tau, 'predicted_ret2'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles2)
# %%
'''
For h = 3
'''

quantiles = [0.025, 0.250, 0.50, 0.750, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles3 = pd.DataFrame(index=quantiles, columns=['predicted_ret3'])

# Schleife über Quantile
for tau in quantiles:
    # Y = ret1 + ret2 + ret3, X = (1, |Rt|)
    Y3 = hist['ret1'].shift(-1).iloc[:-1] + hist['ret2'].shift(-2).iloc[:-2] + hist['ret3'].shift(-3).iloc[:-3]  # ret1 + ret2 + ret3
    Y3 = Y3.dropna()  # Entfernen von NaN-Werten in Y
    X = sm.add_constant(hist['Rt'].abs().iloc[:-3].loc[Y3.index])  # (1, |Rt|) mit den gleichen Indizes wie Y

    # Quantilregression
    quant_reg = sm.QuantReg(Y3, X).fit(q=tau)

    # Vorhersagen für h=3
    recent_Rt = hist['Rt'].iloc[-3]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles3.loc[tau, 'predicted_ret3'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles3)


# %%
'''
For h = 4
'''
pred_quantiles4 = pd.DataFrame(index=quantiles, columns=['predicted_ret4'])

# Schleife über Quantile
for tau in quantiles:
    # Y = ret1 + ret2 + ret3, X = (1, |Rt|)
    Y4 = hist['ret1'].shift(-1).iloc[:-1] + hist['ret2'].shift(-2).iloc[:-2] + hist['ret3'].shift(-3).iloc[:-3] + hist['ret4'].shift(-4).iloc[:-4]  # ret1 + ret2 + ret3
    Y4 = Y4.dropna()  # Entfernen von NaN-Werten in Y
    X = sm.add_constant(hist['Rt'].abs().iloc[:-4].loc[Y4.index])  # (1, |Rt|) mit den gleichen Indizes wie Y

    # Quantilregression
    quant_reg = sm.QuantReg(Y4, X).fit(q=tau)

    # Vorhersagen für h=3
    recent_Rt = hist['Rt'].iloc[-4]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles4.loc[tau, 'predicted_ret4'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles4)
# %%
'''
For h = 5
'''

pred_quantiles5 = pd.DataFrame(index=quantiles, columns=['predicted_ret5'])

# Schleife über Quantile
for tau in quantiles:
    # Y = ret1 + ret2 + ret3, X = (1, |Rt|)
    Y5 = hist['ret1'].shift(-1).iloc[:-1] + hist['ret2'].shift(-2).iloc[:-2] + hist['ret3'].shift(-3).iloc[:-3] + hist['ret4'].shift(-4).iloc[:-4] + hist['ret5'].shift(-5)  # ret1 + ret2 + ret3
    Y5 = Y5.dropna()  # Entfernen von NaN-Werten in Y
    X = sm.add_constant(hist['Rt'].abs().iloc[:-5].loc[Y5.index])  # (1, |Rt|) mit den gleichen Indizes wie Y

    # Quantilregression
    quant_reg = sm.QuantReg(Y5, X).fit(q=tau)

    # Vorhersagen für h=3
    recent_Rt = hist['Rt'].iloc[-5]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles5.loc[tau, 'predicted_ret5'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles5)
# %%
pred1 = pred_quantiles.values
pred2 = pred_quantiles2['predicted_ret2'].values
pred3 = pred_quantiles3['predicted_ret3'].values
pred4 = pred_quantiles4['predicted_ret4'].values
pred5 = pred_quantiles5['predicted_ret5'].values
#%%
pred1 = pred1.reshape(-1,1)
pred2 = pred2.reshape(-1,1)
pred3 = pred3.reshape(-1,1)
pred4 = pred4.reshape(-1,1)
pred5 = pred5.reshape(-1,1)

#%%
forecasts = np.column_stack((pred1, pred2, pred3, pred4, pred5))
#%%
forecasts_sub = forecasts.T
#%%
from datetime import datetime
#%%
forecastdate = datetime(2024, 1, 10, 00, 00)
# %%

df_sub_dax = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "DAX",
    "horizon": [str(i) + " day" for i in (1,2,5,6,7)],
    "q0.025": forecasts_sub[:,0],
    "q0.25": forecasts_sub[:,1],
    "q0.5": forecasts_sub[:,2],
    "q0.75": forecasts_sub[:,3],
    "q0.975": forecasts_sub[:,4]})
df_sub_dax


#%%
PATH = "/Users/sophiasiefert/Documents/Vorlesungen /Master/PTFSC/PTSFC Submission JahrMonatTag_Rachel"

df_sub_dax.to_csv(PATH+ "Quantile_Regression_DAX_Abgabe7.csv", index=False)


# %%

'''
Plotting the Quantile Predictions for the returns
'''
#%%
'''
-––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
QUANTILE REGRESSION WITH FURTHER EXPLANATORY VARIABLES FOR RETURN PREDICTIONS
=> Wether the price went up or down
=> wether the returns went up or down
---------------------------------------------------------------------------------------
'''

'''
For h = 1 
Quantile Predictions for the log returns ret1
with further explanatory variables 
X => 1, |Rt|, price_up, ret1_up
'''
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles = pd.DataFrame(index=quantiles, columns=['predicted_ret1'])

# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+1, X = (1, |Rt|, price_up, return_up)
    Y = hist['ret1'].shift(-1).iloc[:-1]  # Renditen für t+1
    X = sm.add_constant(hist[['Rt', 'price_up', 'ret1_up']].abs().iloc[:-1])  # (1, |Rt|, price_up, return_up)

    # Quantilregression
    quant_reg = sm.QuantReg(Y, X).fit(q=tau)

    # Vorhersagen für h=1
    recent_Rt = hist['Rt'].iloc[-1]
    recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
    recent_return_up = 1 if hist['ret1'].iloc[-1] > 0 else 0
    

    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret1_up': recent_return_up}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles.loc[tau, 'predicted_ret1'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles)

#%%
'''
For h = 2 
Quantile Predictions for the log returns ret1
with further explanatory variables 
X => 1, |Rt|, price_up, ret1_up
'''
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles2 = pd.DataFrame(index=quantiles, columns=['predicted_ret2'])

# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+1, X = (1, |Rt|, price_up, return_up)
    Y = hist['ret2'].shift(-1).iloc[:-1]  # Renditen für t+1
    X = sm.add_constant(hist[['Rt', 'price_up', 'ret2_up']].abs().iloc[:-1])  # (1, |Rt|, price_up, return_up)
    

    # Quantilregression
    quant_reg = sm.QuantReg(Y, X).fit(q=tau)

    # Vorhersagen für h=1
    recent_Rt = hist['Rt'].iloc[-1]
    recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
    recent_return_up = 1 if hist['ret2'].iloc[-1] > 0 else 0
   

    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret2_up': recent_return_up}, index=[0])
    

    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles2.loc[tau, 'predicted_ret2'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles2)

#%%
'''
For h = 3 
Quantile Predictions for the log returns ret1
with further explanatory variables 
X => 1, |Rt|, price_up, ret3_up
'''
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles3 = pd.DataFrame(index=quantiles, columns=['predicted_ret3'])

# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+1, X = (1, |Rt|, price_up, return_up)
    Y = hist['ret2'].shift(-1).iloc[:-1]  # Renditen für t+1
    X = sm.add_constant(hist[['Rt', 'price_up', 'ret3_up']].abs().iloc[:-1])  # (1, |Rt|, price_up, return_up)

    # Quantilregression
    quant_reg = sm.QuantReg(Y, X).fit(q=tau)

    # Vorhersagen für h=1
    recent_Rt = hist['Rt'].iloc[-1]
    recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
    recent_return_up = 1 if hist['ret3'].iloc[-1] > 0 else 0

    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret3_up': recent_return_up}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles3.loc[tau, 'predicted_ret3'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles3)

#%%
'''
For h = 4 
Quantile Predictions for the log returns ret4
with further explanatory variables 
X => 1, |Rt|, price_up, ret4_up
'''
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles4 = pd.DataFrame(index=quantiles, columns=['predicted_ret4'])

# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+1, X = (1, |Rt|, price_up, return_up)
    Y = hist['ret2'].shift(-1).iloc[:-1]  # Renditen für t+1
    X = sm.add_constant(hist[['Rt', 'price_up', 'ret4_up']].abs().iloc[:-1])  # (1, |Rt|, price_up, return_up)

    # Quantilregression
    quant_reg = sm.QuantReg(Y, X).fit(q=tau)

    # Vorhersagen für h=1
    recent_Rt = hist['Rt'].iloc[-1]
    recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
    recent_return_up = 1 if hist['ret4'].iloc[-1] > 0 else 0
  

    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret4_up': recent_return_up}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles4.loc[tau, 'predicted_ret4'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles4)
#%%
'''
For h = 5 
Quantile Predictions for the log returns ret4
with further explanatory variables 
X => 1, |Rt|, price_up, ret4_up
'''
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles5 = pd.DataFrame(index=quantiles, columns=['predicted_ret5'])

# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+1, X = (1, |Rt|, price_up, return_up)
    Y = hist['ret2'].shift(-1).iloc[:-1]  # Renditen für t+1
    X = sm.add_constant(hist[['Rt', 'price_up', 'ret5_up']].abs().iloc[:-1])  # (1, |Rt|, price_up, return_up)

    # Quantilregression
    quant_reg = sm.QuantReg(Y, X).fit(q=tau)

    # Vorhersagen für h=1
    recent_Rt = hist['Rt'].iloc[-1]
    recent_price_up = 1 if recent_Rt > hist['Close'].iloc[-2] else 0
    recent_return_up = 1 if hist['ret4'].iloc[-1] > 0 else 0


    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt), 'price_up': recent_price_up, 'ret5_up': recent_return_up}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    pred_quantiles5.loc[tau, 'predicted_ret5'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles5)

#%%
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
forecasts_sub_quant_more = forecasts.T
# %%
forecasts_sub_quant_more
# %%
from datetime import datetime
#%%
forecastdate = datetime(2023, 12, 20, 00, 00)
# %%

df_sub_dax_MoreQuant = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "DAX",
    "horizon": [str(i) + " day" for i in (1,2,5,6,7)],
    "q0.025": forecasts_sub_quant_more[:,0],
    "q0.25": forecasts_sub_quant_more[:,1],
    "q0.5": forecasts_sub_quant_more[:,2],
    "q0.75": forecasts_sub_quant_more[:,3],
    "q0.975": forecasts_sub_quant_more[:,4]})
df_sub_dax_MoreQuant


#%%
x = np.arange(5)+1
_ = plt.plot(x,forecasts_sub, ls="", marker="o", c="black")
_ = plt.xticks(x, x)
_ = plt.plot((x,x),(forecasts_sub[:,0], forecasts_sub[:,-1]),c='black')

#%%

'''
=================================================
------
Quantile Regression with a ROLLING WINDOW ANSATZ
------
=================================================
'''
#%%
'''
------
Model 1

=> Rolling Window Ansatz wird verwendet um über die Trainingsdaten zu iterieren
-----
'''

#%%
total_samples = len(hist)
train_size = int(total_samples * 0.7)  # 70% für das Training
validation_size = int(total_samples * 0.15)  # 15% für die Validierung
test_size = total_samples - train_size - validation_size  # 15% für den Test

# Aufteilung der Daten
train_data = hist.iloc[:train_size]
validation_data = hist.iloc[train_size:train_size + validation_size]
test_data = hist.iloc[train_size + validation_size:]

window_size = 50
#window_size = 10

for i in range(len(train_data) - window_size):
    # Daten für das aktuelle Fenster
    window_data = hist.iloc[i:i + window_size]

    # Schleife über Quantile
    for tau in quantiles:
        # Y = Rt+1:t+1, X = (1, |Rt|, price_up, return_up)
        Y = window_data['ret1'].shift(-1).iloc[:-1]  
        X = sm.add_constant(window_data[['Rt']].abs().iloc[:-1])

        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=1
        recent_Rt = window_data['Rt'].iloc[-1]
      

        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles1_rw_ansatz.loc[tau, 'predicted_ret1'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles1_rw_ansatz)




#%%
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles1_rw_ansatz = pd.DataFrame(index=quantiles, columns=['predicted_ret1'])

# Rolling Window-Größe
#window_size = 50
window_size = 10

# Schleife über den Rolling Window
for i in range(len(hist) - window_size):
    # Daten für das aktuelle Fenster
    window_data = hist.iloc[i:i + window_size]

    # Schleife über Quantile
    for tau in quantiles:
        # Y = Rt+1:t+1, X = (1, |Rt|, price_up, return_up)
        Y = window_data['ret1'].shift(-1).iloc[:-1]  
        X = sm.add_constant(window_data[['Rt']].abs().iloc[:-1])

        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=1
        recent_Rt = window_data['Rt'].iloc[-1]
      

        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles1_rw_ansatz.loc[tau, 'predicted_ret1'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles1_rw_ansatz)

#%%

'''
------
Model 2
=> Rolling Window Ansatz wird verwendet um über die Trainingsdaten zu iterieren
-----
'''

#%%
total_samples = len(hist)
train_size = int(total_samples * 0.7)  # 70% für das Training
validation_size = int(total_samples * 0.15)  # 15% für die Validierung
test_size = total_samples - train_size - validation_size  # 15% für den Test

# Aufteilung der Daten
train_data = hist.iloc[:train_size]
validation_data = hist.iloc[train_size:train_size + validation_size]
test_data = hist.iloc[train_size + validation_size:]

window_size = 50
#window_size = 10

for i in range(len(train_data) - window_size):
    # Daten für das aktuelle Fenster
    window_data = hist.iloc[i:i + window_size]

    # Schleife über Quantile
    for tau in quantiles:
        # Y = Rt+1:t+1, X = (1, |Rt|, price_up, return_up)
        Y = window_data['ret2'].shift(-1).iloc[:-1]  
        X = sm.add_constant(window_data[['Rt']].abs().iloc[:-1])

        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=1
        recent_Rt = window_data['Rt'].iloc[-1]
      

        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles2_rw_ansatz.loc[tau, 'predicted_ret2'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles2_rw_ansatz)








#%%
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles2_rw_ansatz = pd.DataFrame(index=quantiles, columns=['predicted_ret2'])

# Rolling Window-Größe
#window_size = 50
window_size = 10


# Schleife über den Rolling Window
for i in range(len(hist) - window_size):
    # Daten für das aktuelle Fenster
    window_data = hist.iloc[i:i + window_size]

    # Schleife über Quantile
    for tau in quantiles:
        # Y = Rt+1:t+1, X = (1, |Rt|, price_up, return_up)
        Y = window_data['ret2'].shift(-1).iloc[:-1]  
        X = sm.add_constant(window_data[['Rt']].abs().iloc[:-1])

        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=1
        recent_Rt = window_data['Rt'].iloc[-1]
      

        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles2_rw_ansatz.loc[tau, 'predicted_ret2'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles2_rw_ansatz)

# %%

#%%

'''
------
Model 3
=> Rolling Window Ansatz wird verwendet um über die Trainingsdaten zu iterieren
-----
'''
#%%
total_samples = len(hist)
train_size = int(total_samples * 0.7)  # 70% für das Training
validation_size = int(total_samples * 0.15)  # 15% für die Validierung
test_size = total_samples - train_size - validation_size  # 15% für den Test

# Aufteilung der Daten
train_data = hist.iloc[:train_size]
validation_data = hist.iloc[train_size:train_size + validation_size]
test_data = hist.iloc[train_size + validation_size:]

window_size = 50
#window_size = 10

for i in range(len(train_data) - window_size):
    # Daten für das aktuelle Fenster
    window_data = hist.iloc[i:i + window_size]

    # Schleife über Quantile
    for tau in quantiles:
       
        Y = window_data['ret3'].shift(-1).iloc[:-1]  
        X = sm.add_constant(window_data[['Rt']].abs().iloc[:-1])

        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=1
        recent_Rt = window_data['Rt'].iloc[-1]
      

        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles3_rw_ansatz.loc[tau, 'predicted_ret3'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles3_rw_ansatz)





#%%
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles3_rw_ansatz = pd.DataFrame(index=quantiles, columns=['predicted_ret3'])

# Rolling Window-Größe
#window_size = 50
window_size = 10

# Schleife über den Rolling Window
for i in range(len(hist) - window_size):
    # Daten für das aktuelle Fenster
    window_data = hist.iloc[i:i + window_size]

    # Schleife über Quantile
    for tau in quantiles:
        # Y = Rt+1:t+1, X = (1, |Rt|, price_up, return_up)
        Y = window_data['ret3'].shift(-1).iloc[:-1]  
        X = sm.add_constant(window_data[['Rt']].abs().iloc[:-1])

        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=1
        recent_Rt = window_data['Rt'].iloc[-1]
      

        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles3_rw_ansatz.loc[tau, 'predicted_ret3'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles3_rw_ansatz)

#%%

'''
------
Model 4
-----
'''
#%%
total_samples = len(hist)
train_size = int(total_samples * 0.7)  # 70% für das Training
validation_size = int(total_samples * 0.15)  # 15% für die Validierung
test_size = total_samples - train_size - validation_size  # 15% für den Test

# Aufteilung der Daten
train_data = hist.iloc[:train_size]
validation_data = hist.iloc[train_size:train_size + validation_size]
test_data = hist.iloc[train_size + validation_size:]

window_size = 50
#window_size = 10

for i in range(len(train_data) - window_size):
    # Daten für das aktuelle Fenster
    window_data = hist.iloc[i:i + window_size]

    # Schleife über Quantile
    for tau in quantiles:

        Y = window_data['ret4'].shift(-1).iloc[:-1]  
        X = sm.add_constant(window_data[['Rt']].abs().iloc[:-1])

        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=1
        recent_Rt = window_data['Rt'].iloc[-1]
      

        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles4_rw_ansatz.loc[tau, 'predicted_ret4'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles4_rw_ansatz)

#%%
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles4_rw_ansatz = pd.DataFrame(index=quantiles, columns=['predicted_ret4'])

# Rolling Window-Größe
#window_size = 50
window_size = 10

# Schleife über den Rolling Window
for i in range(len(hist) - window_size):
    # Daten für das aktuelle Fenster
    window_data = hist.iloc[i:i + window_size]

    # Schleife über Quantile
    for tau in quantiles:
        # Y = Rt+1:t+1, X = (1, |Rt|, price_up, return_up)
        Y = window_data['ret4'].shift(-1).iloc[:-1]  
        X = sm.add_constant(window_data[['Rt']].abs().iloc[:-1])

        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=1
        recent_Rt = window_data['Rt'].iloc[-1]
      

        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles4_rw_ansatz.loc[tau, 'predicted_ret4'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles4_rw_ansatz)


# %%

#%%

'''
------
Model 5
=> Mit dem Trainingsdatensatz
-----
'''
#%%
total_samples = len(hist)
train_size = int(total_samples * 0.7)  # 70% für das Training
validation_size = int(total_samples * 0.15)  # 15% für die Validierung
test_size = total_samples - train_size - validation_size  # 15% für den Test

# Aufteilung der Daten
train_data = hist.iloc[:train_size]
validation_data = hist.iloc[train_size:train_size + validation_size]
test_data = hist.iloc[train_size + validation_size:]

#window_size = 50
window_size = 10

for i in range(len(train_data) - window_size):
    # Daten für das aktuelle Fenster
    window_data = hist.iloc[i:i + window_size]

    # Schleife über Quantile
    for tau in quantiles:
       
        Y = window_data['ret5'].shift(-1).iloc[:-1]  
        X = sm.add_constant(window_data[['Rt']].abs().iloc[:-1])

        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=1
        recent_Rt = window_data['Rt'].iloc[-1]
      

        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles5_rw_ansatz.loc[tau, 'predicted_ret5'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles5_rw_ansatz)

#%%


#%%
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# DataFrame zur Speicherung der Vorhersagen
pred_quantiles5_rw_ansatz = pd.DataFrame(index=quantiles, columns=['predicted_ret5'])

# Rolling Window-Größe
window_size = 50
#window_size = 10

# Schleife über den Rolling Window
for i in range(len(hist) - window_size):
    # Daten für das aktuelle Fenster
    window_data = hist.iloc[i:i + window_size]

    # Schleife über Quantile
    for tau in quantiles:
        # Y = Rt+1:t+1, X = (1, |Rt|, price_up, return_up)
        Y = window_data['ret5'].shift(-1).iloc[:-1]  
        X = sm.add_constant(window_data[['Rt']].abs().iloc[:-1])

        # Quantilregression
        quant_reg = sm.QuantReg(Y, X).fit(q=tau)

        # Vorhersagen für h=1
        recent_Rt = window_data['Rt'].iloc[-1]
      

        new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(recent_Rt)}, index=[0])
        prediction = quant_reg.predict(new_data)

        # Speichern der Vorhersage im DataFrame
        pred_quantiles5_rw_ansatz.loc[tau, 'predicted_ret5'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(pred_quantiles5_rw_ansatz)


#%%

p_wa_1 = pred_quantiles1_rw_ansatz.values
p_wa_2 = pred_quantiles2_rw_ansatz.values
p_wa_3 = pred_quantiles3_rw_ansatz.values
p_wa_4 = pred_quantiles4_rw_ansatz.values
p_wa_5 = pred_quantiles5_rw_ansatz.values
# %%
p_wa_1 = p_wa_1.reshape(-1,1)
p_wa_2 = p_wa_2.reshape(-1,1)
p_wa_3 = p_wa_3.reshape(-1,1)
p_wa_4 = p_wa_4.reshape(-1,1)
p_wa_5 = p_wa_5.reshape(-1,1)

#%%

f_rolling_window = np.column_stack((p_wa_1, p_wa_2, p_wa_3, p_wa_4, p_wa_5))
#%%
f_rolling_window = f_rolling_window.T
# %%
f_rolling_window
# %%
from datetime import datetime
#%%
forecastdate = datetime(2023, 12, 20, 00, 00)
# %%

df_sub_quant_rolling_window = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "DAX",
    "horizon": [str(i) + " day" for i in (1,2,5,6,7)],
    "q0.025": f_rolling_window[:,0],
    "q0.25": f_rolling_window[:,1],
    "q0.5": f_rolling_window[:,2],
    "q0.75": f_rolling_window[:,3],
    "q0.975": f_rolling_window[:,4]})
df_sub_quant_rolling_window



























#%%
'''
------------------
Forecast evaluation
---------------------
Quantile Score
q_hat = the forecast of the tau quantile
y = the realization
tau = thefive different quantiles =>[0.025, 0.25, 0.5, 0.75, 0.975]
for tau = 0.5 we have the absolute error
!!! Smaller Scores are better
'''
# %%

'''Loading the new Data so that we can get the true y and compare it with the predicted quantiles '''
#%%
import yfinance as yf
dax_ticker = yf.Ticker("^GDAXI")
dax_data = dax_ticker.history(period ="max")

#%%
dax_data.set_index('Date', inplace = True)
#%%
dax_data.index.name = 'Date'
#%%
dax_data.index = pd.to_datetime(dax_data.index)
#%%
dax_data.index = dax_data.index.date
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
'''
Funktion die für die Forecast Evaluation
=> Gibt die wahren y basierend auf den horizon h und dem forecast_date heraus
'''
forecast_date = '2023-12-14'

# %%
forecast_date = pd.to_datetime(forecast_date)

# Extrahiere die Daten für die nächsten 5 Tage ab dem 'forecast_date'
next_5_days = dax_data.loc[forecast_date:forecast_date + pd.DateOffset(days=6)]
# %%
print(next_5_days)
# %%
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
true_return_1 = next_5_days['ret1'].iloc[0]
#%%
true_return_2 = next_5_days['ret2'].iloc[1]
true_return_2 
# %%
true_return_3 = next_5_days['ret3'].iloc[2]
# %%
true_return_4 = next_5_days['ret4'].iloc[3]
# %%
true_return_5 = next_5_days['ret4'].iloc[4]
#%%

true_return_1 = test_data['ret1']
true_return_1 


#%%
#quant1_hat = df_sub_dax['q0.025']
#quant2_hat = df_sub_dax['q0.25']
#quant3_hat = df_sub_dax['q0.5']
#quant4_hat = df_sub_dax['q0.75']
#quant5_hat = df_sub_dax['q0.975']
#%%
#quant1_hat = df_sub_dax_MoreQuant['q0.025']
#quant2_hat = df_sub_dax_MoreQuant['q0.25']
#quant3_hat = df_sub_dax_MoreQuant['q0.5']
#quant4_hat = df_sub_dax_MoreQuant['q0.75']
#quant5_hat = df_sub_dax_MoreQuant['q0.975']

#%%
quant1_hat = df_sub_quant_rolling_window['q0.025']
quant2_hat = df_sub_quant_rolling_window['q0.25']
quant3_hat = df_sub_quant_rolling_window['q0.5']
quant4_hat = df_sub_quant_rolling_window['q0.75']
quant5_hat = df_sub_quant_rolling_window['q0.975']


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

# %%
'''
Den Quantile Loss für das Baseline Model mit t_last = 500 berechnen
'''


''' Dataset der Vorhersage vom Baselinemodel importieren'''
from BaselineModel_DAX import erstelle_dataset
baseline_dataset = erstelle_dataset()
print(baseline_dataset)

#%%
quant1_hat_baseline = baseline_dataset['q0.025']
quant2_hat_baseline = baseline_dataset['q0.25']
quant3_hat_baseline = baseline_dataset['q0.5']
quant4_hat_baseline = baseline_dataset['q0.75']
quant5_hat_baseline = baseline_dataset['q0.975']

# %%
tau_values = np.array([0.025, 0.25, 0.5, 0.75, 0.975])

vectorized_quantile_score_ret1 = np.vectorize(quantile_score, otypes=[float])
scores_ret1_baseline = vectorized_quantile_score_ret1(quant1_hat_baseline, true_return_1, tau_values)

vectorized_quantile_score_ret2 = np.vectorize(quantile_score, otypes=[float])
scores_ret2_baseline = vectorized_quantile_score_ret2(quant2_hat_baseline, true_return_2, tau_values)
                                             
vectorized_quantile_score_ret3 = np.vectorize(quantile_score, otypes=[float])
scores_ret3_baseline = vectorized_quantile_score_ret3(quant3_hat_baseline, true_return_3, tau_values)

vectorized_quantile_score_ret4 = np.vectorize(quantile_score, otypes=[float])
scores_ret4_baseline = vectorized_quantile_score_ret4(quant4_hat_baseline, true_return_4, tau_values)
                                             
vectorized_quantile_score_ret5 = np.vectorize(quantile_score, otypes=[float])
scores_ret5_baseline = vectorized_quantile_score_ret5(quant5_hat_baseline, true_return_5, tau_values)

# Ergebnisse anzeigen
print('Quantile Scores Baseline Model for y = ret1 and the 5 quantiles')
print(scores_ret1_baseline)
print('Quantile Scores Baseline Model for y = ret2 and the 5 quantiles')
print(scores_ret2_baseline)
print('Quantile Scores Baseline Model for y = ret3 and the 5 quantiles')
print(scores_ret3_baseline)
print('Quantile Scores Baseline Model for y = ret4 and the 5 quantiles')
print(scores_ret4_baseline)
print('Quantile Scores Baseline Model for y = ret5 and the 5 quantiles')
print(scores_ret5_baseline)

# %%

'''
---------------------
Coverage Probability
=> 90% and 50%
--------------------
'''
#%%
'''
Coverage Probability for the Quantile Regression with the Rolling Window Ansatz and window size = 10
'''

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



#%%
        
'''
---------
Sharpness 
---------
=> Differenz zwischen den oberen und unteren Quantilen für jedes Return
'''

sharpness_values = {}

for i, actual_return in enumerate(actual_returns, start=1):
    # Benennung der Spalten für untere und obere Quantile
    lower_quantile_col = f'q{0.025}'
    upper_quantile_col = f'q{0.975}'

    # Sharpness für [0.025, 0.975]
    sharpness_values[f'sharpness_return_{i}_0.025_0.975'] = quantile_hats[upper_quantile_col][i-1] - quantile_hats[lower_quantile_col][i-1]

    # Benennung der Spalten für Quantile [0.25, 0.75]
    lower_quantile_col = f'q{0.25}'
    upper_quantile_col = f'q{0.75}'

    # Sharpness für [0.25, 0.75]
    sharpness_values[f'sharpness_return_{i}_0.25_0.75'] = quantile_hats[upper_quantile_col][i-1] - quantile_hats[lower_quantile_col][i-1]

# Ausgabe der Ergebnisse
for key, sharpness in sharpness_values.items():
    print(f'Schärfe für {key}: {sharpness:.4f}')



#%%
#%%
#coverage_probabilities = {}

#for q in quantiles:
    # Benennung der Spalten für untere und obere Grenzen
#    lower_bound_col = f'q{q}'
#    upper_bound_col = f'q{1.0 - q}'  # Oberstes Quantil ist 1.0 minus das unterste Quantil

    # Überprüfen Sie, ob die untere Grenze kleiner als die obere Grenze ist
#    if q < 0.5:
        # Berechnen Sie die Coverage Probability für jedes Quantil
#        is_covered = (actual_returns >= quantile_hats[lower_bound_col]) & (actual_returns <= quantile_hats[upper_bound_col])
#        coverage_probability = is_covered.mean()

        # Speichern Sie die Coverage Probability für das aktuelle Quantil
#        coverage_probabilities[f'quantile_{q}'] = coverage_probability

#for q, coverage in coverage_probabilities.items():
#    print(f'Coverage Probability für lower Quantil [{q}] und upper Quantile [1 - {q}]: {coverage:.2%}')

#%%


























# %%
'''
For h = 1 
Quantile Predictions for the log returns ret1
Trained on the training data set 90%
Predictions are made using the test data 10%
'''
from sklearn.model_selection import train_test_split
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]


p1 = pd.DataFrame(index=quantiles, columns=['predicted_ret1'])

train_data, test_data = train_test_split(hist, test_size=0.1, random_state=42, shuffle = False)
# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y_train = train_data['Rt'].shift(-1).iloc[:-1]  # Rt+1:t+1
    X_train = sm.add_constant(train_data['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    quant_reg = sm.QuantReg(Y_train, X_train).fit(q=tau)

    test_Rt = test_data['Rt'].iloc[-1]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(test_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    p1.loc[tau, 'predicted_ret1'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(p1)
# %%
'''
For h = 2 
Quantile Predictions for the log returns ret2
Trained on the training data set 90%
Predictions are made using the test data 10%1
'''
from sklearn.model_selection import train_test_split
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

p2 = pd.DataFrame(index=quantiles, columns=['predicted_ret2'])

train_data, test_data = train_test_split(hist, test_size=0.1, random_state=42, shuffle = False)
# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y_train = train_data['ret2'].shift(-1).iloc[:-1]  # Rt+1:t+1
    X_train = sm.add_constant(train_data['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    quant_reg = sm.QuantReg(Y_train, X_train).fit(q=tau)

    test_Rt = test_data['Rt'].iloc[-1]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(test_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    p2.loc[tau, 'predicted_ret2'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(p2)
# %%

'''
For h = 3 
Quantile Predictions for the log returns ret3
Trained on the training data set 90%
Predictions are made using the test data 10%1
'''
from sklearn.model_selection import train_test_split
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

p3 = pd.DataFrame(index=quantiles, columns=['predicted_ret3'])

train_data, test_data = train_test_split(hist, test_size=0.1, random_state=42, shuffle = False)
# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y_train = train_data['ret3'].shift(-1).iloc[:-1]  # Rt+1:t+1
    X_train = sm.add_constant(train_data['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    quant_reg = sm.QuantReg(Y_train, X_train).fit(q=tau)

    test_Rt = test_data['Rt'].iloc[-1]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(test_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    p3.loc[tau, 'predicted_ret3'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(p3)
# %%
'''
For h = 4 
Quantile Predictions for the log returns ret4
Trained on the training data set 90%
Predictions are made using the test data 10%
'''
from sklearn.model_selection import train_test_split
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

p4 = pd.DataFrame(index=quantiles, columns=['predicted_ret4'])

train_data, test_data = train_test_split(hist, test_size=0.1, random_state=42, shuffle = False)
# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y_train = train_data['ret4'].shift(-1).iloc[:-1]  # Rt+1:t+1
    X_train = sm.add_constant(train_data['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    quant_reg = sm.QuantReg(Y_train, X_train).fit(q=tau)

    test_Rt = test_data['Rt'].iloc[-1]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(test_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    p4.loc[tau, 'predicted_ret4'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(p4)
# %%
'''
For h = 5 
Quantile Predictions for the log returns ret5
Trained on the training data set 90%
Predictions are made using the test data 10%
'''
from sklearn.model_selection import train_test_split
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

p5 = pd.DataFrame(index=quantiles, columns=['predicted_ret5'])

train_data, test_data = train_test_split(hist, test_size=0.1, random_state=42, shuffle = False)
# Schleife über Quantile
for tau in quantiles:
    # Y = Rt+1:t+h, X = (1, |Rt|)
    Y_train = train_data['ret5'].shift(-1).iloc[:-1]  # Rt+1:t+1
    X_train = sm.add_constant(train_data['Rt'].abs().iloc[:-1])  # (1, |Rt|)

    quant_reg = sm.QuantReg(Y_train, X_train).fit(q=tau)

    test_Rt = test_data['Rt'].iloc[-1]
    new_data = pd.DataFrame({'const': 1, 'abs_Rt': np.abs(test_Rt)}, index=[0])
    prediction = quant_reg.predict(new_data)

    # Speichern der Vorhersage im DataFrame
    p5.loc[tau, 'predicted_ret5'] = prediction.iloc[0]

# Ausgabe der Vorhersagen
print("Vorhersagen für verschiedene Quantile:")
print(p5)
# %%

p1 = p1.values
p2 = p2.values
p3 = p3.values
p4 = p4.values
p5 = p5.values
# %%
p1 = p1.reshape(-1,1)
p2 = p2.reshape(-1,1)
p3 = p3.reshape(-1,1)
p4 = p4.reshape(-1,1)
p5 = p5.reshape(-1,1)

#%%

f_quant = np.column_stack((p1, p2, p3, p4, p5))
#%%
f_quant = f_quant.T
# %%
f_quant
# %%
from datetime import datetime
#%%
forecastdate = datetime(2023, 12, 20, 00, 00)
# %%

df_quant = pd.DataFrame({
    "forecast_date": forecastdate.strftime("%Y-%m-%d"),
    "target": "DAX",
    "horizon": [str(i) + " day" for i in (1,2,5,6,7)],
    "q0.025": f_quant[:,0],
    "q0.25": f_quant[:,1],
    "q0.5": f_quant[:,2],
    "q0.75": f_quant[:,3],
    "q0.975": f_quant[:,4]})
df_quant
# %%
def quantile_score(q_hat, y, tau):
    if q_hat > y:
        return 2 * (1 - tau) * (q_hat - y)
    elif y >= q_hat:
        return 2 * tau * (y - q_hat)
    else:
        # Fall, in dem q_hat gleich y ist (Randfall)
        return 0
# %%
