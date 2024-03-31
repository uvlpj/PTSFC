#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm

#%%
msft = yf.Ticker("^GDAXI")
#%%
all_dax = msft.history(period="max")
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
# getting all eight data sets for the historic data  
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
tau = [.025, .25, .5, .75, .975]
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
hist
#%%
next_5_days
#%%
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

# %%

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
quant1_hat = df_sub_dax['q0.025']
quant2_hat = df_sub_dax['q0.25']
quant3_hat = df_sub_dax['q0.5']
quant4_hat = df_sub_dax['q0.75']
quant5_hat = df_sub_dax['q0.975']

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
# %%
'''
Coverage Probability
50% 
95% 
'''

actual_returns = [true_return_1, true_return_2, true_return_3, true_return_4, true_return_5]
quantile_hats = [quant1_hat, quant2_hat, quant3_hat, quant4_hat, quant5_hat]
quantile_hats = pd.DataFrame(quantile_hats)
quantile_hats = quantile_hats.T 

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

#%%
'''
Sharpness
50%
95%
'''
sharpness_0025_0975 = quantile_hats['q0.975'] - quantile_hats['q0.025']

# Schärfe für 0.25 und 0.75 berechnen
sharpness_025_075 = quantile_hats['q0.75'] - quantile_hats['q0.25']

print("Schärfe für 0.025 und 0.975:")
print(sharpness_0025_0975)
print("\nSchärfe für 0.25 und 0.75:")
print(sharpness_025_075)     

#%%
'''
Interval Score 
50%
95%
'''
interval_scores = {}

alpha = 0.05  # Alpha-Wert

for i, actual_return in enumerate(actual_returns, start=1):
    interval_scores[f'Return_{i}'] = {}

    # Benennung der Spalten für untere und obere Grenzen
    lower_bound_col = 'q0.025'
    upper_bound_col = 'q0.975'

    # Berechnung des Interval Scores für den aktuellen Rückgabewert
    lower = quantile_hats.loc[i - 1, lower_bound_col]
    upper = quantile_hats.loc[i - 1, upper_bound_col]
    interval_score = 0

    if (actual_return >= lower) and (actual_return <= upper):
        interval_score = 2 * alpha * np.abs(actual_return - (lower + upper) / 2)
    elif actual_return < lower:
        interval_score = 2 * (1 - alpha) * (lower - actual_return)
    elif actual_return > upper:
        interval_score = 2 * (1 - alpha) * (actual_return - upper)

    # Speichern des Interval Scores für den aktuellen Rückgabewert
    interval_scores[f'Return_{i}']['Interval_Score'] = interval_score

# Ausgabe der Ergebnisse
for return_key, return_scores in interval_scores.items():
    print(f'Interval Score {alpha}, {return_key}: {return_scores["Interval_Score"]:.4f}')
# %%
