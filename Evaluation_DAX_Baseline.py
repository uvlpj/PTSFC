#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

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
for i in range(5):
    all_dax["ret"+str(i+1)] = compute_return(all_dax["Close"].values, h=i+1)  

#%%
tau = [.025, .25, .5, .75, .975]
#%%
#start = '2024-01-04'
#end = '2024-01-10'
#next_5_days = hist[(hist['Date'] >= start) & (hist['Date'] <= end)]
#hist = hist[hist['Date'] <= '2024-01-03']
#%%
start = '2024-01-18'
end = '2024-01-24'
next_5_days = hist[(hist['Date'] >= start) & (hist['Date'] <= end)]
hist = hist[hist['Date'] <= '2024-01-17']
#%%
next_5_days
#%%
hist
#%%
hist = hist.dropna()
#%%
pred_baseline = np.zeros((5,5))
#%%
last_t = 500

for i in range(5):
    ret_str = "ret"+str(i+1)
    
    pred_baseline[i,:] = np.quantile(hist[ret_str].iloc[-last_t:], q=tau)

#%%
pred_baseline
#%%
from datetime import datetime
forecastdate = datetime(2023, 11, 22, 00, 00)
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

#%%
# Quantile Score

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
quant1_hat = df_baseline_sub['q0.025']
quant2_hat = df_baseline_sub['q0.25']
quant3_hat = df_baseline_sub['q0.5']
quant4_hat = df_baseline_sub['q0.75']
quant5_hat = df_baseline_sub['q0.975']

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
# %%
'''
Interval Score 
50%
95%
'''
interval_scores = {}

alpha = 0.5  # Alpha-Wert

for i, actual_return in enumerate(actual_returns, start=1):
    interval_scores[f'Return_{i}'] = {}

    # Benennung der Spalten für untere und obere Grenzen
    lower_bound_col = 'q0.25'
    upper_bound_col = 'q0.75'

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
