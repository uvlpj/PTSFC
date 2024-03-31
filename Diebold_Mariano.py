#%%
import numpy as np
from scipy.stats import t
#%%
'''
Mean Quantile Score of the Baseline Model
'''
S_Baseline_mean = 2.33601
#%%
'''
Paired Quantile Scores of the Baseline Model
'''
S_Baseline = [1.64220501, #1
              2.2443719, #2
              2.818616, #3
              1.920074, #4
              1.3614541, #5
              1.30284193, #6
              2.86294640, #7
              1.268996, #8
              3.8969334, #9
              4.0417098, #10
              ]
#%%
S_Linear_mean = 2.4404669

S_Linear = [1.06193,
                 1.0511023,
                 1.04818756,
                 1.814841,
                 2.64492873,
                 3.7412849,
                 7.86210,
                 3.02236,
                 1.057701,
              1.1002136   ]
#%%
'''
Mean Quantile Score of the Linear Model 1
'''
S_Quantile_one_mean = 2.46981
#%%
'''
Paired Quantile Scores of the Linear Model 1
'''
S_Quantile_1 = [1.526103, #1
           1.321096, #2
           1.15199075, #3 
           1.63976968, #4
            2.63079280, #5
            3.8974543,  #6
            7.45756102, #7 
            2.920648, #8
            1.071341, #9
            1.0813517 ] #10
#%%
S_Quantile_2_mean = 1.904

S_Qauntile_2 = [0.99151352,
                0.9607880,
                1.39903509,
               1.385074,
                 1.69926826,
                2.471836,
                5.068084,
                  2.6972983,
                    1.3998289,
                      0.972918  ]

#%%
S_QRF_mean = 1.89169

S_QRF = [0.975799,
              1.01577,
              1.5776262,
              1.5021470,
              1.951655,
              2.248530,
              4.93168,
              2.3357616,
              1.7164220,
              0.661564]
#%%
#%%
N = 10
# %%

var = 0
for _ in range(N):
    for i in range(10):
        var += (S_QRF[i] - S_Linear[i]) ** 2

var = var/(N)
print("Die berechnete Summe beträgt:", var)

# %%
#t_statistic = (np.sqrt(N)*(S_Baseline_mean - S_Quantile_2_mean))/ np.sqrt(var)
#t_statistic = (np.sqrt(N)*(S_Quantile_one_mean - S_Baseline_mean) )/ np.sqrt(var)
#t_statistic = (np.sqrt(N)*(S_Linear_one_mean - S_Baseline_mean) )/ np.sqrt(var)
#t_statistic = (np.sqrt(N)*(S_QRF_mean - S_Baseline_mean) )/ np.sqrt(var)
t_statistic = (np.sqrt(N)*(S_QRF_mean - S_Linear_mean) )/ np.sqrt(var)

#t_statistic = (np.sqrt(N)*(S_Quantile_2_mean - S_Quantile_one_mean) )/ np.sqrt(var)


# Berechnung der Freiheitsgrade
degrees_of_freedom = N - 1

# Berechnung des p-Werts
p_value = 2 * (1 - t.cdf(np.abs(t_statistic), df=degrees_of_freedom))

# Ausgabe der Ergebnisse
print("Teststatistik:", t_statistic)
print("p-Wert:", p_value)

# %%
from scoring import interval_score
# %%
