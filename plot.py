from Sir import SIR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logit, expit
import seaborn as sns

# percentage_unobserved = 0.2557773028247493
# max_R0 = 2.874329120666961
# min_R0 = 2.0442062661933393
# reaction_days = 15.0
# gamma = 0.2303999807017334

x = ([ 0.2557773 ,  2.87432912,  2.04420627, 15.0,  0.23039998])
percentage_unobserved, max_R0, min_R0, reaction_days, gamma = x

def calculate_dynamic_R0(max_R0, min_R0, reaction_days, simulation_days=200):
    dynamic_R0 = expit(np.linspace(-5, 3, num=reaction_days))[::-1]
    dynamic_R0 = dynamic_R0 * (max_R0 - min_R0) + min_R0
    dynamic_R0 = np.concatenate((dynamic_R0, np.repeat(dynamic_R0[-1], simulation_days)))
    return dynamic_R0


sir = SIR(
    N=1304851, I0=1/percentage_unobserved, beta=calculate_dynamic_R0(max_R0, min_R0, int(reaction_days), 300)*gamma,
    gamma=gamma, days=150)
S, I, R = sir.run()
plt.figure(figsize=(8, 6))
plt.title("SIR model")
plt.plot(S, color='b', label='susceptible')
plt.plot(I, color='r', label='infected')
plt.plot(R, color='g', label='removed')
plt.legend()
plt.xlabel("Days from start of infection")
plt.ylabel("Cases")
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()
