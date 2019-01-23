from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib import pylab as plt
import numpy as np
from sklearn import linear_model

######################################
# new figure following the comments ##
######################################


df = pd.read_excel('ABIDE_full_GSIQ.xlsx').reset_index(drop=True)
cols = df.columns.tolist()
cols = cols[:-6] + cols[4:5] + cols[3:4] + cols[5:]
df = df[cols] 

X = df.values[:, 0:6] #Adi-Ados
ss_X = StandardScaler()
X = ss_X.fit_transform(X)

X_colnames = ["ADI-R Soc.", "ADI-R Comm.", "ADI-R Repet. Behav.", "ADOS Soc.",
                "ADOS Comm.", "ADOS Repet. Behav."]
                
df_scaled = pd.DataFrame(X, columns=X_colnames)
df_scaled["Age"] = df["Age"]

##################
# first figure ###
##################
df_scaled["TOT ADI"] = (df_scaled["ADI-R Soc."] + df_scaled["ADI-R Comm."] + df_scaled["ADI-R Repet. Behav."])/3
df_scaled["TOT ADOS"] = (df_scaled["ADOS Soc."] + df_scaled["ADOS Comm."] + df_scaled["ADOS Repet. Behav."])/3
color = ['#e74c3c', '#3498db']

plt.figure(figsize=(8,6))
plt.plot(df_scaled.Age, df_scaled["TOT ADI"].values, "o", color='r', label="ADI-R tot")
# plt.axhline(y=df_scaled["TOT ADI"].mean(), xmin=-3, xmax=3, linewidth=2, color = '5', alpha=0.3)
plt.plot(df_scaled.Age, df_scaled["TOT ADOS"].values, "x", color='b', label="ADOS tot")
# plt.axhline(y=df_scaled["TOT ADOS"].mean(), xmin=-3, xmax=3, linewidth=2, color = 'b', alpha=0.3)
plt.legend(loc="upper right", fontsize=16)
plt.ylim([-3, 3])
plt.ylabel("SCORE", fontsize=16, weight="bold")
plt.xlabel("AGE", fontsize=16, weight="bold")
plt.tight_layout()
# plt.savefig('Total_vs_age_raw.png', DPI=400)
plt.show()


##################
# second figure ##
##################


color = ['#e74c3c', '#3498db', '#F47D7D', '#FBEF69', '#98E466', '#000000']

plt.figure(figsize=(8,6))
#adir
plt.plot(df_scaled.Age, df_scaled["ADI-R Soc."].values, "x", color='r', label="ADI-R Soc.")
plt.plot(df_scaled.Age, df_scaled["ADI-R Comm."].values, ">", color='r', label="ADI-R Comm.")
plt.plot(df_scaled.Age, df_scaled["ADI-R Repet. Behav."].values, "s", color='r', label="ADI-R Repet. Behav.")
#ados
plt.plot(df_scaled.Age, df_scaled["ADOS Soc."].values, "x", color='b', label="ADOS Soc.")
plt.plot(df_scaled.Age, df_scaled["ADOS Comm."].values, ">", color='b', label="ADOS Comm.")
plt.plot(df_scaled.Age, df_scaled["ADOS Repet. Behav."].values, "s", color='b', label="ADOS Repet. Behav.")

plt.legend(loc="upper right", fontsize=16)
plt.ylim([-3, 3])
plt.ylabel("SCORE", fontsize=16, weight="bold")
plt.xlabel("AGE", fontsize=16, weight="bold")
plt.tight_layout()
# plt.savefig('scores_vs_age_raw.png', DPI=400)
plt.show()
