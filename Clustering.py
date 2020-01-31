
"""
Clustering Fig 1 + SFig3
2018/2019
Author:   
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
"""

print(__doc__)


import numpy as np
from scipy.stats import scoreatpercentile
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib import pylab as plt


############
### DATA ###
############

df = pd.read_excel('ABIDE_full_GSIQ.xlsx', index_col=[0]).reset_index(drop=True)
df_raw = df.drop(['Sex', 'Age', 'FIQ'], axis=1)


# put the data columns into a coherent order 
# ADI-R soc., ADI-R comm., ADI-R repet. behav., ADOS soc., ADOS comm., ADOS repet. behav., 
cols = df_raw.columns.tolist()
cols = cols[:-3] + cols[-2:-1] + cols[-3:-2] + cols[-1:]
df_raw = df_raw[cols] 

X = df_raw.values
X_colnames = ["ADI-R Soc.", "ADI-R Comm.", "ADI-R Repet. Behav.", "ADOS Soc.",
                "ADOS Comm.", "ADOS Repet. Behav."]
# Standardize the data
ss_X = StandardScaler()
X_scaled = ss_X.fit_transform(X)

df_scaled = pd.DataFrame(X_scaled, columns=X_colnames)

########################
### START THE ANALYSIS #
########################

########## R code to apply the package NbClust #######
"""
R code:

# instal package (Cran package: Munster)
install.packages("NbClust") 
"NbClust" %in% rownames(installed.packages())
require("NbClust")

# copy the dataset from Excel sheet
my.data <- read.table(pipe("pbpaste"), sep = "\t", header=TRUE)
set.seed(42)

# check the data
head(my.data)

# standardize the data
my.data <- scale(my.data)

# check again
head(my.data)

# Apply the nbclust package
NbClust(my.data, min.nc = 2, max.nc = 5, method="kmeans")
""" 
######### OUTPUT R #####################

# * Among all indices:                                                
# * 6 proposed 2 as the best number of clusters 
# * 15 proposed 3 as the best number of clusters 
# * 1 proposed 4 as the best number of clusters 
# * 1 proposed 5 as the best number of clusters 
# 
#                    ***** Conclusion *****                            
# 
# * According to the majority rule, the best number of clusters is  3 
#########################################

#####################
#### plot Fig. 1 ####
#####################

# Plotting function for the x axis to be centered
def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='left'):
    axes = []
    if which in ['x', 'both']:
        axes.append(ax.xaxis)
    elif which in ['y', 'both']:
        axes.append(ax.yaxis)
    for axis in axes:
        for t in axis.get_ticklabels():
            t.set_horizontalalignment(ha)
            t.set_rotation(rotation)
            t.set_rotation_mode(rotation_mode)

# run the kmeans with K=3 and plot it
n_clust = 3
clust = KMeans(n_clusters=n_clust, random_state=42)
X_cl = clust.fit_transform(X_scaled)
X_cl_labels = clust.labels_
sns.set(style="white", context="talk")

# Set up the matplotlib figure
f, axarr = plt.subplots(n_clust, 1, figsize=(12, 8), sharex=True)

# Take care of the colors
my_palette = ['#e74c3c'] * 3 + ['#3498db'] * 4

sns.set_palette(my_palette)
for i_cl in range(n_clust):
    cl_mean = np.mean(X_scaled[X_cl_labels == i_cl], axis=0) + 1
    n_subs = np.sum(X_cl_labels == i_cl)
    ax = sns.barplot(X_colnames, cl_mean, palette=my_palette, ax=axarr[i_cl])
    if i_cl == 0:
        ax.xaxis.set_ticks_position('top')
        rotateTickLabels(ax, 45, 'x')
        ax.xaxis.set_ticklabels(X_colnames)
    elif i_cl == (n_clust - 1):
        ax.tick_params(axis='x',labelbottom='off')
    ax.set_xlabel('%i patients' % n_subs)
    ax.set_ylabel("Group %i" %(i_cl+1))
    ax.set_ylim([0, 2.1])

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout(h_pad=3)
plt.savefig('kmeans_%icl.png' % n_clust)
plt.show()

##############################
#### Get data for STable3 ####
##############################

# Explore results of 3 clusters:
X_cl_labels = clust.labels_
df["Clust_numb"] = X_cl_labels


# Create a dataframe for each cluster
df_c1 = df[df["Clust_numb"] == 0]
df_c2 = df[df["Clust_numb"] == 1]
df_c3 = df[df["Clust_numb"] == 2]

# Describe each
df_c1.describe()
df_c2.describe()
df_c3.describe()

# check gender
np.unique(df_c1.Sex, return_counts=True) #65males - 10females
np.unique(df_c2.Sex, return_counts=True) #87 males - 10 females
np.unique(df_c3.Sex, return_counts=True) #81 males, 13 females

# Check age
df_c1.Age[df_c1.Age<18] = 0.
df_c1.Age[df_c1.Age>=18] = 1.
np.unique(df_c1.Age, return_counts=True) #55 ados - 20 adults

df_c2.Age[df_c2.Age<18] = 0.
df_c2.Age[df_c2.Age>=18] = 1.
np.unique(df_c2.Age, return_counts=True) #79 ados - 18 adults

df_c3.Age[df_c3.Age<18] = 0.
df_c3.Age[df_c3.Age>=18] = 1.
np.unique(df_c3.Age, return_counts=True) #73 ados - 21 adults



##############################
#### Plot the SFig. 4 ####
##############################

# use the standardized data
df_scaled ["Clust_numb"] = X_cl_labels

# Create a dataframe for each cluster
df_c1 = df_scaled [df_scaled ["Clust_numb"] == 0]
df_c2 = df_scaled [df_scaled ["Clust_numb"] == 1]
df_c3 = df_scaled [df_scaled ["Clust_numb"] == 2]

# plot it
color = ['#e74c3c'] * 3 + ['#3498db'] * 3
for ind, domain in enumerate(X_colnames):
    plt.figure(figsize=(8,6))
    for loop in range(0, 3):
        if loop == 0:
            plt.plot(df_c1.index, df_c1[domain].values, "o", color='g', label="Group 1")
            plt.axhline(y=df_c1.mean()[domain], xmin=0, xmax=7, linewidth=2, color = 'g', alpha=0.3)
        if loop == 1:
            plt.plot(df_c2.index, df_c2[domain].values, "o", color='y', label="Group 2")
            plt.axhline(y=df_c2.mean()[domain], xmin=0, xmax=7, linewidth=2, color = 'y', alpha=0.3)
        if loop == 2:
            plt.plot(df_c3.index, df_c3[domain].values, "o", color='b', label="Group 3")
            plt.axhline(y=df_c3.mean()[domain], xmin=0, xmax=7, linewidth=2, color = 'b', alpha=0.3)

    plt.legend(loc="lower left", fontsize=16)
    plt.ylim([-3, 3])
    plt.ylabel("SCORE", fontsize=16, weight="bold")
    plt.xlabel("PATIENTS", fontsize=16, weight="bold")
    plt.title(domain, fontsize=20, weight="bold", color=color[ind])    
    plt.tight_layout()
    # plt.savefig("scatter_3cl_{}".format(domain))
    plt.show()












