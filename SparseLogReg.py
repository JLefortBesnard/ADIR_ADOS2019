import numpy as np
from scipy.stats import scoreatpercentile
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pylab as plt
import matplotlib.patches as mpatches
from sklearn import linear_model
# from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_excel('ABIDE_full_GSIQ.xlsx')
df = df.reset_index(drop=True)

cols = df.columns.tolist()
cols = cols[:-6] + cols[4:5] + cols[3:4] + cols[5:]
df = df[cols]

# items as array
X = df.values[:, 0:6] #Adi-Ados
ss_X = StandardScaler()
X = ss_X.fit_transform(X)
X_colnames = ["ADI-R Soc.", "ADI-R Comm.", "ADI-R Repet. Behav.", "ADOS Soc.",
                "ADOS Comm.", "ADOS Repet. Behav."]
                                

# outcome classification
y = np.sum(df.values[:, 0:6], axis=1)
ss_y = StandardScaler()
y = ss_y.fit_transform(y.reshape(-1, 1))
y_bin = np.array(
    y >= scoreatpercentile(y, 50), dtype=np.int32)  # a classification problem !
y_bin = np.squeeze(y_bin)


########################
### START THE ANALYSIS #
########################

plt.close('all')
subplot_xlabel = '0 domain >> 6 domains'
n_verticals = 25
C_grid = np.logspace(-3.5, 1, n_verticals)


coef_list2 = []
acc_list2 = []
for i_step, my_C in enumerate(C_grid):
    sample_accs = []
    sample_coef = []
    for i_subsample in range(100):
        folder = StratifiedShuffleSplit(y_bin, 100, test_size=0.1,
                                        random_state=i_subsample)
        indexes = iter(folder)
        train_inds, test_inds = next(indexes)
        clf2 = LogisticRegression(penalty='l1', C=my_C, verbose=False)
        clf2.fit(X[train_inds, :], y_bin[train_inds])
        # acc = clf2.score(X[test_inds, :], y_bin[test_inds])
        pred_y = clf2.predict(X[test_inds, :])
        acc = (pred_y == y_bin[test_inds]).mean()
        sample_accs.append(acc)
        sample_coef.append(clf2.coef_[0, :])
        # sample_coef.append(clf2.coef_)

    coef_list2.append(np.mean(np.array(sample_coef), axis=0))
    acc_list2.append(np.mean(sample_accs))
    print("C: %.4f acc: %.2f"%(my_C, acc))
    
coef_list2 = np.squeeze(np.array(coef_list2))
acc_list2 = np.array(acc_list2)




#PLOTTING
# plot paths
n_cols = 3
n_rows = 1

my_palette = np.array([
    # '#4BBCF6',
    '#F47D7D', '#FBEF69', '#98E466', '#000000',
    '#A7794F', '#CCCCCC', '#85359C', '#FF9300', '#FF0030'
])
my_colors = np.array(['???????'] * coef_list2.shape[-1])
i_col = 0
new_grp_pts_x = []
new_grp_pts_y = []
new_grp_pts_col = []
new_grp_pts_total = []

for i_vertical, (params, acc, C) in enumerate(zip(
    coef_list2, acc_list2, C_grid)):
    b_notset = my_colors == '???????'
    b_nonzeros = params != 0
    b_coefs_of_new_grp = np.logical_and(b_notset, b_nonzeros)
    
    if np.sum(b_coefs_of_new_grp) > 0:
        # import pdb; pdb.set_trace()
        # we found a new subset that became 0
        for new_i in np.where(b_coefs_of_new_grp == True)[0]:
            # color all coefficients of the current group
            cur_col = my_palette[i_col]
            my_colors[new_i] = cur_col
            
        new_grp_pts_x.append(C)
        new_grp_pts_y.append(acc)
        new_grp_pts_col.append(cur_col)
        new_grp_pts_total.append(np.sum(b_nonzeros))
        i_col += 1
    

# plotting

f, axarr = plt.subplots(nrows=n_rows, ncols=n_cols,
    figsize=(15, 10), facecolor='white')
t, i_col = 0, 0

for i_line in range(X.shape[-1]):
    axarr[i_col].plot(np.log10(C_grid),
        coef_list2[:, i_line], label=X_colnames[i_line],
            color=my_colors[i_line], linewidth=1.5)

# axarr[0].set_xticks(np.arange(len(C_grid)))
# axarr[0].set_xticklabels(np.log10(C_grid))  #, rotation=75)
axarr[i_col].set_xlabel(subplot_xlabel, fontsize=16)
axarr[i_col].legend(loc='upper left', fontsize=14)
axarr[0].grid(True)
# axarr[i_col].set_ylabel('Item groups', fontsize=16)
axarr[0].set_title('Domain groups', fontsize=20)
axarr[0].set_xticks([])

# axarr[1].axis('off')
axarr[1].plot(np.log10(C_grid), acc_list2, color='#000000',
                 linewidth=1.5)
# axarr[1].set_title('ACCURACY')
axarr[1].set_ylim(0.4, 1.00)
axarr[1].grid(True)
# axarr[1].set_xticklabels(np.log10(C_grid), '')
axarr[1].set_xticks([])
axarr[1].set_xlabel(subplot_xlabel, fontsize=16)
# axarr[1].set_ylabel('Out-of-sample accuracy', fontsize=16)
axarr[1].set_title('Out-of-sample accuracy', fontsize=20)
# plt.rcParams.update({'font.size': 14})
plt.rc('ytick', labelsize=14)

for i_pt, (x, y, col, n_coefs) in enumerate(zip(
    new_grp_pts_x, new_grp_pts_y, new_grp_pts_col, new_grp_pts_total)):
    axarr[1].plot(np.log10(x), y,
                  marker='o', color=col, markersize=8.0)
    axarr[1].text(
        np.log10(x) - 1.6,
        y + 0.003,
        '%i domains' % n_coefs, fontsize=14)


sns.heatmap(coef_list2.T[::1], cbar=False, vmin=-1.5, vmax=1.5, cmap="coolwarm")
# plt.xticks(np.arange(len(C_grid)) + 0.75, np.log10(C_grid))
plt.yticks(np.arange(len(X_colnames)) + 0.5,
           X_colnames[::1], rotation=0, fontsize=14)
# axarr[2].set_ylabel('Item importance', fontsize=16)
axarr[2].set_title('Domain importance', fontsize=20)
axarr[2].set_xlabel(subplot_xlabel, fontsize=16)
axarr[2].set_xticks([])

# plt.suptitle('Predictiveness of ADOS and ADI-R domains on Autism severity',fontsize=16, y=1, weight="bold")

plt.tight_layout()
# plt.savefig('SparseLogReg.png', DPI=400, facecolor='white')

plt.show()



