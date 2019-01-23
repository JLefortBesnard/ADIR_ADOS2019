
"""
Nested logistic regression Fig 3-4-5 + SFig5-6-7 + data for STable4 
2018/2019
Author:   
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
"""



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import ShuffleSplit
import seaborn as sns
from matplotlib import pylab as plt
from scipy.stats import scoreatpercentile
from sklearn.metrics import r2_score
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import scipy
from sklearn.metrics import confusion_matrix
import itertools

np.random.seed(0)

# for rotating labels later on
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



        
#####################################################################
            ##########################################
            ################ Age as Y ################
            ##########################################
#####################################################################

# Set the raw data
df = pd.read_excel('ABIDE_full_GSIQ.xlsx').reset_index(drop=True)
df.Sex[df.Sex == 1] = "Men"
df.Sex[df.Sex == 2] = "Women"

cols = df.columns.tolist()
cols = cols[:-6] + cols[4:5] + cols[3:4] + cols[5:]
df = df[cols] 


X = df.values[:, 0:6] #Adi-Ados
ss_X = StandardScaler()
X = ss_X.fit_transform(X)
X_colnames = ["ADI-R Soc.", "ADI-R Comm.", "ADI-R Repet. Behav.", "ADOS Soc.",
                "ADOS Comm.", "ADOS Repet. Behav."]
df_scaled = pd.DataFrame(X, columns=X_colnames)
df_scaled["FIQ"] = df.FIQ
df_scaled["Sex"] = df.Sex
df_scaled["Age"] = df.Age

df = df_scaled


######################################
# graph 2: Y = Age, X = IQ + gender #  DEFINING DATASET
######################################

X_df1 = df[df.Sex == "Men"][df.FIQ > df.FIQ.median()].reset_index(drop=True)  # men low FIQ #110 subjects

X_df2 = df[df.Sex == "Men"][df.FIQ < df.FIQ.median()].reset_index(drop=True)  # men high FIQ #111 subjects

X_df3 = df[df.Sex == "Women"][df.FIQ > df.FIQ.median()].reset_index(drop=True)  # women low FIQ #17 subjects

X_df4 = df[df.Sex == "Women"][df.FIQ < df.FIQ.median()].reset_index(drop=True)  # women high FIQ # 16 subjects

X_all = [X_df1, X_df2, X_df3, X_df4]
n = [len(X_df1), len(X_df2), len(X_df3), len(X_df4)]


##################################
#### CLASSIFICATION ANALYSIS ##### UP OR DOWNSAMPLING + SUBSAMPLING-LOGREG() 100 TIMES 
##################################

# one for each df
acc_list_bs = [[], [], [], []]
coef_list_bs = [[], [], [], []]
acc_list_original = [[], [], [], []]
coef_list_original = [[], [], [], []]
predictions = {}

for ind, df in enumerate(X_all):
    print("n_df={}".format(df.shape))
    # divide into two category, adult and non ADULT
    df_ado = df[df.Age < 18]
    df_adult = df[df.Age >= 18]
    print("n_ado={}".format(df_ado.shape))
    print("n_adult={}".format(df_adult.shape))
    # transform into a binary variable
    df_ado.Age = 0
    df_adult.Age = 1

    # always more ados than adults thus oversampling the adult dataset
    if np.float(len(df_ado))/np.float(len(df_adult)) <= 3:  #1/3 is an arbitrary accepted maximum oversizing
        
        # UPSAMPLING
        sample_index_up = np.random.choice(df_adult.index, len(df_ado))
        df_adult_up = df_adult.loc[sample_index_up]
        
        n[ind] = len(df_ado)*2 #length of the majority group
        
        # get the weight form the original sample
        df_original = pd.concat([df_adult_up, df_ado], ignore_index=True)
        # prepare the arrays of data          
        X = df_original.values[:, 0:6] #Adi-Ados
        ss_X = StandardScaler()
        X = ss_X.fit_transform(X)            
        y = df_original.Age
        
        #run the logistic regression for the specific subsample 
        clf = LogisticRegression()
        clf.fit(X, y)
        acc = clf.score(X, y)
        y_pred = clf.predict(X)
        predictions["pred:real_{}".format(ind)] = [y_pred, y]

        acc_list_original[ind].append(acc)
        coef_list_original[ind].append(clf.coef_)
        print("acc: {}".format(acc))
        
        for i_subsample in range(100):
            # bootstrapping
            sample_index = np.random.choice(df_original.index, len(df_original)) # resample with replacement
            df_bs = df_original.loc[sample_index]
                    
            # prepare the arrays of data          
            X = df_bs.values[:, 0:6] #Adi-Ados
            ss_X = StandardScaler()
            X = ss_X.fit_transform(X)            
            y = df_bs.Age
            
            #run the logistic regression for the specific subsample 
            clf = LogisticRegression()
            clf.fit(X, y)
            acc = clf.score(X, y)

            acc_list_bs[ind].append(acc)
            coef_list_bs[ind].append(clf.coef_)
            # print "acc: %.2f" % (acc)
            
    else:
        if len(df_adult) >= 10:
            sample_index_down = np.random.choice(df_ado.index, len(df_adult))
            df_ado_down = df_ado.loc[sample_index_down]
            # size of the total df because of the downsampling
            n[ind] = len(df_adult)*2
            
            # get the weight from the analysis with the original sample
            df_original = pd.concat([df_adult, df_ado_down], ignore_index=True)
            # prepare the arrays of data          
            X = df_original.values[:, 0:6] #Adi-Ados
            ss_X = StandardScaler()
            X = ss_X.fit_transform(X)            
            y = df_original.Age
        
            #run the logistic regression for our original sample
            clf = LogisticRegression()
            clf.fit(X, y)
            acc = clf.score(X, y)
            y_pred = clf.predict(X)
            predictions["pred:real_{}".format(ind)] = [y_pred, y]

            acc_list_original[ind].append(acc)
            coef_list_original[ind].append(clf.coef_)
            print("acc: {}".format(acc))
            
            
            for i_subsample in range(100):
                # boostrapping
                sample_index = np.random.choice(df_original.index, len(df_original)) # resample with replacement
                df_bs = df_original.loc[sample_index]
                
                # prepare the arrays of data          
                X = df_bs.values[:, 0:6] #Adi-Ados
                ss_X = StandardScaler()
                X = ss_X.fit_transform(X)            
                y = df_bs.Age
                
                #run the logistic regression for the specific subsample 
                clf = LogisticRegression()
                clf.fit(X, y)
                acc = clf.score(X, y)

                acc_list_bs[ind].append(acc)
                coef_list_bs[ind].append(clf.coef_)
                # print "acc: %.2f" % (acc)

        else:
            print("NOT ENOUGH DATA FOR X_df{}".format(ind+1))
            print("n_Ado = {}, n_Adult = {}".format(len(df_ado), len(df_adult)))
            not_enough = [len(df_ado), len(df_adult)]
            
    

#  Create the df to plot the bootsrap confidence interval of the weights
all_df = []
for num_df in range(0, 4): # there isn't enough data for df_4
    if coef_list_bs[num_df] != []:
        all_df.append(pd.DataFrame(np.squeeze(np.array(coef_list_bs[num_df])),
                    columns = X_colnames))
    else:
        print("No data for df {}".format(num_df+1))
        all_df.append([])


bs_errorbar_per_df = {}
for num_df, df in enumerate(all_df):
    bs_err_per_items = {}
    for ind, items in enumerate(df):
        bs_err = scipy.stats.scoreatpercentile(df[items], [5, 95], interpolation_method='fraction', axis=None)
        bs_err_per_items[items] = bs_err
    bs_errorbar_per_df[num_df] = bs_err_per_items

# get rid of unnecessary dimension in the list of coef
for ind, i in enumerate(coef_list_original):
    coef_list_original[ind] = np.squeeze(i)

# calculating boostrap CI + and - values above and under the original data
err_ = []
for df in range(0, len(bs_errorbar_per_df)):
    err_l_ = []
    err_u_ = []
    if bs_errorbar_per_df[df] != {}:
        for ind, names in enumerate(X_colnames):
            err_l = coef_list_original[df][ind] - bs_errorbar_per_df[df][names][0]
            err_u = bs_errorbar_per_df[df][names][1] - coef_list_original[df][ind]
            err_l_.append(err_l)
            err_u_.append(err_u)
        err = [err_l_, err_u_]
        err_.append(err)
    else:
        err_.append([])


#################################################################
# Extract information about which point is important to analyse #
#################################################################
for i in range(0, 3):
    if i == 0:
        print("-----men high IQ------")
    if i == 1:
        print("-----men low IQ------")
    if i == 2:
        print("-----women high IQ------")
    # compute the average absolute weight for each domain
    ave = np.mean(abs(coef_list_original[i]))    
    for ind, j in enumerate(X_colnames):
        err_bars = bs_errorbar_per_df[i][j]
        value = coef_list_original[i][ind]
        # print("{}, value = {}, error bars = {}".format(j, value, err_bars))
        if abs(value) >= ave - (ave*20/100):
            if err_bars[0] > 0 and err_bars[1] > 0:
                print("Interpret: {}".format(j))        
            elif err_bars[0] < 0 and err_bars[1] < 0:
                print("Interpret: {}".format(j))
            else:
                # compute the percentage of above and below 0:
                tot = abs(err_bars[0]) + abs(err_bars[1])
                lower_bound_perc = (abs(err_bars[0]) * 100)/tot
                upper_bound_perc = (abs(err_bars[1]) * 100)/tot
                # print(lower_bound_perc)
                # print(upper_bound_perc)
                if lower_bound_perc <25 or upper_bound_perc<25:
                    print("Interpret: {}".format(j))     
        

                    
            
        




#####################
## PLOT ############# IQ + gender = age
#####################


X = np.array([1, 2 , 3, 4, 5, 6])

f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")
labels = [""] + [X_colnames[i] for i in range(0, len(X_colnames))] # labels starts at index 1
for i in range(1, 5):
    Y = np.squeeze(coef_list_original[i-1]) # the weights obtained with the original sample
    if i == 1: # men high iq
        err = err_[i-1]
        ax1 = plt.subplot(2, 2, 3)
        ax1.errorbar(X, Y, yerr = np.array(err), fmt='mo', ecolor="c", capthick=2)
        # plt.bar(X, Y, yerr=err)
        ax1.set_xticklabels(labels)
        rotateTickLabels(ax1, -55, 'x')
        acc = (acc_list_original[i-1][0] * 100).astype(int) 
        plt.text(4, 1.5, "Accuracy=%i%%, n=%i participants" %(acc, n[i-1]), fontsize=8, ha='center', style="italic")
        plt.xlabel("")
        plt.ylabel("MALE", fontsize=12, weight="bold")
        plt.axhline(0, color='grey')
        plt.grid(True, axis="y")

    elif i == 2: # men low iq
        err = err_[i-1]
        ax2 = plt.subplot(2, 2, 4)
        ax2.errorbar(X, Y, yerr = np.array(err), fmt='mo', ecolor="c", capthick=2)
        # plt.bar(X, Y, yerr=err)
        ax2.set_xticklabels(labels)
        rotateTickLabels(ax2, -55, 'x')
        acc = (acc_list_original[i-1][0] * 100).astype(int) 
        plt.text(4, 1.5, "Accuracy=%i%%, n=%i participants" %(acc, n[i-1]), fontsize=8, ha='center', style="italic")
        plt.xlabel("")
        plt.ylabel("")
        plt.axhline(0, color='grey')
        plt.grid(True, axis="y")

    elif i == 3: # women high iq
        err = err_[i-1]
        ax3 = plt.subplot(2, 2, 1)
        ax3.errorbar(X, Y, yerr = np.array(err), fmt='mo', ecolor="c", capthick=2)
        # plt.bar(X, Y, yerr=err)
        # ax3.set_xticklabels(labels)
        ax3.get_xaxis().set_visible(False)
        plt.title("HIGH FIQ", fontsize=12, weight="bold")
        acc = (acc_list_original[i-1][0] * 100).astype(int) 
        plt.text(4, 1.5, "Accuracy=%i%%, n=%i participants" %(acc, n[i-1]), fontsize=8, ha='center', style="italic")
        plt.xlabel("")
        plt.ylabel("FEMALE", fontsize=12, weight="bold") 
        plt.axhline(0, color='grey')
        plt.grid(True, axis="y")

    else: # women low iq
        ax4 = plt.subplot(2, 2, 2)
        ax4.get_xaxis().set_visible(False)
        # ax4.set_xticklabels(labels)
        # sns.barplot(x="items", y="coefs", data=df_4, palette=my_palette)
        # plt.text(0.5, 0.5, "score=%0.2f, n=%i" %(acc_list[i-1], n[i-1]), fontsize=7, ha='center', style="italic")
        plt.text(0.5, 1.5, "Not enough data. %i adults, %i teenagers." %(not_enough[1], not_enough[0]), fontsize=8, ha='center', style="italic")
        plt.grid(True, axis="y")
        plt.axhline(0, color='grey')
        plt.title("LOW FIQ", fontsize=12, weight="bold")
        plt.xlabel("")
        plt.ylabel("")
    # plt.tight_layout()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=12)
    plt.ylim([-2.1, 2.1])
    sns.despine(bottom=True)


plt.gcf().subplots_adjust(bottom=0.25)
# plt.suptitle("Predict adult vs adolescence from gender and IQ", y=0.95, fontsize=14, weight="bold")
# plt.savefig("Predict_Age_bs")
plt.show()




############################
# plot confusion matrices #
###########################

f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")
for i in range(1, 5):
    ix = i - 1
    class_names = ["Teenager", "Adult"]
    class_names_fake = ["", ""]


    if i == 1: # men high iq
        ax1 = plt.subplot(2, 2, 3)
        y_pred = predictions["pred:real_{}".format(ix)][0]
        y_true = predictions["pred:real_{}".format(ix)][1].values

        # matrix
        cm = confusion_matrix(y_true, y_pred)
        print(len(y_true))
        print(cm)
        cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = (cm_ * 100) # percentage
        for indx, i in enumerate(cm):
            for indy, j in enumerate(i):
                j = round(j, 2)
                print(j)
                cm[indx, indy] = j
        print(cm)
        plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=12)
        plt.yticks(tick_marks, class_names, fontsize=12)
        rotateTickLabels(ax1, -55, 'x')
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]) + "%",
                     horizontalalignment="center",
                     color= "black", fontsize=15)
        plt.xlabel('Predicted label', fontsize=15)
        plt.ylabel("MALE", fontsize=20, weight="bold")



    elif i == 2: # men low iq
        ax2 = plt.subplot(2, 2, 4)
        y_pred = predictions["pred:real_{}".format(ix)][0]
        y_true = predictions["pred:real_{}".format(ix)][1].values

        # matrix
        cm = confusion_matrix(y_true, y_pred)
        print(len(y_true))
        print(cm)
        cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = (cm_ * 100) # percentage
        for indx, i in enumerate(cm): # round at 0.01
            for indy, j in enumerate(i):
                j = round(j, 2)
                print(j)
                cm[indx, indy] = j
        print(cm)
        plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=12)
        rotateTickLabels(ax2, -55, 'x')
        # no tick on y axis
        tick_marks = np.arange(len(class_names_fake))
        plt.yticks(tick_marks, class_names_fake, fontsize=12)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]) + "%",
                     horizontalalignment="center",
                     color= "black", fontsize=15)
        plt.xlabel('Predicted label', fontsize=15)
        plt.ylabel("True label", fontsize=15)
        ax2.yaxis.set_label_position("right")



    elif i == 3: # women high iq
        ax3 = plt.subplot(2, 2, 1)
        y_pred = predictions["pred:real_{}".format(ix)][0]
        y_true = predictions["pred:real_{}".format(ix)][1].values
        # matrix
        cm = confusion_matrix(y_true, y_pred)
        print(len(y_true))
        print(cm)
        cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = (cm_ * 100)
        for indx, i in enumerate(cm): # round at 0.01
            for indy, j in enumerate(i):
                j = round(j, 2)
                print(j)
                cm[indx, indy] = j
        print(cm)
        plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(class_names))
        plt.yticks(tick_marks, class_names, fontsize=12)
        rotateTickLabels(ax3, -55, 'x')
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]) + "%",
                     horizontalalignment="center",
                     color= "black", fontsize=15)
        # no tick on x axis
        tick_marks = np.arange(len(class_names_fake))
        plt.xticks(tick_marks, class_names_fake, fontsize=12)
        # plt.xticks([])
        plt.xlabel("")
        plt.ylabel("FEMALE", fontsize=20, weight="bold")
        plt.title("HIGH FIQ", fontsize=20, weight="bold")



    else: # women low iq
        ax4 = plt.subplot(2, 2, 2)
        ax4.get_xaxis().set_visible(False)
        # ax4.set_xticklabels(labels)
        # sns.barplot(x="items", y="coefs", data=df_4, palette=my_palette)
        # plt.text(0.5, 0.5, "score=%0.2f, n=%i" %(acc_list[i-1], n[i-1]), fontsize=7, ha='center', style="italic")
        plt.text(0.5, 0.5, "Not enough data. %i adults, %i teenagers." %(not_enough[1], not_enough[0]), fontsize=8, ha='center', style="italic")
        plt.title("LOW FIQ", fontsize=20, weight="bold")
        # no tick on x axis
        tick_marks = np.arange(len(class_names_fake))
        plt.xticks(tick_marks, class_names_fake, fontsize=12)
        plt.yticks(tick_marks, class_names_fake, fontsize=12)
        plt.xlabel("")
        plt.ylabel("True label", fontsize=15)
        ax4.yaxis.set_label_position("right")

plt.tight_layout()
# plt.savefig("confusion_matrix_AGE")
plt.show()


































#####################################################################
            ##########################################
            ################ Gender as Y ################
            ##########################################
#####################################################################
# Set the raw data
df = pd.read_excel('ABIDE_full_GSIQ.xlsx').reset_index(drop=True)
df.Sex[df.Sex == 2] = 0 #0 = Women, 1 = men

cols = df.columns.tolist()
cols = cols[:-6] + cols[4:5] + cols[3:4] + cols[5:]
df = df[cols] 

X = df.values[:, 0:6] #Adi-Ados
ss_X = StandardScaler()
X = ss_X.fit_transform(X)
X_colnames = ["ADI-R Soc.", "ADI-R Comm.", "ADI-R Repet. Behav.", "ADOS Soc.",
                "ADOS Comm.", "ADOS Repet. Behav."]
df_scaled = pd.DataFrame(X, columns=X_colnames)
df_scaled["FIQ"] = df.FIQ
df_scaled["Sex"] = df.Sex
df_scaled["Age"] = df.Age
                
df = df_scaled                
                
                

                
######################################
# graph 3: Y = Gender, X = IQ + Age #
######################################

X_df1 = df[df.Age > 18][df.FIQ > df.FIQ.median()].reset_index(drop=True)  # adult high FIQ 33 subjects

X_df2 = df[df.Age > 18][df.FIQ < df.FIQ.median()].reset_index(drop=True)  # adult low FIQ #17 subjects

X_df3 = df[df.Age < 18][df.FIQ > df.FIQ.median()].reset_index(drop=True)  # non adult high FIQ #94 subjects

X_df4 = df[df.Age < 18][df.FIQ < df.FIQ.median()].reset_index(drop=True)  # non adult low FIQ #110 subjects

X_all = [X_df1, X_df2, X_df3, X_df4]
n = [len(X_df1), len(X_df2), len(X_df3), len(X_df4)]               


##################################
#### CLASSIFICATION ANALYSIS #####
##################################

# one for each df
acc_list_bs = [[], [], [], []]
coef_list_bs = [[], [], [], []]
acc_list_original = [[], [], [], []]
coef_list_original = [[], [], [], []]

predictions = {}

for ind, df in enumerate(X_all):
    # define dataframe for each gender
    df_men = df[df.Sex == 1]
    df_women = df[df.Sex == 0]
    
    # always more men than women thus oversampling the women dataset
    if np.float(len(df_men))/np.float(len(df_women)) <= 3: #1/3 is an arbitrary accepted maximum oversizing
        
        # UPSAMPLING
        sample_index_up = np.random.choice(df_women.index, len(df_men))
        df_women_up = df_women.loc[sample_index_up]
        
        n[ind] = len(df_men)*2
        
        # get the weight form the original sample
        df_original = pd.concat([df_women_up, df_men], ignore_index=True)
        # prepare the arrays of data          
        X = df_original.values[:, 0:6] #Adi-Ados
        ss_X = StandardScaler()
        X = ss_X.fit_transform(X)            
        y = df_original.Sex
        
        #run the logistic regression for the specific subsample 
        clf = LogisticRegression()
        clf.fit(X, y)
        acc = clf.score(X, y)
        y_pred = clf.predict(X)
        predictions["pred:real_{}".format(ind)] = [y_pred, y]

        acc_list_original[ind].append(acc)
        coef_list_original[ind].append(clf.coef_)
        print("acc: {}".format(acc))
    
            
    
        for i_subsample in range(100):
            # bootstrapping
            sample_index = np.random.choice(df_original.index, len(df_original)) # resample with replacement
            df_bs = df_original.loc[sample_index]
                        
            # prepare the arrays of data          
            X = df_bs.values[:, 0:6] #Adi-Ados
            ss_X = StandardScaler()
            X = ss_X.fit_transform(X)            
            y = df_bs.Sex
            
            #run the logistic regression for the specific subsample 
            clf = LogisticRegression()
            clf.fit(X, y)
            acc = clf.score(X, y)

            acc_list_bs[ind].append(acc)
            coef_list_bs[ind].append(clf.coef_)


    else:
        if len(df_women) >= 7:
            
            # DOWNSAMPLING
            sample_index_down = np.random.choice(df_men.index, len(df_women))
            df_men_down = df_men.loc[sample_index_down]
            # size of the total df because of the downsampling
            n[ind] = len(df_women)*2

            # get the weight from the analysis with the original sample
            df_original = pd.concat([df_women, df_men_down], ignore_index=True)
            # prepare the arrays of data          
            X = df_original.values[:, 0:6] #Adi-Ados
            ss_X = StandardScaler()
            X = ss_X.fit_transform(X)            
            y = df_original.Sex
        
            #run the logistic regression for our original sample
            clf = LogisticRegression()
            clf.fit(X, y)
            acc = clf.score(X, y)
            y_pred = clf.predict(X)
            predictions["pred:real_{}".format(ind)] = [y_pred, y]

            acc_list_original[ind].append(acc)
            coef_list_original[ind].append(clf.coef_)
            print("acc: {}".format(acc))
            
            
            for i_subsample in range(100):
                # bootstrapping
                sample_index = np.random.choice(df_original.index, len(df_original)) # resample with replacement
                df_bs = df_original.loc[sample_index]
                                        
                # prepare the arrays of data          
                X = df_bs.values[:, 0:6] #Adi-Ados
                ss_X = StandardScaler()
                X = ss_X.fit_transform(X)            
                y = df_bs.Sex
                
                #run the logistic regression for the specific subsample 
                clf = LogisticRegression()
                clf.fit(X, y)
                acc = clf.score(X, y)

                acc_list_bs[ind].append(acc)
                coef_list_bs[ind].append(clf.coef_)

        else:
            print("NOT ENOUGH DATA FOR X_df{}".format(ind+1))
            print("n_Men = {}, n_Women = {}".format(len(df_men), len(df_women)))
            not_enough = [len(df_men), len(df_women)]


#  Create the df to plot the bootsrap confidence interval of the weights
all_df = []
for num_df in range(0, 4): # there isn't enough data for df_4
    if coef_list_bs[num_df] != []:
        all_df.append(pd.DataFrame(np.squeeze(np.array(coef_list_bs[num_df])),
                    columns = X_colnames))
    else:
        print("No data for df {}".format(num_df+1))
        all_df.append([])


bs_errorbar_per_df = {}
for num_df, df in enumerate(all_df):
    bs_err_per_items = {}
    for ind, items in enumerate(df):
        bs_err = scipy.stats.scoreatpercentile(df[items], [5, 95], interpolation_method='fraction', axis=None)
        bs_err_per_items[items] = bs_err
    bs_errorbar_per_df[num_df] = bs_err_per_items

# get rid of unnecessary dimension in the list of coef
for ind, i in enumerate(coef_list_original):
    coef_list_original[ind] = np.squeeze(i)

# calculating boostrap CI + and - values above and under the original data
err_ = []
for df in range(0, len(bs_errorbar_per_df)):
    err_l_ = []
    err_u_ = []
    if bs_errorbar_per_df[df] != {}:
        for ind, names in enumerate(X_colnames):
            err_l = coef_list_original[df][ind] - bs_errorbar_per_df[df][names][0]
            err_u = bs_errorbar_per_df[df][names][1] - coef_list_original[df][ind]
            err_l_.append(err_l)
            err_u_.append(err_u)
        err = [err_l_, err_u_]
        err_.append(err)
    else:
        err_.append([])


#################################################################
# Extract information about which point is important to analyse #
#################################################################
for i in range(0, 3):
    if i == 0:
        print("-----adult high IQ------")
    if i > 0:
        i += 1
    if i == 2:
        print("-----ado high IQ------")
    if i == 3:
        print("-----ado low IQ------")
    # compute the average absolute weight for each domain
    ave = np.mean(abs(coef_list_original[i]))    
    for ind, j in enumerate(X_colnames):
        err_bars = bs_errorbar_per_df[i][j]
        value = coef_list_original[i][ind]
        # print("{}, value = {}, error bars = {}".format(j, value, err_bars))
        if abs(value) >= ave - (ave*20/100):
            if err_bars[0] > 0 and err_bars[1] > 0:
                print("Interpret: {}".format(j))        
            elif err_bars[0] < 0 and err_bars[1] < 0:
                print("Interpret: {}".format(j))
            else:
                # compute the percentage of above and below 0:
                tot = abs(err_bars[0]) + abs(err_bars[1])
                lower_bound_perc = (abs(err_bars[0]) * 100)/tot
                upper_bound_perc = (abs(err_bars[1]) * 100)/tot
                # print(lower_bound_perc)
                # print(upper_bound_perc)
                if lower_bound_perc <25 or upper_bound_perc<25:
                    print("Interpret: {}".format(j))


#####################
## PLOT ############# IQ + age = gender
#####################

X = np.array([1, 2 , 3, 4, 5, 6])

f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")
labels = [""] + [X_colnames[i] for i in range(0, len(X_colnames))] # labels starts at index 1

for i in range(1, 5):
    Y = np.squeeze(coef_list_original[i-1]) # the weights obtained with the original sample
    if i == 1: # high iq adult
        err = err_[i-1]
        ax1 = plt.subplot(2, 2, i)
        ax1.errorbar(X, Y, yerr = np.array(err), fmt='mo', ecolor="c", capthick=2)
        # plt.bar(X, Y, yerr=err)
        # ax1.set_xticklabels(labels)
        ax1.get_xaxis().set_visible(False)
        acc = (acc_list_original[i-1][0] * 100).astype(int) 
        plt.text(4, 1.5, "Accuracy=%i%%, n=%i participants" %(acc, n[i-1]), fontsize=8, ha='center', style="italic")
        plt.ylabel("ADULT", fontsize=12, weight="bold")
        plt.title("HIGH FIQ", fontsize=12, weight="bold")
        plt.xlabel("")
        plt.axhline(0, color='grey')
        plt.grid(True, axis="y")
        
    elif i == 2: # low iq adult
        ax2 = plt.subplot(2, 2, i)
        ax2.get_xaxis().set_visible(False)
        plt.text(0.5, 1.5, "Not enough data. %i females, %i males." %(not_enough[1], not_enough[0]), fontsize=8, ha='center', style="italic")
        plt.ylabel("")
        plt.xlabel("")
        plt.grid(True, axis="y")
        plt.axhline(0, color='grey')
        plt.title("LOW FIQ", fontsize=12, weight="bold")
        
    elif i == 3: # non adult high FIQ
        err = err_[i-1]
        ax3 = plt.subplot(2, 2, i)
        ax3.errorbar(X, Y, yerr = np.array(err), fmt='mo', ecolor="c", capthick=2)
        # plt.bar(X, Y, yerr=err)
        ax3.set_xticklabels(labels)
        rotateTickLabels(ax3, -55, 'x')
        acc = (acc_list_original[i-1][0] * 100).astype(int) 
        plt.text(4, 1.5, "Accuracy=%i%%, n=%i participants" %(acc, n[i-1]), fontsize=8, ha='center', style="italic")
        plt.xlabel("")
        plt.ylabel("TEENAGER", fontsize=12, weight="bold") 
        plt.axhline(0, color='grey')
        plt.grid(True, axis="y")
              
    else: # non adult low FIQ
        err = err_[i-1]
        ax4 = plt.subplot(2, 2, i)
        ax4.errorbar(X, Y, yerr = np.array(err), fmt='mo', ecolor="c", capthick=2)
        # plt.bar(X, Y, yerr=err)
        ax4.set_xticklabels(labels)
        rotateTickLabels(ax4, -55, 'x')
        acc = (acc_list_original[i-1][0] * 100).astype(int) 
        plt.text(4, 1.5, "Accuracy=%i%%, n=%i participants" %(acc, n[i-1]), fontsize=8, ha='center', style="italic")
        plt.ylabel("")
        plt.xlabel("")
        plt.axhline(0, color='grey')
        plt.grid(True, axis="y")
        
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=12)
    plt.ylim([-2.1, 2.1])
    sns.despine(bottom=True)

     
plt.gcf().subplots_adjust(bottom=0.25)
# plt.suptitle("Predict women vs men from age and IQ", y=0.95, fontsize=14, weight="bold")
# plt.savefig("Predict_gender_bs")
plt.show()


# 
# # plot confusion matrices
# title = ["adult high IQ", "adolescent high IQ", "adolescent low IQ"]
# for ind, ix in enumerate([0, 2, 3]): # no first lap
#     y_pred = predictions["pred:real_{}".format(ix)][0]
#     y_true = predictions["pred:real_{}".format(ix)][1].values
# 
# 
#     class_names = ["Women", "Men"]
# 
#     cnf_matrix = confusion_matrix(y_true, y_pred)
#     print "***"
#     print cnf_matrix
#     print "-"
#         # print sensitivity_specificity(y_true, y_pred)
#     print check_cm(y_true, y_pred)
#     print "----"
#     print "***"
#     f, ([ax1, ax2]) = plt.subplots(1, 2, figsize=(8, 8))
#     ax1 = plt.subplot(1, 2, 1)
#     plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title= title[ind] + ' without normalization')
#     rotateTickLabels(ax1, -55, 'x')
#     ax2 = plt.subplot(1, 2, 2)
#     plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title= title[ind] + ' with normalization')
#     rotateTickLabels(ax2, -55, 'x')
#     plt.savefig("confusion_matrix_Gender{}".format(title[ind]))
#     plt.show()



############################
# plot confusion matrices #
###########################


f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")
for i in range(1, 5):
    ix = i - 1
    class_names = ["Female", "Male"]
    class_names_fake = ["", ""]


    if i == 1: # adult high iq
        ax1 = plt.subplot(2, 2, 1)
        y_pred = predictions["pred:real_{}".format(ix)][0]
        y_true = predictions["pred:real_{}".format(ix)][1].values
        # matrix
        cm = confusion_matrix(y_true, y_pred)
        print(len(y_true))
        print(cm)
        cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = (cm_ * 100) # percentage
        for indx, i in enumerate(cm):
            for indy, j in enumerate(i):
                j = round(j, 2)
                print(j)
                cm[indx, indy] = j
        print(cm)
        plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(class_names))
        plt.yticks(tick_marks, class_names, fontsize=12)
        rotateTickLabels(ax1, -55, 'x')
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]) + "%",
                     horizontalalignment="center",
                     color= "black", fontsize=15)
        # no tick on x axis
        tick_marks = np.arange(len(class_names_fake))
        plt.xticks(tick_marks, class_names_fake, fontsize=12)
        # plt.xticks([])
        plt.xlabel("")
        plt.ylabel("ADULT", fontsize=20, weight="bold")
        
        plt.title("HIGH FIQ", fontsize=20, weight="bold")


    
    elif i == 2: # low iq adult
        ax2 = plt.subplot(2, 2, 2)
        ax2.get_xaxis().set_visible(False)
        # ax4.set_xticklabels(labels)
        # sns.barplot(x="items", y="coefs", data=df_4, palette=my_palette)
        # plt.text(0.5, 0.5, "score=%0.2f, n=%i" %(acc_list[i-1], n[i-1]), fontsize=7, ha='center', style="italic")
        plt.text(0.5, 0.5, "Not enough data. %i females, %i males." %(not_enough[1], not_enough[0]), fontsize=8, ha='center', style="italic")
        plt.title("LOW FIQ", fontsize=20, weight="bold")
        # no tick on x axis
        tick_marks = np.arange(len(class_names_fake))
        plt.xticks(tick_marks, class_names_fake, fontsize=12)
        plt.yticks(tick_marks, class_names_fake, fontsize=12)
        plt.xlabel("")
        plt.ylabel("True label", fontsize=15)
        ax2.yaxis.set_label_position("right")
    
                
    elif i == 3: # non adult high FIQ
        ax3 = plt.subplot(2, 2, 3)
        y_pred = predictions["pred:real_{}".format(ix)][0]
        y_true = predictions["pred:real_{}".format(ix)][1].values
        
        # matrix
        cm = confusion_matrix(y_true, y_pred)
        print(len(y_true))
        print(cm)
        cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = (cm_ * 100) # percentage
        for indx, i in enumerate(cm):
            for indy, j in enumerate(i):
                j = round(j, 2)
                print(j)
                cm[indx, indy] = j
        print(cm)
        plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=12)
        plt.yticks(tick_marks, class_names, fontsize=12)
        rotateTickLabels(ax3, -55, 'x')
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]) + "%",
                     horizontalalignment="center",
                     color= "black", fontsize=15)
        plt.xlabel('Predicted label', fontsize=15)
        plt.ylabel("TEENAGER", fontsize=20, weight="bold")
        
              
    elif i == 4: # non adult low FIQ
        ax4 = plt.subplot(2, 2, 4)
        y_pred = predictions["pred:real_{}".format(ix)][0]
        y_true = predictions["pred:real_{}".format(ix)][1].values
        
        # matrix
        cm = confusion_matrix(y_true, y_pred)
        print(len(y_true))
        print(cm)
        cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = (cm_ * 100) # percentage
        for indx, i in enumerate(cm):
            for indy, j in enumerate(i):
                j = round(j, 2)
                print(j)
                cm[indx, indy] = j
        print(cm)
        plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=12)
        rotateTickLabels(ax4, -55, 'x')
        # no tick on y axis
        tick_marks = np.arange(len(class_names_fake))
        plt.yticks(tick_marks, class_names_fake, fontsize=12)
        
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]) + "%",
                     horizontalalignment="center",
                     color= "black", fontsize=15)
        plt.xlabel('Predicted label', fontsize=15)
        plt.ylabel("True label", fontsize=15)
        ax4.yaxis.set_label_position("right")

plt.tight_layout()
# plt.savefig("confusion_matrix_GENDER")
plt.show()




















                
                
#####################################################################
            ##########################################
            ################ FIQ as Y ################
            ##########################################
#####################################################################

np.random.seed(0)
# Set the raw data
df = pd.read_excel('ABIDE_full_GSIQ.xlsx').reset_index(drop=True)
df.Sex[df.Sex == 1] = "Men"
df.Sex[df.Sex == 2] = "Women"

cols = df.columns.tolist()
cols = cols[:-6] + cols[4:5] + cols[3:4] + cols[5:]
df = df[cols] 


X = df.values[:, 0:6] #Adi-Ados
ss_X = StandardScaler()
X = ss_X.fit_transform(X)
X_colnames = ["ADI-R Soc.", "ADI-R Comm.", "ADI-R Repet. Behav.", "ADOS Soc.",
                "ADOS Comm.", "ADOS Repet. Behav."]
df_scaled = pd.DataFrame(X, columns=X_colnames)
df_scaled["FIQ"] = df.FIQ
df_scaled["Sex"] = df.Sex
df_scaled["Age"] = df.Age

df = df_scaled

######################################
# graph 1: Y = IQ, X = age + gender #
######################################

X_df1 = df[df.Sex == "Men"][df.Age > 18].reset_index(drop=True) # men adult


X_df2 = df[df.Sex == "Men"][df.Age < 18].reset_index(drop=True) # men non-adult


X_df3 = df[df.Sex == "Women"][df.Age > 18].reset_index(drop=True) # women adult  # 8 subjects...


X_df4 = df[df.Sex == "Women"][df.Age < 18].reset_index(drop=True) # women non-adult

X_all = [X_df1, X_df2, X_df3, X_df4]
n = [len(X_df1), len(X_df2), len(X_df3), len(X_df4)]

##################################
#### CLASSIFICATION ANALYSIS #####
##################################

# one for each df
acc_list_bs = [[], [], [], []]
coef_list_bs = [[], [], [], []]
acc_list_original = [[], [], [], []]
coef_list_original = [[], [], [], []]
predictions = {}

for ind, df in enumerate(X_all):
    # divide into two category, high and low IQ
    df_lowIQ = df[df.FIQ < df.FIQ.median()]
    df_highIQ = df[df.FIQ >= df.FIQ.median()]
    # transform into a binary variable
    df_lowIQ.FIQ = 0
    df_highIQ.FIQ = 1
    
    # get the weight form the original sample
    df_original = pd.concat([df_lowIQ, df_highIQ], ignore_index=True)
    # prepare the arrays of data          
    X = df_original.values[:, 0:6] #Adi-Ados
    ss_X = StandardScaler()
    X = ss_X.fit_transform(X)            
    y = df_original.FIQ
        
    #run the logistic regression for the specific subsample 
    clf = LogisticRegression()
    clf.fit(X, y)
    acc = clf.score(X, y)
    y_pred = clf.predict(X)
    predictions["pred:real_{}".format(ind)] = [y_pred, y]
    
    acc_list_original[ind].append(acc)
    coef_list_original[ind].append(clf.coef_)
    print("acc: {}".format(acc))

    # no down or upsampling for FIQ, since it's a 50/50 outcome
    
    #subsample each group
    for i_subsample in range(100):
        # bootstrapping
        sample_index = np.random.choice(df_original.index, len(df_original)) # resample with replacement
        df_bs = df_original.loc[sample_index]
        
        if len(np.unique(df_bs.FIQ)) == 1:
            while len(np.unique(df_bs.FIQ)) == 1:
                sample_index = np.random.choice(df_original.index, len(df_original)) # resample with replacement
                df_bs = df_original.loc[sample_index]
                
        # prepare the arrays of data          
        X = df_bs.values[:, 0:6] #Adi-Ados
        ss_X = StandardScaler()
        X = ss_X.fit_transform(X)            
        y = df_bs.FIQ
        
        #run the logistic regression for the specific subsample 
        clf = LogisticRegression()
        clf.fit(X, y)
        acc = clf.score(X, y)

        acc_list_bs[ind].append(acc)
        coef_list_bs[ind].append(clf.coef_)
        # print "acc: %.2f" % (acc)

    
#  Create the df to plot the bootsrap confidence interval of the weights
all_df = []
for num_df in range(0, 4): # there isn't enough data for df_4
    if coef_list_bs[num_df] != []:
        all_df.append(pd.DataFrame(np.squeeze(np.array(coef_list_bs[num_df])),
                    columns = X_colnames))
    else:
        print("No data for df {}".format(num_df))

bs_errorbar_per_df = {}
for num_df, df in enumerate(all_df):
    bs_err_per_items = {}
    for ind, items in enumerate(df):
        bs_err = scipy.stats.scoreatpercentile(df[items], [5, 95], interpolation_method='fraction', axis=None)
        bs_err_per_items[items] = bs_err
    bs_errorbar_per_df[num_df] = bs_err_per_items

# get rid of unnecessary dimension in the list of coef
for ind, i in enumerate(coef_list_original):
    coef_list_original[ind] = np.squeeze(i)

# calculating boostrap CI + and - values above and under the original data
err_ = []
for df in range(0, len(bs_errorbar_per_df)):
    err_l_ = []
    err_u_ = []
    for ind, names in enumerate(X_colnames):
        err_l = coef_list_original[df][ind] - bs_errorbar_per_df[df][names][0]
        err_u = bs_errorbar_per_df[df][names][1] - coef_list_original[df][ind]
        err_l_.append(err_l)
        err_u_.append(err_u)
    err = [err_l_, err_u_]
    err_.append(err)    


#################################################################
# Extract information about which point is important to analyse #
#################################################################
for i in range(0, 4):
    if i == 0:
        print("-----adult men------")
    if i == 1:
        print("-----ado men------")
    if i == 2:
        print("-----adult women------")
    if i == 3:
        print("-----ado women------")
    # compute the average absolute weight for each domain
    ave = np.mean(abs(coef_list_original[i]))    
    for ind, j in enumerate(X_colnames):
        err_bars = bs_errorbar_per_df[i][j]
        value = coef_list_original[i][ind]
        # print("{}, value = {}, error bars = {}".format(j, value, err_bars))
        if abs(value) >= ave - (ave*20/100):
            if err_bars[0] > 0 and err_bars[1] > 0:
                print("Interpret: {}".format(j))        
            elif err_bars[0] < 0 and err_bars[1] < 0:
                print("Interpret: {}".format(j))
            else:
                # compute the percentage of above and below 0:
                tot = abs(err_bars[0]) + abs(err_bars[1])
                lower_bound_perc = (abs(err_bars[0]) * 100)/tot
                upper_bound_perc = (abs(err_bars[1]) * 100)/tot
                # print(lower_bound_perc)
                # print(upper_bound_perc)
                if lower_bound_perc <25 or upper_bound_perc<25:
                    print("Interpret: {}".format(j))


    



#####################
## PLOT ############# gender + age = IQ
#####################

X = np.array([1, 2 , 3, 4, 5, 6])

f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")
labels = [""] + [X_colnames[i] for i in range(0, len(X_colnames))] # labels starts at index 1
for i in range(1, 5):
    Y = np.squeeze(coef_list_original[i-1]) # the weights obtained with the original sample
    if i == 1: # men adult
        err = err_[i-1]
        ax1 = plt.subplot(2, 2, i)
        ax1.errorbar(X, Y, yerr = np.array(err), fmt='mo', ecolor="c", capthick=2)
        # plt.bar(X, Y, yerr=err)
        # ax1.set_xticklabels(labels)
        ax1.get_xaxis().set_visible(False)
        plt.ylabel("MALE", fontsize=12, weight="bold")
        plt.title("ADULT", fontsize=12, weight="bold")
        acc = (acc_list_original[i-1][0] * 100).astype(int) 
        plt.text(4, 1.5, "Accuracy=%i%%, n=%i participants" %(acc, n[i-1]), fontsize=8, ha='center', style="italic")
        plt.xlabel("")
        plt.axhline(0, color='grey')
        plt.grid(True, axis="y")
        
    elif i == 2: # men non adult
        err = err_[i-1]
        ax2 = plt.subplot(2, 2, i)
        ax2.errorbar(X, Y, yerr = np.array(err), fmt='mo', ecolor="c", capthick=2)
        # plt.bar(X, Y, yerr=err)
        # ax2.set_xticklabels(labels)
        ax2.get_xaxis().set_visible(False)
        acc = (acc_list_original[i-1][0] * 100).astype(int) 
        plt.text(4, 1.5, "Accuracy=%i%%, n=%i participants" %(acc, n[i-1]), fontsize=8, ha='center', style="italic")
        plt.xlabel("")
        plt.ylabel("")
        plt.title("TEENAGER", fontsize=12, weight="bold")
        plt.axhline(0, color='grey')
        plt.grid(True, axis="y")

    elif i == 3: # women adult
        err = err_[i-1]
        ax3 = plt.subplot(2, 2, i)
        ax3.errorbar(X, Y, yerr = np.array(err), fmt='mo', ecolor="c", capthick=2)
        # plt.bar(X, Y, yerr=err)
        ax3.set_xticklabels(labels)
        rotateTickLabels(ax3, -55, 'x')
        acc = (acc_list_original[i-1][0] * 100).astype(int) 
        plt.text(4, 1.5, "Accuracy=%i%%, n=%i participants" %(acc, n[i-1]), fontsize=8, ha='center', style="italic")
        plt.xlabel("")
        plt.ylabel("FEMALE", fontsize=12, weight="bold")
        plt.axhline(0, color='grey')
        plt.grid(True, axis="y")
        
    else: # women non adult
        err = err_[i-1]
        ax4 = plt.subplot(2, 2, i)
        ax4.errorbar(X, Y, yerr = np.array(err), fmt='mo', ecolor="c", capthick=2)
        # plt.bar(X, Y, yerr=err)
        ax4.set_xticklabels(labels)
        rotateTickLabels(ax4, -55, 'x')
        acc = (acc_list_original[i-1][0] * 100).astype(int) 
        plt.text(4, 1.5, "Accuracy=%i%%, n=%i participants" %(acc, n[i-1]), fontsize=8, ha='center', style="italic")
        plt.ylabel("")
        plt.xlabel("")
        plt.axhline(0, color='grey')
        plt.grid(True, axis="y")
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=12)
    plt.ylim([-2.1, 2.1])
    sns.despine(bottom=True)

      
plt.gcf().subplots_adjust(bottom=0.25)
# plt.suptitle("Predict high vs low IQ from gender and age", y=0.95, fontsize=14, weight="bold")
# plt.savefig("predict_IQ_bs")
plt.show()


# # plot confusion matrices
# title = ["Men adult", "Men adolescent", "Women adult", "Women adolescent"]
# for ix in range(0, len(predictions)):
# 
#     y_pred = predictions["pred:real_{}".format(ix)][0]
#     y_true = predictions["pred:real_{}".format(ix)][1].values
# 
#     class_names = ["Low IQ", "High IQ"]
# 
#     cnf_matrix = confusion_matrix(y_true, y_pred)
#     print "***"
#     print cnf_matrix
#     print "-"
#         # print sensitivity_specificity(y_true, y_pred)
#     print check_cm(y_true, y_pred)
#     print "----"
#     print "***"
#     f, ([ax1, ax2]) = plt.subplots(1, 2, figsize=(8, 8))
#     ax1 = plt.subplot(1, 2, 1)
#     plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title=title[ix] + ' without normalization')
#     rotateTickLabels(ax1, -55, 'x')
#     ax2 = plt.subplot(1, 2, 2)
#     plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title=title[ix] + ' with normalization')
#     rotateTickLabels(ax2, -55, 'x')
#     plt.savefig("confusion_matrix_IQ{}".format(title[ix]))
#     plt.show()




############################
# plot confusion matrices #
###########################


f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")
for i in range(1, 5):
    ix = i - 1
    class_names = ["LOW FIQ", "HIGH FIQ"]
    class_names_fake = ["", ""]


    if i == 1: # men adult
        ax1 = plt.subplot(2, 2, 1)
        y_pred = predictions["pred:real_{}".format(ix)][0]
        y_true = predictions["pred:real_{}".format(ix)][1].values
        # matrix
        cm = confusion_matrix(y_true, y_pred)
        print(len(y_true))
        print(cm)
        cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = (cm_ * 100) # percentage
        for indx, i in enumerate(cm):
            for indy, j in enumerate(i):
                j = round(j, 2)
                print(j)
                cm[indx, indy] = j
        print(cm)
        plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(class_names))
        plt.yticks(tick_marks, class_names, fontsize=12)
        rotateTickLabels(ax1, -55, 'x')
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]) + "%",
                     horizontalalignment="center",
                     color= "black", fontsize=15)
        # no tick on x axis
        tick_marks = np.arange(len(class_names_fake))
        plt.xticks(tick_marks, class_names_fake, fontsize=12)
        # plt.xticks([])
        plt.xlabel("")
        plt.ylabel("MALE", fontsize=20, weight="bold")
        
        plt.title("ADULT", fontsize=20, weight="bold")


    
    elif i == 2: # men non-adult
        ax2 = plt.subplot(2, 2, 2)
        y_pred = predictions["pred:real_{}".format(ix)][0]
        y_true = predictions["pred:real_{}".format(ix)][1].values
        # matrix
        cm = confusion_matrix(y_true, y_pred)
        print(len(y_true))
        print(cm)
        cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = (cm_ * 100) # percentage
        for indx, i in enumerate(cm):
            for indy, j in enumerate(i):
                j = round(j, 2)
                print(j)
                cm[indx, indy] = j
        print(cm)
        plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(class_names_fake))
        plt.xticks(tick_marks, class_names_fake, fontsize=12)
        plt.yticks(tick_marks, class_names_fake, fontsize=12)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]) + "%",
                     horizontalalignment="center",
                     color= "black", fontsize=15)
        
        plt.title("TEENAGER", fontsize=20, weight="bold")
        # no tick on x axis
        tick_marks = np.arange(len(class_names_fake))
        plt.xticks(tick_marks, class_names_fake, fontsize=12)
        plt.yticks(tick_marks, class_names_fake, fontsize=12)
        plt.xlabel("")
        plt.ylabel("True label", fontsize=15)
        ax2.yaxis.set_label_position("right")
    
                
    elif i == 3: # women adult
        ax3 = plt.subplot(2, 2, 3)
        y_pred = predictions["pred:real_{}".format(ix)][0]
        y_true = predictions["pred:real_{}".format(ix)][1].values
        
        # matrix
        cm = confusion_matrix(y_true, y_pred)
        print(len(y_true))
        print(cm)
        cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = (cm_ * 100) # percentage
        for indx, i in enumerate(cm):
            for indy, j in enumerate(i):
                j = round(j, 2)
                print(j)
                cm[indx, indy] = j
        print(cm)
        plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=12)
        plt.yticks(tick_marks, class_names, fontsize=12)
        rotateTickLabels(ax3, -55, 'x')
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]) + "%",
                     horizontalalignment="center",
                     color= "black", fontsize=15)
        plt.xlabel('Predicted label', fontsize=15)
        plt.ylabel("FEMALE", fontsize=20, weight="bold")
        
              
    elif i == 4: # women non-adult
        ax4 = plt.subplot(2, 2, 4)
        y_pred = predictions["pred:real_{}".format(ix)][0]
        y_true = predictions["pred:real_{}".format(ix)][1].values
        
        # matrix
        cm = confusion_matrix(y_true, y_pred)
        print(len(y_true))
        print(cm)
        cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = (cm_ * 100) # percentage
        for indx, i in enumerate(cm):
            for indy, j in enumerate(i):
                j = round(j, 2)
                print(j)
                cm[indx, indy] = j
        print(cm)
        plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=12)
        rotateTickLabels(ax4, -55, 'x')
        # no tick on y axis
        tick_marks = np.arange(len(class_names_fake))
        plt.yticks(tick_marks, class_names_fake, fontsize=12)
        
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]) + "%",
                     horizontalalignment="center",
                     color= "black", fontsize=15)
        plt.xlabel('Predicted label', fontsize=15)
        plt.ylabel("True label", fontsize=15)
        ax4.yaxis.set_label_position("right")

plt.tight_layout()
# plt.savefig("confusion_matrix_FIQ")
plt.show()



























def sensitivity_specificity(y_true, y_pred):
    ll = hl = hh = lh = 0
    for ind, i in enumerate(y_true):
        if i == y_pred[ind] == 0:
            ll +=1
        if i == y_pred[ind] == 1:
            hh +=1
        if i != y_pred[ind]:
            if i == 1:
                hl +=1
            if i == 0:
                lh +=1
    print(ll, hh, lh, hl)
    sensitivity = float(ll)/float((ll+lh))
    specificity = float(hh)/float((hh+hl))
    return sensitivity, specificity 
    
def check_cm(y_true, y_pred):
    ll = hl = hh = lh = 0
    for ind, i in enumerate(y_true):
        if i == y_pred[ind] == 0:
            ll +=1
        if i == y_pred[ind] == 1:
            hh +=1
        if i != y_pred[ind]:
            if i == 1:
                hl +=1
            if i == 0:
                lh +=1
    print(ll, lh)
    print(hl, hh)
    

    
    
