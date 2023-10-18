import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_imp(df,imp_data,cols):
    sns.set()
    fig, axes = plt.subplots(nrows = len(cols), ncols = 2)
    fig.set_size_inches(8, 20)

    for index, variable in enumerate(cols):
        sns.distplot(df[variable].dropna(), kde = False, ax = axes[index, 0])
        sns.distplot(imp_data["IMP" + variable], kde = False, ax = axes[index, 0], color = 'red')

        sns.boxplot(data = pd.concat([df[variable], imp_data["IMP" + variable]], axis = 1),
                    ax = axes[index, 1])

    plt.tight_layout()

def random_imputation(df, feature):

    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
    return df

def ZS(data, threshold):
    mean = np.mean(data)
    sd = np.std(data) # calculate standard deviation
    outliers = [] # create empty list to store outliers
    for i in data: # detect outliers
        z = (i - mean) / sd  # calculate z-score
        if abs(z) > threshold:  # identify outliers
            outliers.append(i)  # add to the empty list

    # print outliers
    print("The detected outliers are: ", str(outliers))

def ZSB(data, threshold):
    # Robust Zscore as a function of median and median
    # median absolute deviation (MAD) defined as
    # z-score = |x – median(x)| / mad(x)
    median = np.median(data)
    median_absolute_deviation = np.median(np.abs(data - median))
    modified_z_scores = (data - median) / median_absolute_deviation
    outliers = data[np.abs(modified_z_scores) > threshold]
    # print outliers
    print("The detected outliers are: ", str(outliers))

def IQR(data):
    sorted(data)
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    outliers = data[((data < lower_range) | (data > upper_range))]
    # print outliers
    print("The detected outliers are: ", str(outliers))

def LOF_outlier_detection(X):
    data = X
    # requires no missing value
    # select top 10 outliers
    from sklearn.neighbors import LocalOutlierFactor

    # fit the model for outlier detection (default)
    clf = LocalOutlierFactor(n_neighbors=4, contamination=0.1)

    clf.fit_predict(X)

    LOF_scores = clf.negative_outlier_factor_
    # Inliers tend to have a negative_outlier_factor_ close to -1, while outliers tend to have a larger score

    #print(LOF_scores)

    outliers = X[LOF_scores < -1.9].index

    print("Outliers: ", data.iloc[outliers])
