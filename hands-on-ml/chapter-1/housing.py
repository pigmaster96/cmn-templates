from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

##importing data
def load_housing_data():
    tarball_path=Path("datasets/housing.tgz")
    #make directory if it doesn't exist
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url="https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url,tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing=load_housing_data()

###wrangling###
import numpy as np
#print(np.shape(housing)) #shape of data matrix
#print(housing.info()) #non-null values for each category. see total_bedrooms is incomplete 
#print(housing["ocean_proximity"].value_counts()) #this only takes a few values
#print(housing.describe()) #this gives rough statistics of the features. 

import matplotlib.pyplot as plt #dependency for pandas .hist()
#housing.hist(bins=50,figsize=(12,8))
#plt.show() #note median income is in thousands, house median age and value are both capped

###splitting data###
## randomly permuted test set:
from sklearn.model_selection import train_test_split
##we want a reproducible split of test/train set, so we set a seed(42)
#rand_train_set,rand_test_set=train_test_split(housing,test_size=0.2,random_state=42) #Split arrays or matrices into random train and test subsets.

##we set up median income "categories" so the test set is representative of popn---"STRATIFIED SAMPLING". we set five categories here:
housing["income_cat"]=pd.cut(housing["median_income"],
                                     bins=[0.,1.5,3.0,4.5,6.,np.inf],
                                     labels=[1,2,3,4,5]) #five bins. not including labels leaves an interval in the dataframe
#print(housing.head())
##bar graph:
#print(housing["income_cat"])
#print(housing["income_cat"].value_counts().sort_index())#first arrange by frequency, then list categories in order
#housing["income_cat"].value_counts().sort_index().plot.bar(grid=True)
#plt.xlabel("Income category")
#plt.ylabel("Number of districts")
#plt.show()
##split using sklearn's StratifiedshuffleSplit() to get multiple test sets---for cross validation
from sklearn.model_selection import StratifiedShuffleSplit #Stratified ShuffleSplit cross-validator 
splitter=StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=42)#this just generates metadata
strat_splits=[]
for train_index,test_index in splitter.split(housing,housing["income_cat"]):#we get 10 different training and test set INDICES
    strat_train_set_n=housing.iloc[train_index]
    strat_test_set_n=housing.iloc[test_index]
    strat_splits.append([strat_train_set_n,strat_test_set_n])
#print(np.shape(strat_splits[0][0]))#training set of first of ten shuffle and stratify splits
##here we'll just look at one split
strat_train_set,strat_test_set=strat_splits[0]
#print(np.shape(strat_test_set))

##train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42) can also be used for a single split

##proportions of each popn demographic in test set for rand vs stratified
#rand_train_set,rand_test_set=train_test_split(housing,test_size=0.2,random_state=42)
#print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))
#print(rand_test_set["income_cat"].value_counts()/len(rand_test_set))
#print(len(rand_test_set)==len(strat_test_set))
##in this case both appear to relatively well represent the popn of interest. However this may not be the case in smaller datasets




