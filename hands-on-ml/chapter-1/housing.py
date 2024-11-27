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
#from sklearn.model_selection import train_test_split
##we want a reproducible split of test/train set, so we set a seed(42)
#train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42) #Split arrays or matrices into random train and test subsets.

##split based on median income "category" so the test set is representative of popn "STRATIFIED SAMPLING" we set five categories here:
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




