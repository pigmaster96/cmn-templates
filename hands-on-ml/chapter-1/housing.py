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

##wrangling##

#print(housing.info()) #non-null values for each category. see total_bedrooms is incomplete 
#print(housing["ocean_proximity"].value_counts()) #this only takes a few values
#print(housing.describe()) #this gives rough statistics of the features. 

#import matplotlib.pyplot as plt #dependency for pandas .hist()

#housing.hist(bins=50,figsize=(12,8))
#plt.show() #note median income is in thousands, house median age and value are both capped

##test set
from sklearn.model_selection import train_test_split
#we want a reproducible split of test/train set, so we set a seed(42)
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42) #Split arrays or matrices into random train and test subsets. 
#note that random sampling on small datasets may not be representative of the cohort! random sampling may not always be the way





