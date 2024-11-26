from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

#importing data
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

#wrangling

#print(housing.info()) #non-null values for each category. see total_bedrooms is incomplete 
#print(housing["ocean_proximity"].value_counts()) #this only takes a few values
#print(housing.describe()) #this gives rough statistics of the features. 

#now some plots
import matplotlib.pyplot as plt #dependency for pandas .hist()

housing.hist(bins=50,figsize=(12,8))
plt.show()







