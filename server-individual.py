import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
defaultDataset = pd.read_csv ("c:/Users/olanrewaju.olaboye/Documents/ml-run-data-individual.csv")
print(defaultDataset.head())
x= defaultDataset[[ 'GDP', 'Rating_change' ]]
y= defaultDataset[['Default_flag']]
Xo = sm.add_constant(x)
logit_model = sm.Logit(y,Xo)
result = logit_model.fit()
print(result.summary2())