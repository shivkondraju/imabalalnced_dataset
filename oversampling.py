
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report,accuracy_score
from sklearn.neighbors import LocalOutlierFactor

#Readding Data file
data=pd.read_csv('imbal_data.csv',low_memory=False)

class MultiColumnLabelEncoder:
    
    def __init__(self, columns = None):
        self.columns = columns # list of column to encode    
    def fit(self, X, y=None):
        return self    
    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        
        output = X.copy()
        
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        
        return output    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

#Actual code starts here
newDF=DataFrameImputer().fit_transform(data)

missing=newDF.columns[newDF.isnull().any()]

newDF=newDF.drop(['REMOTE_START_PARKING_ASSIST_CD', 'NEAR_FIELD_COMMUNICATION_FLG',
       'TIRE_MOBILE_KIT_FLG', 'PREFERRED_CHANNEL_CD', 'PERSONICX_CATEGORY_CD'],axis=1)

buy = newDF[newDF['upgrd_customer_class']==1]

noBuy = newDF[newDF['upgrd_customer_class']==0]


newBuy=DataFrameImputer().fit_transform(buy)
newNoBuy=DataFrameImputer().fit_transform(noBuy)

missingBuy=newBuy.columns[newBuy.isnull().any()]
missingNoBuy=newNoBuy.columns[newNoBuy.isnull().any()]

le = MultiColumnLabelEncoder()
trsfBuy = le.fit_transform(newBuy.astype(str))
trsfNoBuy = le.fit_transform(newNoBuy.astype(str))


# Implementing Oversampling for Handling Imbalanced 
le = MultiColumnLabelEncoder()
newDF = le.fit_transform(newDF.astype(str))
columns = newDF.columns.tolist()
