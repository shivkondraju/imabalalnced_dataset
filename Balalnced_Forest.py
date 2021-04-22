import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier


#############################
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


#Actual Code starts here
data=pd.read_csv('UPGRD_SECN_DATA_ABT2.csv',low_memory=False)

columns=data.columns.tolist()
columns=[c for c in columns if c not in ['upgrd_customer_class']]

X=data[columns]


Y=data['upgrd_customer_class']


newDF=DataFrameImputer().fit_transform(X)


missing=newDF.columns[newDF.isnull().any()]

newDF=newDF.drop(['REMOTE_START_PARKING_ASSIST_CD', 'NEAR_FIELD_COMMUNICATION_FLG',
       'TIRE_MOBILE_KIT_FLG', 'PREFERRED_CHANNEL_CD', 'PERSONICX_CATEGORY_CD'],axis=1)

le = MultiColumnLabelEncoder()
X = le.fit_transform(X.astype(str))

transformer = RobustScaler().fit(X)
X=transformer.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42,stratify=Y)

#Balanced Random Forest
brf = BalancedRandomForestClassifier(n_estimators=300,random_state=0)
brf.fit(X_train,y_train)
print(f1_score(y_test,brf.predict(X_test)))

