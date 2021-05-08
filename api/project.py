import string
from sklearn.ensemble import RandomForestRegressor
SEED=2020
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from numpy import mean
from collections import Counter
import numpy as np
import re
import pandas as pd
import pickle
import joblib
from sklearn.model_selection import cross_val_score

df= pd.read_csv(r'odessa_apts_prices_en_2020.csv')
df.drop('desc', axis=1, inplace=True)
numcol = df.select_dtypes(include=[np.number]).columns.drop('price')
catcol = df.select_dtypes(include=[np.object]).columns
df = df.drop_duplicates()
SEED=2020
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
le = LabelEncoder()


def imputer(df):
    '''*NaN values imputainon to mode for cats and median for nums*'''
    for col in df:
        if col in catcol:
            df.loc[:, col]=df.loc[:, col].fillna(df.loc[:, col].mode()[0])
        else: df.loc[:, col].fillna(df.loc[:, col].median(), inplace=True)
    return df

def encoder(df):
    district_category = {'Malinovsky':1, 'Kievsky':2, 'Primorsky':3, 'Suvorovsky':4}
    # Mapping 'district' to group
    df.loc[:, 'district']=df.loc[:, 'district'].map(district_category)

    # Bundle rare descriptions with 'Other' category
    df.loc[:,'type'] = df.loc[:,'type'].replace(['Old fund', 'Czech', 'Khrushchevka', 'Stalinka', 'Cellular', 'Belgian', 'Kharkiv', 
                                    'Under construction', 'Moscow', 'Guest', 'Jugoslavsky', 'Private house', 'A small family', 'Renovation',
                                    'After overhaul', 'After builders', 'Residential clean', 'After makeup'], 'Other')
    type_category = {'New ':1, 'Special project':2, 'Other':3}
    df.loc[:, 'type'] = df.loc[:,'type'].map(type_category)

    # Bundle rare condition with new categories
    df.loc[:,'cond'] = df.loc[:,'cond'].replace(['Renovation', 'After overhaul', 'After makeup', "Author's design", 'Design Classic', 'Modern design', 'NaN'], 'After renovation') 
    df.loc[:,'cond'] = df.loc[:,'cond'].replace(['After builders', 'Brick', 'Block-brick', "Monolith", 'Blocky', 'Expanded clay-concrete', 'House under construction', 
                                    'Aerated concrete', 'Shell rock', 'Building materials'], 'After builders') 
    df.loc[:,'cond'] = df.loc[:,'cond'].replace(['Need. in cap. renovation', 'Need. in cosm. renovation', 'Need. in tech. renovation'], 'Need renovation') 
    cond_category = {'After renovation':1, 'After builders':2, 'Need renovation':3, 'Residential clean':4}
    df.loc[:,'cond']= df.loc[:,'cond'].map(cond_category)

    # Bundle rare walls with new categories
    df.loc[:,'walls'] = df.loc[:,'walls'].replace(['Metal-plastic', 'Metalwork', 'Plastic', "Wood", 'Mixed', 'Reed, dranka ', 'NaN'], 'Other')
    walls_category = {'Brick':1, 'Silicate brick':1, 'Monolith':2, 'Concrete':2, 'Reinforced concrete':2, 'Shell rock':3, 'Shell brick':3, 'Block-brick':4, 'Blocky':4,
                    'Panel': 5, 'Aerated concrete':6, 'Foam concrete':6, 'Expanded clay-concrete': 6, 'Other':7}
    df.loc[:,'walls']=df.loc[:,'walls'].map(walls_category)


    return(df)


def one_value_cols_remover(df):
    ''' remove cols with 1 value'''
    one_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(one_value_cols, axis=1, inplace=True)
    return df

def drop_outliers(df):
    """
    Returns clear df without outliers based on IQR method
    corresponding to the observations containing more than 2 outliers according
    to the Tukey method.
    """
    df= df.loc[(df.price<170000) & (df.area<132)]
    outlier_indices = []

    # iterate over features(columns)
    for col in ['price', 'rooms', 'floor', 'floors', 'area']:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.2 * IQR 

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 2)
    # Drop outliers
    df = df.drop(multiple_outliers, axis = 0).reset_index(drop=True)
    return df

def feature_extractor(df):
    df = imputer(df)
    df = encoder(df)
    df = one_value_cols_remover(df)
    df = drop_outliers(df)
    return df



def main(df):
    df = feature_extractor(df)
    X = df.drop('price', axis=1).reset_index(drop=True)
    y = np.log1p(df.price)
    rf = RandomForestRegressor(max_depth=12, max_features=4, max_samples=0.9,
                      min_samples_leaf=2, min_samples_split=3, n_estimators=60,
                      n_jobs=-1, random_state=SEED)
    rf.fit(X, y)
    scores = cross_val_score(rf, X, y, scoring='r2', cv=3, n_jobs=-1)
    s_mean = mean(scores)
    # save the model to disk
    filename = 'finalized_model.pkl'
    pickle.dump(rf, open(filename,'wb'))
    return print ('Done')


# model = model(df)
if __name__ == "__main__":
    main(df)