import string
from sklearn.ensemble import RandomForestRegressor
from nltk.stem import WordNetLemmatizer
SEED=2020
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
# from db import Session, Features, Apartments
from numpy import mean
from collections import Counter
import numpy as np
import re
import pandas as pd
import pickle
import joblib
from sklearn.model_selection import cross_val_score

print ('Started')
df= pd.read_csv(r'odessa_apts_prices_en_2020.csv')
numcol = df.select_dtypes(include=[np.number]).columns.drop('price')
catcol = df.select_dtypes(include=[np.object]).columns
df = df.drop_duplicates()
PUNCT_TO_REMOVE = string.punctuation
lemmatizer = WordNetLemmatizer()
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
        outlier_step = 1.5 * IQR ## can be increased to 1.7 sometimes

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 2)
#     print(df.loc[multiple_outliers][['price', 'rooms', 'floor', 'floors', 'area']]) # Show the outliers rows
    # Drop outliers
    df = df.drop(multiple_outliers, axis = 0).reset_index(drop=True)
#     print('Outliers dropped')
    return df



def feature_creator(df):
    '''polynomial and other features creation'''
    col = df.columns

    #new features based on rooms
    df['is_one_room'] = df['rooms'].map(lambda x: 1 if x == 1 else 0)
    df['is_two_room'] = df['rooms'].map(lambda x: 1 if x == 2 else 0)
    df['is_three_room'] = df['rooms'].map(lambda x: 1 if x == 3 else 0)
    df['is_four_room'] = df['rooms'].map(lambda x: 1 if x == 4 else 0)
    #new features based on floor
    df['is_first_floor'] = df['floor'].map(lambda x: 1 if x == 1 else 0)
    df['is_second_floor'] = df['floor'].map(lambda x: 1 if x == 2 else 0)
    df['is_third_floor'] = df['floor'].map(lambda x: 1 if x == 3 else 0)
    df['more_10_floor'] = df['floor'].map(lambda x: 1 if x >=10 else 0)
    # new features based on floors
    df['is_one_floor'] = df['floors'].map(lambda x: 1 if x == 1 else 0)
    df['upto_5_floors'] = df['floors'].map(lambda x: 1 if (x> 1) and (x <= 5) else 0)
    df['upto_9_floors'] = df['floors'].map(lambda x: 1 if (x> 5) and (x <= 9) else 0)
    df['upto_16_floors'] = df['floors'].map(lambda x: 1 if (x>9) and (x <= 16) else 0)
    df['more_16_floors'] = df['floors'].map(lambda x: 1 if x > 16 else 0)
    # new features based on area
    df['small_area'] = df['area'].map(lambda x: 1 if x<=42.5 else 0)
    df['medium_area'] = df['area'].map(lambda x: 1 if (x> 42.5) and (x <= 76) else 0)
    df['large_area'] = df['area'].map(lambda x: 1 if (x>76) and (x <= 115) else 0)
    df['very_large_area'] = df['area'].map(lambda x: 1 if x > 115 else 0)



    for i in numcol:

        '''new features creation based on numerical features like as polynomial features, sqrt, log, exp'''
        df[i+'**3'] = df[i]**3
        df[i+'sqrt'] = np.sqrt(df[i])
        df[i+'log']=df[i].apply(lambda x: 0 if (x == 0) else (-np.log(-x) if x < 0 else np.log(x)))
        for j in numcol:
            df[i + "*" + j] = df[i]*df[j]   
            df[i + "/" + j] = df[i]/df[j]
            df[i + "/" + j+'**2'] = (df[i]/df[j])**2
            df[i + "/" + j+'**3'] = (df[i]/df[j])**3


    return df


def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def description_transformator(df):
    df['desc_len'] = df['desc'].apply(lambda x: len(str(x)))
    df["desc"] = df["desc"].apply(
        lambda text: remove_punctuation(text))
    df["desc"] = df["desc"].apply(lambda text: lemmatize_words(text))
    df['near_sea'] = df['desc'].str.contains('sea').astype(int)
    df['near_school'] = df['desc'].str.contains('school').astype(int)
    df['near_kindergarten'] = df['desc'].str.contains(
        'kindergarten').astype(int)
    df['near_park'] = df['desc'].str.contains('park').astype(int)
    df['parking'] = df['desc'].str.contains('parking').astype(int)
    df['new'] = df['desc'].str.contains('new').astype(int)
    df['with_builtin'] = df['desc'].str.contains('builtin').astype(int)
    df['after_renovation'] = df['desc'].str.contains(
        'renovation', 'renovated').astype(int)
    df['large'] = df['desc'].str.contains('large', 'spacious').astype(int)
    df['good'] = df['desc'].str.contains('good', 'excellent').astype(int)
    df.drop('desc', axis=1, inplace=True)
    return df

def collinearity_filter(df):
    '''the function accepts dataset and returns dataset without multicollinear features based on train dataset collinear features only'''
    # Absolute value correlation matrix
    corr_matrix = df.corr().abs().round(2)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Select columns with Pearson's correlations above threshold
    collinear_features = [column for column in upper.columns if any(upper[column] > 0.9)]
    features_filtered = df.drop(columns = collinear_features)
    features_best = []
    features_best.append(features_filtered.columns.tolist()) ## best features list 
    df = df[features_best[0]] # updated dataset
    return df


def feature_extractor(df):
    df = imputer(df)
    df = encoder(df)
    df = one_value_cols_remover(df)
    df = drop_outliers(df)
    df = feature_creator(df)
    df = description_transformator(df)
    return df



def main(df):
    df = feature_extractor(df)
    X = df.drop('price', axis=1).reset_index(drop=True)
    X['n0'] = (X == 0).sum(axis=1) ## one add feature
    y = np.log1p(df.price)
    rf = RandomForestRegressor(max_depth=60, max_samples=0.8, min_samples_leaf=2,
                      min_samples_split=3, n_estimators=600, n_jobs=-1,
                      random_state=2020)
    rf.fit(X, y)
    scores = cross_val_score(rf, X, y, scoring='r2', cv=3, n_jobs=-1)
    s_mean = mean(scores)
    print('Mean R2 before save model: %.3f' % (s_mean))
    # save the model to disk
    filename = 'finalized_model.pkl'
    pickle.dump(rf, open(filename,'wb'))
#     # open model
#     model = pickle.load(open(filename,'rb'))
#     scores = cross_val_score(model, X, y, scoring='r2', cv=3, n_jobs=-1)
#     s_mean = mean(scores)
#     print('Mean R2 after save model: %.3f' % (s_mean))
    return print ('Done')


# model = model(df)
if __name__ == "__main__":
    main(df)