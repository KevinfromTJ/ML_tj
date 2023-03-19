import numpy as np


def handle_missing(df,PRD_LABEL="SalePrice"):
    feats_cat = df.select_dtypes(include = ["object"]).columns
    feats_num = df.select_dtypes(exclude = ["object"]).columns
    feats_num=feats_num.drop(PRD_LABEL)
    feats_num=feats_num.drop('Id')
    for col in feats_num:
        df[col] = df[col].fillna(df[col].median())
    for col in feats_cat:
        df[col] = df[col].fillna(df[col].mode()[0])         

    return df


def clean_data(df):
    df['totalArea'] = df['GrLivArea'] + df['GarageArea'] + df['TotalBsmtSF']
    tmp = df.drop(df[df['totalArea'] > 8000].index)
    tmp['totalArea'] = np.log1p(tmp['totalArea'])
    tmp = tmp.drop(tmp[tmp['GrLivArea'] > 4000].index)
    tmp = tmp.drop(tmp[(tmp['GrLivArea'] > 3000) & (tmp['MSZoning'] == 'RM')].index)
    tmp = tmp.drop(tmp[(tmp['GrLivArea'] > 3000) & (tmp['MSZoning'] == 'RH')].index)
    tmp = tmp.drop(tmp[tmp['LotFrontage'] > 300].index)
    tmp = tmp.drop(tmp[tmp['LotArea'] > 100000].index)
    tmp['LotArea'] = np.log1p(tmp['LotArea'])
    tmp = tmp.drop('Street', axis=1)
    tmp = tmp.drop('Alley', axis=1)
    tmp = tmp.drop('Utilities', axis=1)
    tmp = tmp.drop(tmp[(tmp['OverallCond'] ==2 )&(tmp['SalePrice'] > 300000 )].index)
    tmp['age']  = 2010 - tmp['YearBuilt']
    tmp['TotalBsmtSF'] = np.log1p(tmp['TotalBsmtSF'])
    tmp['1stFlrSF'] = np.log1p(tmp['1stFlrSF'])
    tmp['GrLivArea'] = np.log1p(tmp['GrLivArea'])
    tmp['TotRmsAbvGrd'] = np.log1p(tmp['TotRmsAbvGrd'])
    tmp = tmp.drop('FireplaceQu',axis=1)
    tmp['GarageArea'] = np.log1p(tmp['GarageArea'])
    tmp['OpenPorchSF'] = np.log1p(tmp['OpenPorchSF'])
    tmp = tmp.drop('3SsnPorch',axis=1)
    tmp['ScreenPorch'] = np.log1p(tmp['ScreenPorch'])
    tmp = tmp.drop('PoolArea',axis=1)
    tmp = tmp.drop('PoolQC',axis=1)
    tmp['MasVnrArea'] = np.log1p(tmp['MasVnrArea'])
    return tmp


'''有些数据以类别形式给出，但实际上具有大小次序关系（比如好、较好、一般、坏;小、中、大）'''
def handle_cat2num(df):
    df = df.replace({
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                    }
                     )
    return df
    


'''有些数据以数值形式给出，但实际上没有次序关系（比如月份）'''
def handle_num2cat(df):
    df = df.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })
    return df


