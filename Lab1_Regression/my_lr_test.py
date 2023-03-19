import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import  RidgeCV, LassoCV, ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler,LabelEncoder



from utilities.my_metrics import mse_test,rmse_test
from utilities.my_polyFeatures import getPolyFeatures_np,getPolyFeatures_pd
from Regressors.my_regressionModels import MyLinearRegression,MyMLPRegression,MyRidgeRegression,MyLassoRegression
from data_preprocess import *
from utilities.my_visualize import draw_sns

# data_root="D:\TJCS\ML&DL\datasets\\boston"
# train_file="boston.csv"
# test_file=None
# PRD_LABEL="MEDV" # 预测标签
data_root="D:\TJCS\ML&DL\datasets\\house-prices-advanced-regression-techniques"
train_file="train.csv"
val_file="val.csv"
test_file="test.csv"
PRD_LABEL="SalePrice" # 预测标签

df_train = pd.read_csv(data_root+"\\"+train_file) if train_file else None

# draw_sns(df_train,0.25)
# exit()

df_test =  pd.read_csv(data_root+"\\"+test_file) if test_file else None
# print(df_train)
# print(df_test)

df_all = pd.concat([df_train, df_test]).reset_index(drop=True)  # 因为要给所有数据编码

feats_cat = df_all.select_dtypes(include = ["object"]).columns
feats_num = df_all.select_dtypes(exclude = ["object"]).columns
feats_num=feats_num.drop(PRD_LABEL)
feats_num=feats_num.drop('Id')


print("数值特征有 : " + str(len(feats_num)))
print("离散特征有 : " + str(len(feats_cat )))
# exit()

# print(df_all)
# feats=list(df_all.columns)
# feats.remove(PRD_LABEL)

'''数据预处理'''
df_all=clean_data(df_all)
df_all=handle_missing(df_all)

df_all=handle_num2cat(df_all)
df_all=handle_cat2num(df_all)

print("各种数据有： \n",df_all.dtypes.value_counts()) # 统计列数据类型

feats_cat = df_all.select_dtypes(include = ["object"]).columns
feats_num = df_all.select_dtypes(exclude = ["object"]).columns
feats_num=feats_num.drop(PRD_LABEL)
feats_num=feats_num.drop('Id')

print("数值特征有 : " + str(len(feats_num)))
print("离散特征有 : " + str(len(feats_cat )))
df_num = df_all[feats_num]
df_cat = df_all[feats_cat]


df_num= df_num.fillna(df_num.median()) # 对（真）数值特征填充缺失
# print(df_num.isnull().sum())
df_num=getPolyFeatures_pd(df_num,mode='both',max_pow=2) # 对（真）数值特征做多项式处理【这一步】
df_cat = pd.get_dummies(df_cat) # 对（真）类别特征独热编码

df_all= pd.concat((df_num, df_cat,df_all[PRD_LABEL]), axis = 1)# 经过自定义函数后变成了np数组，要转回来
# print(df_all)
# print(df_all.columns.has_duplicates)
# print(df_all[PRD_LABEL].notna())


'''分离预测和测试'''
df_train_enc =df_all[df_all[PRD_LABEL].notna()].reset_index(drop=True)
print(df_train_enc.shape)


# df_train_X= df_train.drop([PRD_LABEL], axis=1)
df_test_enc = df_all[df_all[PRD_LABEL].isna()].reset_index(drop=True)

# print(df_train_enc)
X_train_test=df_train_enc
y_train_test=df_train_enc[PRD_LABEL]

print(X_train_test.shape)
print(y_train_test.shape)

# print(np.power(X,2))

# Find most important features relative to target
# print("Find most important features relative to target")
corr = df_train_enc.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
# print(corr.SalePrice)

# print(X_train_val.shape)


corr_sp=corr.SalePrice[corr.SalePrice.notnull()]
corr_sp=corr_sp[ abs(corr_sp)>0.5 ]
# print(corr_sp)



X_train_test=X_train_test[pd.DataFrame(corr_sp).index.drop(PRD_LABEL)]
print(X_train_test.shape)
# exit()

y_train_test=np.log1p(y_train_test)
'''处理好的特征选取'''

# exit()
feats_cat = X_train_test.select_dtypes(include = ["object"]).columns
feats_num = X_train_test.select_dtypes(exclude = ["object"]).columns

X_train,X_test,y_train,y_test = train_test_split(X_train_test,y_train_test,test_size = 0.3,random_state= 0)


scaler=StandardScaler().fit(X_train.loc[:,feats_num])
X_train.loc[:,feats_num]=scaler.transform(X_train.loc[:,feats_num])
X_test.loc[:,feats_num]=scaler.transform(X_test.loc[:,feats_num])


# print ( X_train.isnull().values.any())
# print ( X_test.isnull().values.any())
# print ( y_train.isnull().values.any())
# print ( y_test.isnull().values.any())

y_ori=y_test

# answer_file="test_full.csv"
# df_answer =  pd.read_csv(data_root+"\\"+answer_file)
# y_test_ori=np.log1p(df_answer['SalePrice'])
# df_test_enc=df_test_enc.drop(PRD_LABEL)

'''
下边就是具体的算法训练了
'''

optim_mode='adam'
# optim_mode='sgd'


'''mlp回归'''
if 1:
    dim_input=X_train.shape[1]                  # 参数设定
    # layer_dims=[200,150,100,50]
    layer_dims=[100,100,50,20]
    mlp_cfg={}
    mlp_cfg['learning_rate']=0.01
    # mlp_cfg["beta1"]=0.9 # 0.9
    mlp_cfg["beta2"]=0.99 # 0.99
    # mlp_cfg["epsilon"]=1e-7 
    my_mlpregressor=MyMLPRegression(            # 初始化回归模型
        layer_dims=layer_dims,
        dim_input=dim_input,
        dropout_keep_ratio=1.0,
        weight_scale=0.1,
        )

    my_mlpregressor.train(                       # 训练
        X_train,
        y_train,
        mode=optim_mode,
        batch_size=128,
        config=mlp_cfg,
        iterations=10000,
        verbose=True,
        output_inv=10000//10,
        )

    my_mlpr_pred=my_mlpregressor.predict(X_test) # 测试


    print(my_mlpr_pred.shape)
    print("my_mlpr R2=",r2_score(y_ori,my_mlpr_pred ))#模型评价, 决定系数
    print(rmse_test(my_mlpr_pred.squeeze(),y_ori))
# exit()
'''lasso回归'''
if 1:
    lasso_cfg={}
    lasso_cfg['learning_rate']=0.005
    my_lasso_re=MyLassoRegression(lam=0.001)
    my_lasso_re.train(
        X_train,
        y_train,
        mode=optim_mode,
        batch_size=128,
        config=lasso_cfg,
        iterations=10000,
        verbose=True,
        output_inv=10000//10,
    )
    my_lasso_pred=my_lasso_re.predict(X_test)
    print("my_lasso R2=",r2_score(y_ori,my_lasso_pred))#模型评价, 决定系数
    print(rmse_test(my_lasso_pred,y_ori))


    lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                            0.3, 0.6, 1], 
                    max_iter = 50000, cv = 10)
    lasso.fit(X_train, y_train)
    alpha = lasso.alpha_
    print("Best alpha :", alpha)
    lasso_pred=lasso.predict(X_test)
    print("lasso R2=",r2_score(y_ori,lasso_pred))#模型评价, 决定系数
    print(rmse_test(lasso_pred,y_ori))

'''ridge回归'''
if 1:
    ridge_cfg={}
    ridge_cfg['learning_rate']=0.005
    my_ridge_re=MyRidgeRegression(lam=0.03)
    my_ridge_re.train(
        X_train,
        y_train,
        mode=optim_mode,
        batch_size=128,
        config=ridge_cfg,
        iterations=10000,
        verbose=True,
        output_inv=10000//10
    )
    # my_ridge_re.fit(X_train,y_train)
    my_ridge_pred=my_ridge_re.predict(X_test)
    print("my_ridge R2=",r2_score(y_ori,my_ridge_pred))#模型评价, 决定系数
    print(rmse_test(my_ridge_pred,y_ori))
    # exit()



'''线性回归（对比库函数）'''
if 1:
    lr=LinearRegression()
    lr.fit(X_train,y_train)
    lr_pre=lr.predict(X_test)
    print("lr R2=",r2_score(y_ori,lr_pre ))#模型评价, 决定系数
    print(rmse_test(lr_pre,y_ori))

    my_lr=MyLinearRegression()
    lr_cfg={}
    lr_cfg['learning_rate']=0.005
    # my_lr.fit(X_train,y_train)
    my_lr.train(
        X_train,
        y_train,
        mode=optim_mode,
        config=lr_cfg,
        iterations=20000,
        batch_size=X_train.shape[0]//10,
        verbose=True,
        output_inv=20000//10
        )
    my_lr_pre = my_lr.predict(X_test)
    print("mylr R2=",r2_score(y_ori,my_lr_pre ))#模型评价, 决定系数
    print(rmse_test(my_lr_pre,y_ori))






