import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler,LabelEncoder




from utilities.my_metrics import mse_test,rmse_test
from utilities.my_polyFeatures import getPolyFeatures
from Regressors.my_regressionModels import MyLinearRegression,MyMLPRegression,MyRidgeRegression,MyLassoRegression

data_root="D:\TJCS\ML&DL\datasets\\boston"
# house-prices-advanced-regression-techniques"
train_file="boston.csv"#"train.csv"
val_file="val.csv"
test_file=None#"test.csv"


df_train = pd.read_csv(data_root+"\\"+train_file) if train_file else None
df_test =  pd.read_csv(data_root+"\\"+test_file) if test_file else None
# print(df_train)
# print(df_test)

df_all = pd.concat([df_train, df_test])  # 给所有数据编码

PRD_LABEL="MEDV" # 预测标签
# PRD_LABEL="SalePrice" # 预测标签

feats=list(df_all.columns)
feats.remove(PRD_LABEL)


df_train_enc =df_all[df_all[PRD_LABEL].notna()].reset_index(drop=True)



# df_train_X= df_train.drop([PRD_LABEL], axis=1)
df_test_enc = df_all[df_all[PRD_LABEL].isna()].reset_index(drop=True)

# print(df_train_enc)
X=df_train_enc[feats]
# print(np.power(X,2))


# X=getPolyFeatures(X,mode='interOnly',max_pow=2)
print(X.shape)


y=df_train_enc[PRD_LABEL]
# y=np.log1p(df_train_enc[PRD_LABEL])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state= 0)

scaler=StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

y_ori=y_test

'''mlp回归'''
dim_input=X.shape[1]
layer_dims=[50,50,50,50]

my_mlpregressor=MyMLPRegression(
    layer_dims=layer_dims,
    dim_input=dim_input
    )

my_mlpregressor.train(
    X_train,
    y_train,
    iterations=5000,
    verbose=False
    )

my_mlpr_pred=my_mlpregressor.predict(X_test)
# print(my_mlpr_pred.shape)


print("my_mlpr R2=",r2_score(y_ori,my_mlpr_pred ))#模型评价, 决定系数
print(rmse_test(my_mlpr_pred.squeeze(),y_ori))

'''lasso回归'''
my_lasso_re=MyLassoRegression()
my_lasso_re.train(
    X_train,
    y_train,
    iterations=5000,
    verbose=False
)
my_lasso_pred=my_lasso_re.predict(X_test)
print("my_lasso R2=",r2_score(y_ori,my_lasso_pred))#模型评价, 决定系数
print(rmse_test(my_lasso_pred,y_ori))


'''ridge回归'''
my_ridge_re=MyRidgeRegression()
my_ridge_re.train(
    X_train,
    y_train,
    iterations=5000,
    verbose=False
)
my_ridge_pred=my_ridge_re.predict(X_test)
print("my_ridge R2=",r2_score(y_ori,my_ridge_pred))#模型评价, 决定系数
print(rmse_test(my_ridge_pred,y_ori))
# exit()



'''线性回归（对比库函数）'''
lr=linear_model.LinearRegression()
lr.fit(X_train,y_train)
lr_pre=lr.predict(X_test)

my_lr=MyLinearRegression()
my_lr.fit(X_train,y_train)
# my_lr.train(X_train,y_train
#             ,learning_rate=0.1,iterations=10000,batch_size=X_train.shape[0],verbose=True)
my_lr_pre = my_lr.predict(X_test)

y_ori=y_test


print("lr R2=",r2_score(y_ori,lr_pre ))#模型评价, 决定系数
print(rmse_test(lr_pre,y_ori))

print("mylr R2=",r2_score(y_ori,my_lr_pre ))#模型评价, 决定系数
print(rmse_test(my_lr_pre,y_ori))

