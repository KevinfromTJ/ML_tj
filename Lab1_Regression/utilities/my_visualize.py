from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def draw_sns(df_train,thres_2zero=0.3):
    #提取数值特征
    df_train_num = df_train.select_dtypes(exclude=["object"])
    #查看数值特征
    df_train_num.head()
    #画出训练集中每个数值变量的分布直方图
    fig_ = df_train_num.hist(figsize=(25, 30), bins=50, color="darkorange", edgecolor="black", xlabelsize=8, ylabelsize=8)


    #定义热图参数
    pd.options.display.float_format="{:,.2f}".format

    #定义相关矩阵
    corr_matrix=df_train_num.corr(method="pearson")

    #为了方便观察，将相关系数绝对值小于thres_2zero的系数用0取代
    corr_matrix[(corr_matrix<thres_2zero)&(corr_matrix>-thres_2zero)]=0

    #修饰热图的上半部分,，使得重复的部分变为空白
    mask=np.triu(np.ones_like(corr_matrix,dtype=bool))

    #颜色
    cmap="viridis"
    #大小
    plt.figure(figsize=(20,9))

    #画图
    sns.heatmap(corr_matrix,
                mask=mask,
                vmax=1.0,vmin=-1.0
                ,linewidths=0.1
                ,annot_kws={"size":9,"color":"black"}
                ,square=True
                ,cmap=cmap
                ,annot=True)

    plt.show()
