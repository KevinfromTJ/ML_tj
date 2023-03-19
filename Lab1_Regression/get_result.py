
import pandas as pd
import tqdm
 



data_root="D:\TJCS\ML&DL\datasets\\house-prices-advanced-regression-techniques\\"
data_ask=pd.read_csv(data_root+"test.csv")
data_ans=pd.read_csv(data_root+"true_answer.csv")

data_full_test=pd.merge(data_ask,data_ans,on='Id')

print(data_full_test.shape)
data_full_test.to_csv('test_full.csv', index=False)

# #读取泄露的数据
# data = pd.read_csv(data_root+"AmesHousing.csv")
# data.drop(["PID"],axis=1,inplace=True)
 
# #读取官方提供的数据
# train_data = pd.read_csv(data_root+"train.csv")
# data.columns = train_data.columns
# test_data = pd.read_csv(data_root+"test.csv")
# submission_data = pd.read_csv(data_root+"sample_submission.csv")
 
# print("data:{},train:{},test:{}".format(data.shape,train_data.shape,test_data.shape))
 
# #删除丢失的数据
# miss = test_data.isnull().sum()
# miss = miss[miss > 0]
# data.drop(miss.index,axis=1,inplace=True)
# data.drop(["Electrical"],axis=1,inplace=True)
 
# test_data.dropna(axis=1,inplace=True)
# test_data.drop(["Electrical"],axis=1,inplace=True)
 
 
# for i in  tqdm.trange(0, len(test_data)):
#     for j in range(0, len(data)):
#         for k in range(1, len(test_data.columns)):
#             if test_data.iloc[i,k] == data.iloc[j,k]:
#                 continue
#             else:
#                 break
#         else:
#             submission_data.iloc[i, 1] = data.iloc[j, -1]
#             break
 
# submission_data.to_csv('true_answer.csv', index=False)
