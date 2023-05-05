'''
图片数据的预处理
'''


from cv2 import normalize,PCAProject


# def preprocess_svm():
#     '''
#     SVM的预处理
#     :return:
#     '''
#     # 读取数据
#     train_data, train_label, test_data, test_label = read_data()
#     # 数据归一化
#     train_data = normalize(train_data)
#     test_data = normalize(test_data)
#     # 数据降维
#     train_data = pca(train_data, 0.9)
#     test_data = pca(test_data, 0.9)
#     # 数据保存
#     save_data(train_data, train_label, test_data, test_label)

def preprocess_rf():
    '''
    RF的预处理
    :return:
    '''
    # 读取数据
    train_data, train_label, test_data, test_label = read_data()
    # 数据归一化
    train_data = normalize(train_data)
    test_data = normalize(test_data)
    # 数据保存
    save_data(train_data, train_label, test_data, test_label)

# 神经网络方法对图片数据的预处理
def preprocess_nn():
    '''
    NN的预处理
    :return:
    '''
    

    