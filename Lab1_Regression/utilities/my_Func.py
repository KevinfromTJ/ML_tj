import numpy as np

def lr_cost_grad_ori(X, y, W, b): 
    """
    多元线性回归的代价和梯度
    
    Inputs:
    - X (N,D)
    - y (N,)
    - W (D,)
    - b (1,)
    
    Returns:
    - cost
    - grad: dict
      - dW
      - db

    """

    grad={}

    # WXPlusB = np.dot(X, W.T) + b
    # cost = np.dot((y - WXPlusB).T, y - WXPlusB) / y.shape[0]
    # w_gradient = -(2 / X.shape[0]) * np.dot((y - WXPlusB).T, X)
    # baise_gradient = -2 * np.dot((y - WXPlusB).T, np.ones(shape=[X.shape[0], 1])) / X.shape[0]


    N = X.shape[0]
    residual=X.dot(W)+b-y
    cost=np.mean((residual)**2)/2
    dW= np.mean(residual.reshape(N,1)*X,axis=0) # 纵轴平均
    db= np.mean(residual)

    grad['dW'],grad['db']=dW,db

    return cost,grad



    

def mlp_last_cost_grad_ori(y_pred, y_true): 
    """
    MLP回归的（最后输出）代价和梯度
    
    Inputs:
    - y_pred (N,)
    - y_true (N,)

    
    Returns:
    - cost
    - dy

    """

    N = y_pred.shape[0]

    residual=y_pred-y_true
    # print(y_pred.shape," ",y_true.shape)

    cost=np.mean(residual**2)/2
    # (y_pred[i]-y_true[i])/N

    dy= residual/N
    # print("dy: ",dy.shape)
  

    return cost,dy



def linear_forward(X,W,b):
    Y=X.dot(W)+b
    cache = (X, W, b) # 反向传播需要

    return Y,cache


def linear_backward(dY, cache):
    X, W, b = cache
    # dY=np.array(dY)
    dX=dY.dot(W.T)
    dW=X.T.dot(dY)
    db=np.sum(dY,axis=0)

    return dX,dW,db

def relu_forward(X):  
    X_relu=X*(X>0) # 技巧令x>0的所有部分为1，其余为0，起到一个“过滤器”的作用
    cache = X
    return X_relu, cache


def relu_backward(dY, cache):

    X = cache
    dX=dY*(X>0)

    return dX


def linear_relu_forward(X,W,b):
    tmp, linear_cache = linear_forward(X, W, b)
    Y, relu_cache = relu_forward(tmp)
    cache = (linear_cache, relu_cache)
    return Y, cache


def linear_relu_backward(dY, cache):
    linear_cache, relu_cache = cache
    dtmp = relu_backward(dY, relu_cache)
    dX, dW, db = linear_backward(dtmp, linear_cache)
    return dX, dW, db
