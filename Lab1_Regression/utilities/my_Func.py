import numpy as np

'''代价与梯度'''
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


'''神经网络各层'''
def dropout_forward(x, mode,p_valid):
    """

    Inputs:
    - x: (N,D)
    - mode: 训练("train")或测试("test"——不用drop)
    - p_valid:某神经元有效的概率（即有效的比例）

    Returns:
    - y: (N,D)
    - cache: 
    """


    mask,y = None, None

    if mode == "train":
        # 注意rand要求传入不定参数的各维度，所以对于shape返回的元组需要加*解掉
        mask=np.random.rand(*x.shape)<p_valid # 小于的被丢弃
        '''注意除p，毕竟只有p/1的神经元激活了'''
        y =x *mask/ p_valid

    elif mode == "test":
        y =x # 测试时全部神经元就全部用上了 

    cache=(p_valid,mask)
    return y, cache

def dropout_backward(dy, cache):
    """

    Inputs:
    - dy: Upstream derivatives, of any shape
    - cache: 
        - p_valid
        - mask
    Returns:
    - dx

    """

    p_valid,mask=cache

    dx=dy/p_valid
    dx[mask<p_valid]=0 # 不激活的神经元没有梯度，激活的梯度就是上一层反向的

    return dx


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


'''学习算法'''
def adam(w, dw, config=None):
    """
    Adam 算法

    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9) # 0.9
    config.setdefault("beta2", 0.999) # 0.99
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    
    
    m,v,t=config['m'],config['v'],config['t']
    lr,beta1,beta2,eps=config['learning_rate'],config['beta1'],config['beta2'],config['epsilon']

    # print(lr)

    config['t'] = t = t + 1                             
    config['m'] = m = beta1 * m + (1 - beta1) * dw      
    mt = m / (1 - beta1**t)                             
    config['v'] = v = beta2 * v + (1 - beta2) * (dw**2) 
    vt = v / (1 - beta2**t)                             
    next_w = w - lr * mt / (np.sqrt(vt) + eps)          
    

    return next_w, config