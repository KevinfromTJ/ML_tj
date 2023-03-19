import numpy as np

from Lab1_Regression.utilities.my_Func import *

class MyLinearRegression(object):
    def __init__(self):
        self.W = None
        self.b = None
        self.Theta = None # W与b拼接


    def fit(self,X,y):
        """
        正规方程直接解得所有theta

        Inputs:
        - X: (N, D) 
        - y: (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        

        Returns:
        """
        bX = np.hstack([np.ones((X.shape[0],1)), X]) #  [b(theta0),X]
        self.Theta = np.linalg.inv(bX.T.dot(bX)).dot(bX.T).dot(y) #(𝑋^𝑇*𝑋)^(−1)*𝑋^𝑇*𝑦
        self.b = self.Theta[0]
        self.W = self.Theta[1:]

    

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        iterations=100,
        batch_size=200,
        verbose=False,
    ):
        """
        训练的方式更新参数

        Inputs:
        - X: (N, D) 
        - y: (N,) 
        

        Returns:
        """
        data_num, feature_dim = X.shape


        if self.W is None:
            self.W = 100*np.random.randn(feature_dim) 
        if self.b is None:
            self.b = 100*np.random.randn(1) 
        
        X=np.array(X)
        y=np.array(y)

        # sgd 优化

        for it in range(iterations):

            train_indices=np.random.choice(range(data_num),batch_size,replace=True)
            X_batch= X[train_indices]
            # print(y.shape)
            y_batch=y [train_indices]
            # print('X_batch.shape: ',X_batch.shape,' ','y_batch.shape: ',y_batch.shape)



     
            cost,grad=lr_cost_grad_ori(X_batch,y_batch,self.W,self.b)


            self.W-=learning_rate*grad['dW']
            self.b-=learning_rate*grad['db']
            


            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, iterations, cost))

        

    def predict(self, X):
        """
        根据训练好或直接计算出的最优W,b预测结果

        Inputs:
        - X: (N, D) 

        Returns:
        - y_pred
        """
      
        y_pred=X.dot(self.W)+self.b
        
        return y_pred

    # def get_cost_grad(self, X_batch, y_batch, reg):
    #     """
    #     TODO: 可以用子类覆写

        
    #     """
    #     cost,grad=lr_cost_grad_ori(X_batch,y_batch,self.W,self.b)
    #     return cost,grad

    #     pass




'''岭回归'''
class MyRidgeRegression(object):
    def __init__(self,lam=0.01):
        self.W = None
        self.b = None
        self.lam =lam  # L2正则化系数
    
    def fit(self,X,y):
        """
        正规方程直接解得所有theta

        Inputs:
        - X: (N, D) 
        - y: (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        

        Returns:
        """
        X = np.hstack([np.ones((X.shape[0],1)), X]) #  [b(theta0),X]
        N,D = X.shape
        dn_lam=D/N*self.lam
        XTX = np.matmul(X.T , X)
        XTX_I_inv = np.linalg.inv(XTX/D+dn_lam/D*np.eye(D))

        Theta = np.matmul(XTX_I_inv,np.matmul(X.T,y)).reshape(-1)/N
        self.b = Theta[0]
        self.W = Theta[1:]

    def train(
            self,
            X,
            y,
            mode='',
            config=None,
            # learning_rate=1e-3,
            iterations=100,
            batch_size=200,
            verbose=False,
    ):
        """
        训练的方式更新参数

        Inputs:
        - X: (N, D)
        - y: (N,)


        Returns:
        """
        data_num, feature_dim = X.shape

        if self.W is None:
            self.W = 100 * np.random.randn(feature_dim)
        if self.b is None:
            self.b = 100 * np.random.randn(1)

        X = np.array(X)
        y = np.array(y)

        # sgd 优化

        for it in range(iterations):

            train_indices = np.random.choice(range(data_num), batch_size, replace=True)

            X_batch = X[train_indices]
            # print(y.shape)
            y_batch = y[train_indices]
            # print('X_batch.shape: ',X_batch.shape,' ','y_batch.shape: ',y_batch.shape)

            cost, grad = lr_cost_grad_ori(X_batch, y_batch, self.W, self.b)
            # print(grad['dW'].shape[0])

            cost += np.sum(self.W**2)*self.lam
            N=grad['dW'].shape[0]

            grad['dW']=grad['dW']+self.lam/N*self.W
            grad['db']=grad['db']+self.lam/N*self.b


            Theta =np.hstack([self.b,self.W])
            dTheta =np.hstack([grad['db'],grad['dW']])

            Theta,config=adam(Theta,dTheta,config)
                
            self.b = Theta[0]
            self.W = Theta[1:]

                
            # self.W -= learning_rate * grad['dW'] + self.lam * learning_rate / grad['dW'].shape[0]*self.W
            # self.b -= learning_rate * grad['db'] + self.lam * learning_rate / grad['dW'].shape[0]*self.b
            # # cost +=
            if verbose and it % 2000 == 0:
                print("iteration %d / %d: loss %f" % (it, iterations, cost))

    def predict(self, X):
        """
        根据训练好或直接计算出的最优W,b预测结果

        Inputs:
        - X: (N, D)

        Returns:
        - y_pred
        """

        y_pred = X.dot(self.W) + self.b

        return y_pred   



'''lasso回归'''
class MyLassoRegression(object):
    def __init__(self,lam=0.01):
            self.W = None
            self.b = None
            self.lam = lam  # 参数

    def train(
            self,
            X,
            y,
            learning_rate=1e-3,
            iterations=100,
            batch_size=200,
            verbose=False,
    ):
        """
        训练的方式更新参数

        Inputs:
        - X: (N, D)
        - y: (N,)


        Returns:
        """
        data_num, feature_dim = X.shape

        if self.W is None:
            self.W = 100 * np.random.randn(feature_dim)
        if self.b is None:
            self.b = 100 * np.random.randn(1)

        X = np.array(X)
        y = np.array(y)

        # sgd 优化

        for it in range(iterations):

            train_indices = np.random.choice(range(data_num), batch_size, replace=True)

            X_batch = X[train_indices]
            # print(y.shape)
            y_batch = y[train_indices]
            # print('X_batch.shape: ',X_batch.shape,' ','y_batch.shape: ',y_batch.shape)

            cost, grad = lr_cost_grad_ori(X_batch, y_batch, self.W, self.b)
            # print(grad['dW'].shape[0])

            cost += np.sum(abs(self.W))*self.lam


            RW = self.W/abs(self.W)


            self.W -= learning_rate * grad['dW'] + self.lam * learning_rate * RW
            self.b -= learning_rate * grad['db']

            if verbose and it % 2000 == 0:
                print("iteration %d / %d: loss %f" % (it, iterations, cost))

    def predict(self, X):
        """
        根据训练好或直接计算出的最优W,b预测结果

        Inputs:
        - X: (N, D)

        Returns:
        - y_pred
        """

        y_pred = X.dot(self.W) + self.b

        return y_pred




'''Elastic Net(就是上边俩加一起，一般不如单独Lasso)'''


'''MLP回归'''
class MyMLPRegression(object):
    """
    结构：
    {linear - [batch/layer norm] - relu - [dropout]} *- linear 

    若干个[线性+可选的正则化+relu激活+可选的dropout]层，最后一层线性

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """
    def __init__(
        self,
        layer_dims, # 总层数-1，即带relu部分的层数
        dim_input,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.01,
        weight_scale=1e-2,
    ):

        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.layer_num = 1 + len(layer_dims)
        self.params = {}

        '''默认scale=1.0 loc(中心位置，即均值)=0.0'''
        dim_in=dim_input
        for i in range(1,self.layer_num):
            self.params['W'+str(i)] = np.random.normal(scale=weight_scale, size=(dim_in, layer_dims[i-1]))
            self.params['b'+str(i)] = np.zeros(layer_dims[i-1])
            if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
                self.params['gamma'+str(i)] = np.ones(layer_dims[i-1])
                self.params['beta'+str(i)] = np.zeros(layer_dims[i-1])
            dim_in=layer_dims[i-1]
            
        self.params['W'+str(self.layer_num)] = np.random.normal(scale=weight_scale, size=(dim_in, 1))
        self.params['b'+str(self.layer_num)] = np.zeros(1)
        '''【注意】 按要求，最后一个linear和softmax之间不需要'''

    def cost_grad_MLP(
        self, 
        X, 
        y):

        forward_nodropout_cache,dropout_cache=[],[]  #存放每一层返回的
        res=None
        for i in range(1,self.layer_num):
            if self.normalization == None:
                res=linear_relu_forward(X,self.params['W'+str(i)],self.params['b'+str(i)])
                
            # elif self.normalization =="batchnorm":
            #     res=linear_bn_relu_forward(X,
            #     self.params['W'+str(i)],self.params['b'+str(i)],
            #     self.params['gamma'+str(i)],self.params['beta'+str(i)],self.bn_params[i-1])

            # elif self.normalization =="layernorm": # 都用bn_param
            #     res=linear_ln_relu_forward(X,
            #     self.params['W'+str(i)],self.params['b'+str(i)],
            #     self.params['gamma'+str(i)],self.params['beta'+str(i)],self.bn_params[i-1])

            forward_nodropout_cache.append(res[1])
            X=res[0]
            
            # if self.use_dropout:
            #     res=dropout_forward(X,self.dropout_param)# dropout_param和bn不一样，每一层一样，只是得到mask不一样
            #     dropout_cache.append(res[1])
            #     X=res[0]
            

        # print('\n')
        y_pred,cache_linear_last=linear_forward(X,self.params['W'+str(self.layer_num)],self.params['b'+str(self.layer_num)])
        

        y_pred=y_pred.reshape(-1,1)
        y=y.reshape(-1,1)

        cost,dy= mlp_last_cost_grad_ori(y_pred,y)
        # print(dy.shape)

        W=list(self.params['W'+str(i)] for i in range(1,self.layer_num+1))
        # print(len(W))
        cost +=0.5*self.reg*(sum(np.sum(W[i]*W[i]) for i in range(0,len(W))))

        '''下边是反向传播——更新梯度'''
        grads={}


        dX_cur,dW_cur,db_cur=linear_backward(dy,cache_linear_last)
        grads['W'+str(self.layer_num)]=dW_cur+self.reg*W[self.layer_num-1]
        grads['b'+str(self.layer_num)]=db_cur

        for i in range(self.layer_num-1,0,-1):
            # if self.use_dropout:
            #     dX_cur= dropout_backward(dX_cur,dropout_cache[i-1])

            if self.normalization == None:
                dX_cur,dW_cur,db_cur=linear_relu_backward(dX_cur, forward_nodropout_cache[i-1])
                
            # elif self.normalization =="batchnorm":
            #     dX_cur,dW_cur,db_cur,dgamma_cur,dbeta_cur=affine_bn_relu_backward(dX_cur, forward_nodropout_cache[i-1])
            #     grads['gamma'+str(i)]=dgamma_cur
            #     grads['beta'+str(i)]=dbeta_cur

            # elif self.normalization =="layernorm":
            #     dX_cur,dW_cur,db_cur,dgamma_cur,dbeta_cur=affine_ln_relu_backward(dX_cur, forward_nodropout_cache[i-1])
            #     grads['gamma'+str(i)]=dgamma_cur
            #     grads['beta'+str(i)]=dbeta_cur

            grads['W'+str(i)]=dW_cur+self.reg*W[i-1]
            grads['b'+str(i)]=db_cur
    
        return cost,grads

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        iterations=100,
        batch_size=200,
        verbose=False,
    ):
        """
        sgd训练的方式更新参数

        Inputs:
        - X: (N, D) 
        - y: (N,) 
        
        """
        data_num, feature_dim = X.shape



        
        X=np.array(X)
        y=np.array(y)

        # sgd 优化

        for it in range(iterations):

            train_indices=np.random.choice(range(data_num),batch_size,replace=True)
            X_batch= X[train_indices]
            # print(y.shape)
            y_batch=y [train_indices]
            # print('X_batch.shape: ',X_batch.shape,' ','y_batch.shape: ',y_batch.shape)



     
            cost,grads=self.cost_grad_MLP(X_batch,y_batch)

            for i in range(1,self.layer_num+1):
                # print(self.params['W'+str(i)].shape,'  ',grads['W'+str(i)].shape)
                self.params['W'+str(i)]-=learning_rate*grads['W'+str(i)]
                self.params['b'+str(i)]-=learning_rate*grads['b'+str(i)]
            


            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, iterations, cost))



    def predict(self, X_test):
        X=X_test
        for i in range(1,self.layer_num):
            if self.normalization == None:
                X,_=linear_relu_forward(X,self.params['W'+str(i)],self.params['b'+str(i)])
                
            # elif self.normalization =="batchnorm":
            #     res=affine_bn_relu_forward(X,
            #     self.params['W'+str(i)],self.params['b'+str(i)],
            #     self.params['gamma'+str(i)],self.params['beta'+str(i)],self.bn_params[i-1])

            # elif self.normalization =="layernorm": # 都用bn_param
            #     res=affine_ln_relu_forward(X,
            #     self.params['W'+str(i)],self.params['b'+str(i)],
            #     self.params['gamma'+str(i)],self.params['beta'+str(i)],self.bn_params[i-1])

            
            # if self.use_dropout:
            #     res=dropout_forward(X,self.dropout_param)# dropout_param和bn不一样，每一层一样，只是得到mask不一样
            #     dropout_cache.append(res[1])
            #     X=res[0]
            

        # print('\n')
        y_pred,_=linear_forward(X,self.params['W'+str(self.layer_num)],self.params['b'+str(self.layer_num)])


        return y_pred

    






    