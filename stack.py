# stack是机器集成学习中一种使用较多的方法，本人主要使用LightGBM实现模型结果的集成。LightGBM支持自定义训练损失函数和评价函数(验证集，选择模型，不要求可导).


import lightgbm as lgb
from sklearn.metrics import mean_squared_error, cohen_kappa_score, accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
from collections import Counter
import os
from itertools import product


root = 'prob_896/'
bx = ['b'+str(i) for i in range(5)]
others = ['r34', 'v3plus', 'v4']
names = bx +  others
in_reg = True

# 1. 获得单模型的预测结果，如果是回归模型，结果介绍回归值；如果是分类模型，结果是分类结果(概率向量可能也可以)
train_label = np.load('{}_label.npy'.format(root+'b0'+'_train'))
valid_label = np.load('{}_label.npy'.format(root+'b0'+'_valid'))
test_label = np.load('{}_label.npy'.format(root+'b0'+'_test'))

train_list = []
valid_list = []
test_list = []
for bx in names:
    modelName = root + bx
    if os.path.exists(modelName):
        print(modelName)
    train_data = np.load('{}_fea.npy'.format(modelName+'_train'))
    valid_data = np.load('{}_fea.npy'.format(modelName+'_valid'))
    test_data = np.load('{}_fea.npy'.format(modelName+'_test'))
    if in_reg:
        train_cla = np.argmax(train_data[:,0:5], 1).reshape(-1,1)
        valid_cla = np.argmax(valid_data[:,0:5], 1).reshape(-1,1)
        test_cla = np.argmax(test_data[:,0:5], 1).reshape(-1,1)
        # print(train_data.shape)
        # print(train_data[:,-2:-1].shape)
        train_pred = np.concatenate([train_cla, train_data[:,-2:-1]], 1)
        valid_pred = np.concatenate([valid_cla, valid_data[:,-2:-1]], 1)
        test_pred = np.concatenate([test_cla, test_data[:,-2:-1]], 1)
    else:
        train_pred = np.argmax(train_data[:,0:5], 1).reshape(-1,1)
        valid_pred= np.argmax(valid_data[:,0:5], 1).reshape(-1,1)
        test_pred = np.argmax(test_data[:,0:5], 1).reshape(-1,1)

    train_list.append(train_pred)
    valid_list.append(valid_pred)
    test_list.append(test_pred)
      
# 2. 单模型的预测结果拼接在一起，作为次级学习器的特征，特征大小(#sample,#model)
meta_train_data = np.concatenate(train_list, 1)
meta_valid_data = np.concatenate(valid_list, 1)
meta_test_data = np.concatenate(test_list, 1)

# 3. 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(meta_train_data, train_label)
lgb_eval = lgb.Dataset(meta_valid_data, valid_label, reference=lgb_train)
lgb_test = lgb.Dataset(meta_test_data, test_label, reference=lgb_train)


# 3. 自定义objective和metric
from scipy.misc import derivative
def myMulClass(y_pred, dtrain):
    # y_pred size: (#sample * #class, )
    num_class = 5
    a = 0.2
    g = 2
    y_true = dtrain.get_label()
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1,num_class, order='F')
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        loss = -( t*np.log(p)+(1-t)*np.log(1-p))
        loss = -(a * y_true + (1 - a) * (1 - y_true)) * ((1 - (y_true * p + (1 - y_true) * (1 - p))) ** g) * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        return loss
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    # flatten in column-major (Fortran-style) order
    return grad.flatten('F'), hess.flatten('F')

def custom_asymmetric_objective(y_pred, dtrain):
    y_true = dtrain.get_label()
    weights = np.array([1, 2.0, 1.0, 4.0, 4.0])
    weights = weights[np.int16(y_true)]
    residual = (y_pred - y_true).astype("float")
    grad = np.where(residual>0, 2.0*residual, 2.7*residual)
    hess = np.where(residual>0, 2.0, 2.7)
    return grad, hess

def custom_asymmetric_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual**2)*10.0, residual**2)
    return "custom_asymmetric_eval", np.mean(loss), False

def myeval(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred_clip = np.clip(y_pred, 0, 4)
    y_pred_int = np.int32(np.round(y_pred_clip))
    loss = cohen_kappa_score(y_true, y_pred_int, labels=list(range(5)), weights='quadratic')
    return "qwk", loss, True

# 训练、测试
def gbm():
    # 2,2,0.1,0.1,4
    mds = [2]
    nls = [2]
    ffs = [0.1]
    bf1s = [0.1]
    bf2s = [4]
    best_qwk = -1
    for md, nl, ff, bf1, bf2 in product(mds, nls, ffs, bf1s, bf2s):
        # 回归器
        params = {
            'seed': 2020,
            'task': 'train',
            'nthread': 6,
            'objective': 'regression_l2',  # 训练 目标函数
            'metric': 'mse',  # 测试 评估函数
            'max_depth': md,
            'lambda_l2': 0.0,
            'lambda_l1': 0.0,
            'num_leaves': nl,  # 叶子节点数
            'learning_rate': 0.03,  # 学习速率
            'feature_fraction': ff,  # 建树的特征选择比例
            'bagging_fraction': bf1,  # 建树的样本采样比例
            'bagging_freq': bf2,  # k 意味着每 k 次迭代执行bagging
            'verbose': 0  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        
        # 分类器
        # params = {
        #     'task': 'train',
        #     'objective': 'multiclass',  # 目标函数
        #     'num_class': 5,
        #     'metric': 'multi_logloss',  # 评估函数
        #     'max_depth': md,
        #     'lambda_l2': 0.0,
        #     'lambda_l1': 0.0,
        #     'num_leaves': nl,  # 叶子节点数
        #     'learning_rate': 0.05,  # 学习速率
        #     'feature_fraction': ff,  # 建树的特征选择比例
        #     'bagging_fraction': bf1,  # 建树的样本采样比例
        #     'bagging_freq': bf2,  # k 意味着每 k 次迭代执行bagging
        #     'verbose': 0  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        # }

        # 训练 cv and train
        # fobj为训练的目标损失函数，feval为评价函数，默认值fobj=None, feval=None。当使用feval时，params中的'metric'设置为'None'.
        gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_eval, fobj=custom_asymmetric_objective, feval=None, early_stopping_rounds=20)
        # learning_rates=lambda iter: 0.01 * (1 ** iter), callbacks=[lgb.reset_parameter(bagging_fraction=[0.8] * 400 + [0.7] * 400)]

        # 预测
        names = ['train', 'valid', 'test']
        datas = [meta_train_data, meta_valid_data, meta_test_data]
        labels = [train_label, valid_label, test_label]
        for data,label,name in zip(datas, labels, names):
            y_prob = gbm.predict(data, num_iteration=gbm.best_iteration)
            if params['objective'] in ['multiclass', 'multi_logloss']:
                y_pred = np.argmax(y_prob, 1)
            else:
                y_pred = np.clip(y_prob, 0, 4)
                y_pred = np.int32(np.round(y_pred))

            acc = accuracy_score(y_pred, label)
            qwk = cohen_kappa_score(label, y_pred, labels=list(range(5)), weights='quadratic')
            print('{:<6} acc:{:.4f}, qwk:{:.4f}'.format(name, acc, qwk))
            cm = confusion_matrix(label, y_pred)
        print(cm)

        if qwk >= best_qwk:
            best_qwk = qwk
            best_param = [md, nl, ff, bf1, bf2]
    print('*'*40)
    print(best_param)
    print(best_qwk)

############################################################
start = time.time()

print('*'*30)
gbm()

end = time.time()
print('time:{:.4f}'.format(end-start))
