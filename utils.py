import numpy as np
import pandas as pd
import sys
path = '/Users/apple/Documents/Python/TianChi/Car_Price'
sys.path.append(path)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# [nan_predicter]: 使用 RandomForestClassifier 来填充['bodyType', 'fuelType', 'gearbox']
def nan_predicter(data, data_predict, verbose=True):
    # 使用 data 上的一个完整数据集作为训练
    full_set = data[(data.bodyType.notnull())&
        (data.fuelType.notnull())&
        (data.gearbox.notnull())]
    feature = ['model', 'brand', 'power','used_time']
    predict_feature = ['bodyType', 'fuelType', 'gearbox']
    X = full_set[feature]
    y = full_set[predict_feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    if verbose:
        print('Fitting NA data...')
    tree = RandomForestClassifier(n_estimators=80, n_jobs=-1)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)
    if verbose:
        print('Predit Accuracy:')
        for column in y_pred.columns:
            print('  ',column, round(sum(y_pred[column]==y_test[column])/len(y_pred), 4), '%')
            
    # 对 data 和 data_predict 上有缺失值的数据进行预测
    if verbose:
        print('Preditting NA data...')
    X_data = data[(data.bodyType.isna()) |
                  (data.fuelType.isna()) |
                  (data.gearbox.isna())][feature]
    X_data_predict = data_predict[(data_predict.bodyType.isna()) |
                                  (data_predict.fuelType.isna()) |
                                  (data_predict.gearbox.isna())][feature]
    Y_data = tree.predict(X_data)
    Y_data_predict = tree.predict(X_data_predict)
    Y_data = pd.DataFrame(Y_data, index=X_data.index, columns=predict_feature)
    Y_data_predict = pd.DataFrame(Y_data_predict, index=X_data_predict.index, 
                                  columns=predict_feature)
    
    # 对 data 和 data_predict 上有缺失值的数据进行补充
    for col in predict_feature:
        data[col][data[col].isna()] = Y_data[col][data[col].isna()]
        data_predict[col][data_predict[col].isna()] = \
        Y_data_predict[col][data_predict[col].isna()]
    if verbose:
        print('Done!')
    return data, data_predict


# [data_loader]: 获取数据（包含数据清洗）
# ---- Example ---- #
# data, X_pred = data_loader(fillna=fillna, verbose=verbose)
def data_loader(fillna='predict', verbose=True,
                data_path = path+'/data/used_car_train_20200313.csv',
                data_predict_path = path+'/data/used_car_testA_20200313.csv'):
    data = pd.read_csv(data_path, sep=' ')
    data_predict = pd.read_csv(data_predict_path, sep=' ')
    
    # 对汽车使用日期数据进行提取
    # 由于部分 regDate 的月份为 0，因此使用 errors='coerce' 强制转换
    # 转换后将缺失值的月份补为 1，并重新计算日期
    for data_set in (data, data_predict):
        data_set['used_time'] = (pd.to_datetime(data_set['creatDate'], format='%Y%m%d', errors='coerce') - 
                            pd.to_datetime(data_set['regDate'], format='%Y%m%d', errors='coerce')).dt.days
        data_set['regDate'][data_set.used_time.isna()] = data_set['regDate'][data_set.used_time.isna()]+1000
        data_set['used_time'] = (pd.to_datetime(data_set['creatDate'], format='%Y%m%d', errors='coerce') - 
                            pd.to_datetime(data_set['regDate'], format='%Y%m%d', errors='coerce')).dt.days    

    # 将字符类型转换为数字类型（未知的赋值为2）
    data['notRepairedDamage'] = data['notRepairedDamage'].map({'0.0':0, '1.0':1, '-':2})
    data_predict['notRepairedDamage'] = data_predict['notRepairedDamage'].map({'0.0':0, '1.0':1, '-':2})

    # SaleID, name, creatDate 对价格预测无作用，删除
    # 所有的 offerType 都是 0 （都提供报价），删除
    # 除了 data 中第 75924 行的 seller 为1，其他都为0，删除
    # regDate 已转换，删除
    drop_columns = ['SaleID','name','seller','offerType','regDate','creatDate']
    data.drop(drop_columns, axis=1, inplace=True)
    data_predict.drop(drop_columns, axis=1, inplace=True)
    # 只有这 38424 行的 model 是缺失的，删除
    data.drop(38424, inplace=True)
    # 将功率和价格太小太大的删除
    data = data[
        (data.power>=10)&(data.power<=600)&
        (data.price>100)&(data.price<40000)]

    # 填充 ['bodyType', 'fuelType', 'gearbox'] 中的缺失值
    if fillna is None: # 不处理
        if verbose:
            print('fillna = None')
    elif fillna == 'predict': # 使用随机森林预测
        if verbose:
            print('fillna = \'predict\'')
            data, data_predict = nan_predicter(data, data_predict)
        else:
            data, data_predict = nan_predicter(data, data_predict, verbose=False)
    elif fillna == 'new_tag': # 填充为 -1
        if verbose:
            print('fillna == \'new_tag\'')
        for col in ['bodyType', 'fuelType', 'gearbox']:
            for D in [data, data_predict]:
                D[col].fillna(-1, inplace=True)
    else:
        print('\n\'fillna\' cannot be', fillna)
        return None
    return data, data_predict


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# [get_data]: 获取训练数据
# ---- Example ---- #
# X, y, X_pred = get_data(fillna='predict', box_cox=True, split=False,
#            verbose=True, scale=True, random_state=520)
def get_data(fillna='predict', box_cox=True, split=False,
            verbose=True, scale=True, random_state=520):
    if verbose:
        print('get_data(fillna={}, box_cox={}, verbose={}, scale={}, random_state={})'.\
            format(fillna, box_cox, verbose, scale, random_state))
    data, X_pred = data_loader(fillna=fillna, verbose=verbose)
    X = data.drop('price', axis=1)
    y = data['price']
    if box_cox:
        if verbose:
            print('\nY_data had been applied box-cox transform!!!')
            print('Use [np.exp(y_pred)] to inverse transform!!!')
        y = pd.Series(np.log(y), index=y.index, name=y.name)

    if scale:
        X_columns = X.columns
        X_scaler = MinMaxScaler().fit(X.append(X_pred))
        X = pd.DataFrame(X_scaler.transform(X), columns=X_columns)
        X_pred = pd.DataFrame(X_scaler.transform(X_pred), columns=X_columns)
    
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=50000, random_state=random_state)
        return X_train, X_test, y_train, y_test, X_pred
    else:
        return X, y, X_pred


from sklearn.metrics import mean_absolute_error
# [MAE_log]: 计算经过 box-cox 变换的预测 MAE（并规范化到长度 50000）
def MAE_log(y_pred , y_test):
    mae = mean_absolute_error(np.exp(y_pred) , np.exp(y_test))
    mae = mae*50000/len(y_test)
    return mae

# [MAE_score]: 利用 estimator 计算经过 box-cox 变换的预测 MAE（并规范化到长度 50000）
def MAE_score(estimator, X , y):
    y_pred = estimator.predict(X)
    mae = mean_absolute_error(np.exp(y_pred) , np.exp(y))
    mae = mae*50000/len(y)
    return mae


# [model_test]: 快速测试模型
# ---- Example ---- #
# # 导入包
# import warnings
# warnings.filterwarnings("ignore")
# import pandas as pd
# %config InlineBackend.figure_format='retina'
# from utils import get_data, model_test
# from sklearn.model_selection import train_test_split
# # 获得数据
# X, y, X_pred= get_data(fillna='predict', box_cox=True, split=False, 
#                        verbose=True, random_state=520)
# X_train, X_test, y_train, y_test = \
#     train_test_split(X,y, test_size=50000, random_state=520)
# # 训练模型
# from sklearn.tree import DecisionTreeRegressor
# estimator = DecisionTreeRegressor(max_depth=13)
# test_score, train_score, y_pred = \
# model_test(estimator, X_train, X_test, y_train, y_test,
# return_train_score=True, plot_predict=True)
# print('test MAE:', test_score)
# print('train MAE:', train_score)
def model_test(estimator, X_train, X_test, y_train, y_test,
               return_train_score=True, plot_predict=False, verbose=True):
    if verbose:
        print('The model_test fun is use for box_cox data!')
    y_pred = estimator.fit(X_train, y_train).predict(X_test)
    test_score = MAE_log(y_pred, y_test)
    if plot_predict:
        pd.DataFrame({'Pred':y_pred[:500] ,'Test':y_test.values[:500]}).\
            plot(figsize=(12,5), title='test_MAE: '+str(round(test_score, 4)))
    if return_train_score:
        train_score = MAE_score(estimator, X_train, y_train)
        return test_score, train_score, y_pred
    else:
        return test_score, y_pred


from sklearn.model_selection import cross_validate
# [model_test_cv]: 快速测试模型（交叉验证）
# ---- Example ---- #
# # 导入包
# from utils import get_data, model_test_cv
# # 获得数据
# X, y, X_pred= get_data(fillna='predict', box_cox=True, split=False, 
#                        verbose=True, random_state=520)
# # 训练模型
# from sklearn.tree import DecisionTreeRegressor
# estimator = DecisionTreeRegressor(max_depth=13)
# mean_test_score, mean_train_score = model_test_cv(estimator, X, y , verbose_eval=1, 
#                                                  cv=3, return_train_score=True)
# print('mean_test_score', mean_test_score)
# print('mean_train_score', mean_train_score)
def model_test_cv(estimator, X, y ,cv=3, 
                  verbose_eval=1, return_train_score=True):
    CV = cross_validate(estimator, X, y, cv=cv, 
                        n_jobs=-1, 
                        verbose=verbose_eval, 
                        scoring=MAE_score, 
                        return_train_score=return_train_score)
    mean_test_score = np.mean(CV['test_score'])
    if return_train_score:
        mean_train_score = np.mean(CV['train_score'])
        return mean_test_score, mean_train_score
    else:
        return mean_test_score


# [submit_fun]: 直接使用估计器预测并保存文件
def submit_fun(estimator, X_pred, 
               path='/Users/apple/Documents/Python/TianChi/Car_Price/', 
               file_name='submit.csv'):
    Y_pred = np.exp(estimator.predict(X_pred))
    submit = pd.read_csv(path+'data/used_car_sample_submit.csv')
    submit['price'] = Y_pred
    submit.to_csv(path+file_name, index=False)
    print('Saved in',path)

# [submit_fun2]: 将预测结果保存文件
def submit_fun2(Y_pred, 
               path='/Users/apple/Documents/Python/TianChi/Car_Price/', 
               file_name='submit.csv'):
    Y_pred = np.exp(Y_pred)
    submit = pd.read_csv(path+'data/used_car_sample_submit.csv')
    submit['price'] = Y_pred
    submit.to_csv(path+file_name, index=False)
    print('Saved in',path)


from sklearn.decomposition import PCA
def pca_module(X, X_pred, n_components=5, verbose=True):
    v_columns = ['v_'+str(num) for num in range(0,15)]
    pca  = PCA().fit(X[v_columns])
    if verbose:
        print('PCA components:', n_components)
        print('Explained Variance:', np.cumsum(pca.explained_variance_ratio_)[n_components])
    column_names = ['pca_'+str(num) for num in range(0, n_components)]
    X_v_data = pd.DataFrame(np.dot(X[v_columns], 
                                   pca.components_[:n_components].T), 
                            columns=column_names)
    X_pred_v_data = pd.DataFrame(np.dot(X_pred[v_columns], 
                                        pca.components_[:n_components].T), 
                                 columns=column_names)
    X.drop(v_columns, axis=1, inplace=True)
    X_pred.drop(v_columns, axis=1, inplace=True)
    X = pd.concat([X, X_v_data], axis=1)
    X_pred = pd.concat([X_pred, X_pred_v_data], axis=1)
    return X, X_pred