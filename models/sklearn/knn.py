import dataset.setbuilder as sb
import models.sklearn.evaluator as eval
from preprocessing import preprocessing_utils
import dataset.dataset as ds
import dataset.utility as utils
import numpy as np
import preprocessing.imputation as imp
import pandas
import models.sklearn.sklearnlinearclass as skc
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn import preprocessing
import pandas as pd

def model():
    return knn(n_neighbors=10, weights='uniform', algorithm='ball_tree', leaf_size=15)


if __name__ == '__main__':
    datas = ds.read_dataset("best_for_customers.csv")

    '''
    datas['Month'] = datas['Date']
    datas['Month'] = datas['Month'].apply(lambda x: x.split("-")[1])
    datas = imp.one_hot_numeric(datas, 'Month', 'Month_')
    datas = imp.one_hot_numeric(datas, 'Region', 'Region_')
    datas = preprocessing_utils.mean_cust_per_month_per_region(datas, utils.get_frame_in_range(datas, 3, 2016, 12, 2017))
    '''

    # With MinMaxScaler performance dropped a lot. Consider reverting
    min_max_scaler = preprocessing.MinMaxScaler()

    datas[['NumberOfSales', 'NumberOfCustomers', 'MeanCustPerShopPerDay', 'maxcust_shop', 'MeanCustPerRegionPerMonth', 'MeanCustPerShopPerMonth',
           'meancust_std_shop', 'meancustshop', 'mincust_shop']] = pd.DataFrame(min_max_scaler.fit_transform(datas[['NumberOfSales',
                                                                                                       'NumberOfCustomers',
                                                                                                       'MeanCustPerShopPerDay',
                                                                                                       'maxcust_shop',
                                                                                                       'MeanCustPerRegionPerMonth',
                                                                                                        'MeanCustPerShopPerMonth',
                                                                                                       'meancust_std_shop',
                                                                                                       'meancustshop',
                                                                                                       'mincust_shop']]))

    datas = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True, df=datas).exclude('NumberOfSales', 'Month').build()


    mod = skc.LinearSklearn(1, model)
    x = datas.xtr
    y = datas.ytr
    mod.train(x, y)
    custpred = mod.predict(datas.xts)
    print("TEST R2: ", eval.r2_score(datas.yts, custpred))
    print("##########################")
    new = pandas.DataFrame()
    new['NumberOfCustomers'] = pandas.Series(custpred)
    ds.save_dataset(new, "knncustpreds1.csv")
