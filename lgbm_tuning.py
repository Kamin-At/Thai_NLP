import sklearn
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
# from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from operator import itemgetter 
import os
from sklearn.metrics import f1_score
from tqdm import trange
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from text_classification import prepare_data_for_text_classification

if __name__ == "__main__":
    tr = pd.read_csv('train_RS_123.csv')
    tr['texts'] = tr['texts'].apply(str)
    te = pd.read_csv('test_RS_123.csv')
    te['texts'] = te['texts'].apply(str)
    list(te['labels'].unique())

    out = prepare_data_for_text_classification(pd.concat((tr,te)).reset_index(drop=True), 
                                             test_dataframe=te, 
                                             max_len=32, 
                                             min_len=2, 
                                             n_gram_range=(1,3),
                                             min_df=2,
                                             word_embedder='fasttext',
                                             tfxidf_path='tf-idf_encoder3',
                                             engine='newmm',
        #                                      threshold_tfxidf=0.01,
                                             verbose=False)
    print('PCA')
    vectorizer = PCA(n_components=0.99)
    X = vectorizer.fit_transform(out['tfxidf_train'])
    # X_te = vectorizer.transform(out['tfxidf_test'])
    print(X.shape)
    # print(X_te.shape)
    # X = np.concatenate((X, X_te))
    # print(X.shape)
    Y = out['tfxidf_label_train']
    # Y = pd.concat((out['tfxidf_label_train'], out['tfxidf_label_test']))
    estimator = lgb.LGBMClassifier(learning_rate = 0.125, metric = 'multi_logloss', 
                            n_estimators = 20, num_leaves = 38, objective='multiclass', class_weight='balanced',num_iterations=1000)


    param_grid = {
        'n_estimators': list(range(10,40,4)),
        'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],
        'max_depth': list(range(4,33,4)),
        'subsample_for_bin': list(range(150000, 250001, 25000)),
        'min_data_in_leaf': [0, 0.01, 0.02, 0.03, 0.04, 0.05],
        'lambda_l2': [0, 0.1, 0.01, 0.001, 1e-4]
    }
    print('param grid')
    print(param_grid)
    gridsearch = GridSearchCV(estimator, param_grid, return_train_score=True,refit=False, scoring=['f1_macro', 'f1_weighted'], n_jobs=-1)
    print('fitting')
    gridsearch.fit(X, Y)#,
    #         eval_metric = ['auc'],
    #         early_stopping_rounds = 5)
    print('fitting completed')
    x = gridsearch.cv_results_
    x = pd.DataFrame(x)
    x.to_csv('lgbm_results.csv', index=False)