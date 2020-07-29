import sklearn
import numpy as np
import pickle
import pandas as pd
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
# from sklearn import preprocessing
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# from sklearn.datasets import load_iris
# X,Y =load_diabetes(return_X_y=True)
# scaler = StandardScaler()
# X,Y = load_iris(return_X_y=True)

# scaler.fit(X)
# X = scaler.transform(X)

class compare_classifiers():
    def __init__(self, 
                X: '(np ndarray) size of num_samples x num_features', 
                Y: '(np ndarray[int]) size of num_samples', 
                n_folds: '(int)', 
                use_pca: '(float) percent of importance' = -1,
                random_state=345):
        self.x = X
        self.y = Y
        self.n_folds = n_folds
        self.kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=random_state)
        self.use_pca = use_pca
        if 0. < use_pca <= 1.0:
            print('Running PCA')
            self.PCA_vectorizer = PCA(n_components=use_pca)
            print('Transforming the input')
            original_num_features = self.x.shape
            self.x = self.PCA_vectorizer.fit_transform(self.x)
            new_num_features = self.x.shape
            print(f'original shape: {original_num_features}, new shape: {new_num_features}')
    def all_models(self):
#         param_grid = ParameterGrid({'C': [0.01, 0.1, 1, 10, 30], 'solver': ['lbfgs', 'liblinear']})
        for C in [0.01, 0.1, 1, 10, 30, 100]:
            model = LogisticRegression(C=C, class_weight='balanced', multi_class='ovr', max_iter=6000, solver='lbfgs')
#             model.set_params(**param)
            yield {'model':model, 'model_name':'Logistic', 'param': {'C': C, 'solver':'lbfgs'}}
        for C in [0.01, 0.1, 1, 10, 30, 100]:
            model = LogisticRegression(C=C, class_weight='balanced', multi_class='ovr', max_iter=6000, solver='liblinear')
#             model.set_params(**param)
            yield {'model':model, 'model_name':'Logistic', 'param': {'C': C, 'solver':'liblinear'}}

#         param_grid = ParameterGrid({'num_leaves': [4, 8, 32, 64], 'solver' = ['lbfgs', 'liblinear']})
#         for param in param_grid:
#             model = LGBMClassifier(class_weight='balanced', solver= , objective='multiclass')
#             model.set_params(**param)
#             yield (model, 'linear')
        
#         param_grid = ParameterGrid({'C': [0.01, 0.1, 1, 10, 100]})
        
        for C in [0.01, 0.1, 1, 10, 30, 100]:
            model = SVC(class_weight='balanced',  kernel = 'linear', max_iter=6000, C=C)
#             model.set_params(**param)
            yield {'model':model, 'model_name':'SVM_linear', 'param': {'C': C}}
        
#         param_grid = ParameterGrid({'C': [0.01, 0.1, 1, 10, 100]})
#         for param in param_grid:
        for C in [0.01, 0.1, 1, 10, 30, 100]:
            model = SVC(class_weight='balanced',  kernel = 'rbf', max_iter=6000, C=C)
#             model.set_params(**param)
            yield {'model':model, 'model_name':'SVM_rbf', 'param': {'C': C}}
        
        param_grid = ParameterGrid({ 'max_depth': [2, 4, 6, 12, 18, 24, 32, None]})
        for param in param_grid:
            model = RandomForestClassifier(class_weight='balanced', max_features='auto')
            model.set_params(**param)
            yield {'model':model, 'model_name':'RandomForest', 'param': param}
            
        model = GaussianNB()
        yield {'model':model, 'model_name':'GaussianNB', 'param': None}
    def fit_models(self):
        across_fold_UWf1s = []
        across_fold_Wf1s = []
        model_list = []
        for ind, (train, test) in enumerate(self.kfold.split(self.x, self.y)):
            print(f'fold: {ind}')
            UWf1s = []
            Wf1s = []
#             print(train)
#             print(test)
            x_tr = self.x[train]
            y_tr = self.y[train]
            x_te = self.x[test]
            y_te = self.y[test]
            print('begin_training')
            for model_dict in self.all_models():
                print('fit model')
                print(model_dict['model_name'])
                print(model_dict['param'])
                model_dict['model'].fit(x_tr, y_tr)
                print('after fitting')
                y_pred = model_dict['model'].predict(x_te)
                print(y_pred)
                print(y_te)
                print('after predicting')
                f12 = f1_score(y_te, y_pred, average='weighted')
                Wf1s.append(f12)
#                 print('f12')
                f11 = f1_score(y_te, y_pred, average='macro')
                UWf1s.append(f11)
#                 print('f11')
                
                if ind == 0:
                    model_list.append(model_dict['model_name'])
#                 del model_dict['model']
            across_fold_UWf1s.append(UWf1s)
            across_fold_Wf1s.append(Wf1s)
            print(across_fold_UWf1s)
            print(across_fold_Wf1s)
        print('1')
        across_fold_UWf1s = np.stack(across_fold_UWf1s)
        across_fold_Wf1s = np.stack(across_fold_Wf1s)
        across_fold_UWf1s = np.mean(across_fold_UWf1s,axis=0)
        across_fold_Wf1s = np.mean(across_fold_Wf1s,axis=0)
        f1s = (across_fold_UWf1s+across_fold_Wf1s)/2
        with open('result_models.pkl', 'wb')as f:
            pickle.dump({'f1s':f1s, 'across_fold_UWf1s': across_fold_UWf1s, 'across_fold_Wf1s': across_fold_Wf1s, 'model_names': model_list},f)
        best_dictionary = {}
        print(len(model_list))
        print(len(f1s))
        for ind, (model_name, f1) in enumerate(zip(model_list, f1s)):
            if model_name not in best_dictionary:
                best_dictionary[model_name] = (ind, f1)
            else:
                if best_dictionary[model_name][1] < f1:
                    best_dictionary[model_name] = (ind, f1)
        for model_name in best_dictionary:
            best_dictionary[model_name] = {'Unweighted_f1': across_fold_UWf1s[best_dictionary[model_name][0]],
                                          'Weighted_f1': across_fold_Wf1s[best_dictionary[model_name][0]],
                                          'Avg_f1': f1s[best_dictionary[model_name][0]]}
        model_names = []
        Unweighted_f1 = []
        Weighted_f1 = []
        Avg_f1 = []
        print('2')
        for model_name in best_dictionary:
            model_names.append(model_names)
            Unweighted_f1.append(best_dictionary[model_name]['Unweighted_f1'])
            Weighted_f1.append(best_dictionary[model_name]['Weighted_f1'])
            Avg_f1.append(best_dictionary[model_name]['Avg_f1'])
        print('3')
        df = pd.DataFrame({'Model': model_names, 'Unweighted_f1score': Unweighted_f1, 
                           'Weighted_f1score':Weighted_f1, 'Avg_f1score': Avg_f1})
        print(df)
        
# a = compare_classifiers(X,Y,n_folds=5)
# a.fit_models()