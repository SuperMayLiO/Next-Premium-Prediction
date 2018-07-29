#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 17:04:40 2018

@author: mayritaspring
"""


import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

default_path = "/Users/mayritaspring/Desktop/T-Brain/Next-Premium-Prediction/"
os.chdir(default_path)

# read data
claim = pd.read_csv('../Data/claim_0702.csv')
policy = pd.read_csv('../Data/policy_0702.csv')


# Function for Measure Performance
from  sklearn  import  metrics
def measure_performance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True, show_roc_auc = True, show_mae = True):
    y_pred = clf.predict(X)
    y_predprob = clf.predict_proba(X)[:,1]
    if show_accuracy:
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred))),"\n"

    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y,y_pred)),"\n"
        
    if show_confusion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y,y_pred)),"\n"  
        
    if show_roc_auc:
        print("ROC AUC Score:{0:.3f}".format(metrics.roc_auc_score(y,y_predprob))),"\n"
        
    if show_mae:
        print("Mean Absolute Error:{0:.3f}".format(metrics.mean_absolute_error(y, y_pred, multioutput='raw_values'))),"\n"

    
# Label encoding (Convert catgorical data to interger catgories)
from sklearn.preprocessing import LabelEncoder
def label_encoder(input_df, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """
    # Label encode categoricals
    categorical_feats = [col for col in input_df.columns if (input_df[col].dtype == 'object' or len(input_df[col].unique().tolist()) < 20)]
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
    return input_df, categorical_feats, encoder_dict


#claim
claim, claim_categorical_feats, claim_encoder_dict = label_encoder(input_df = claim)
X = claim.drop('Premium', axis=1)
y = claim.Premium

#policy
policy, policy_categorical_feats, policy_encoder_dict = label_encoder(input_df = policy)
X = policy.drop('Premium', axis=1)
y = policy.Premium


df_new = claim.join(other = policy,on = ['Policy_Number'], how = 'left')


#Step 1: parameters to be tuned
def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
    params = {'application':'regression','num_iterations':4000, 'learning_rate':0.05, 'early_stopping_round':100, 'metric':'l1'}
    params["num_leaves"] = int(round(num_leaves))
    params['feature_fraction'] = max(min(feature_fraction, 1), 0)
    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    params['max_depth'] = round(max_depth)
    params['lambda_l1'] = max(lambda_l1, 0)
    params['lambda_l2'] = max(lambda_l2, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['l1'])
    return min(cv_result['l1-mean'])



#Step 2: Set the range for each parameter
lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                        'feature_fraction': (0.1, 0.9),
                                        'bagging_fraction': (0.8, 1),
                                        'max_depth': (5, 8.99),
                                        'lambda_l1': (0, 5),
                                        'lambda_l2': (0, 3),
                                        'min_split_gain': (0.001, 0.1),
                                        'min_child_weight': (5, 50)}, random_state=0)


# ### Step 3: Bayesian Optimization: Maximize
# lgbBO.maximize(init_points=init_round, n_iter=opt_round)


# ### Step 4: Get the parameters
# lgbBO.res['max']['max_params']

# ### Put all together
def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, categorical_feature = claim_categorical_feats, free_raw_data=False)
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight, max_bin):
        params = {'application':'regression','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'l1'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['max_bin'] = int(round(max_bin))
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['l1'])
        return min(cv_result['l1-mean'])
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50),
                                           'max_bin': (5, 50)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return best parameters
    return lgbBO.res['max']['max_params']

opt_params = bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=10, n_folds=3, random_seed=6, n_estimators=100, learning_rate=0.05)
print('Bayesian Optimization Parameters...')
print(opt_params)

####################################################################################
# lightgbm with Bayesian optimization
# Prepare dataset 
from sklearn import cross_validation
seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)

# LGBM with Bayesian Optimization
LGBM_bayes = LGBMClassifier(
    num_leaves=35,
    feature_fraction = 0.1,
    bagging_fraction = 0.8,
    max_depth=6,
    lambda_l1 = 1.4,
    lambda_l2 = 0.07,
    min_split_gain=0.04,
    min_child_weight=37,
    max_bin = 27,
    random_state=0, 
    n_estimators=100, 
    learning_rate=0.05,
    application = 'regression',
    num_iterations = 10000, 
    early_stopping_round = 100, 
    metric = 'l1'
  )

LGBM_bayes_fit = LGBM_bayes.fit(X_train, y_train, eval_metric= 'l1', verbose= 100, early_stopping_rounds= 200)

# measure performance
LGBM_bayes_measure = measure_performance(X = X_test, y = y_test, clf = LGBM_bayes, show_mae = True)
print(LGBM_bayes_measure)

# feature importances
print('Feature importances:', list(LGBM_bayes.feature_importances_))

# visualization
print('Plot feature importances...')
ax = lgb.plot_importance(LGBM_bayes_fit, max_num_features=10)
plt.show()

# Submission file
testing_set = pd.read_csv('../Data/testing-set.csv')
test_df = label_encoder(testing_set)[0]

#Bayes Optimization
out_bayes = pd.DataFrame({"Policy_Number":test_df["Policy_Number"], "Next_Premium":LGBM_bayes.predict_proba(test_df)[:,1]})
out_bayes.to_csv("submissions_policy_LGBM_bayesian.csv", index=False)
