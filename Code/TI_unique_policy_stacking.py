# Forked from excellent kernel : https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features
# From Kaggler : https://www.kaggle.com/jsaguiar
# Just removed a few min, max features. U can see the CV is not good. Dont believe in LB.
import os
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from bayes_opt import BayesianOptimization
import datetime
from sklearn.metrics import mean_absolute_error

# Load in our libraries
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
    
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, exclude_list = [""], nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if (df[col].dtype == 'object' or len(df[col].unique().tolist()) < 20) and( col not in exclude_list) ]
    #if ("Next_Premium" in categorical_columns): categorical_columns.remove('Next_Premium')
    #if ("Policy_Number" in categorical_columns): categorical_columns.remove('Policy_Number')
    #if ("Insured's_ID" in categorical_columns): categorical_columns.remove("Insured's_ID")
    #if ("Prior_Policy_Number" in categorical_columns): categorical_columns.remove('Prior_Policy_Number')
    #if ("Vehicle_identifier" in categorical_columns): categorical_columns.remove('Vehicle_identifier')
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

'''
import os
os.chdir(r"C:\Users\chingyu\Documents\code\Next-Premium-Prediction\Code")
num_rows = None
nan_as_category = True
num_folds = 5
stratified = False
debug = False
'''

# Preprocess training-set.csv and testing-set.csv
def PreprocessTrainTest(num_rows = None):
    # Read data and merge
    df = pd.read_csv('../data/training-set.csv', nrows= num_rows)
    test_df = pd.read_csv('../data/testing-set.csv', nrows= num_rows)
    try:
        test_df = test_df.drop("Next_Premium", axis=1)
    except:
        0
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df)#.reset_index()
    print("Unique policy number in training and testing data: {}".format(len(set(df.loc[:,'Policy_Number']))))
    del test_df
    gc.collect()
    return df

#df = PreprocessTrainTest(num_rows)


# Preprocess claim_0702.csv
def PreprocessClaim(num_rows = None, nan_as_category = False):
    claim = pd.read_csv('../data/claim_0702.csv', nrows= num_rows)
    print("Unique policy number in claim data: {}".format(len(set(claim.loc[:,'Policy_Number']))))
    exclude_list = ["Next_Premium", "Claim_Number", "Policy_Number", "DOB_of_Driver", "Vehicle_identifier",
                    ]# remove useless variable:
                    #'Claim_Status_(close,_open,_reopen_etc)']
 
    claim["Nature_of_the_claim"] = claim["Nature_of_the_claim"].astype(object)
    claim["Driver's_Gender"] = claim["Driver's_Gender"].astype(object)
    claim["Driver's_Relationship_with_Insured"] = claim["Driver's_Relationship_with_Insured"].astype(object)
    claim["Marital_Status_of_Driver"] = claim["Marital_Status_of_Driver"].astype(object)
    claim["Claim_Status_(close,_open,_reopen_etc)"] = claim["Claim_Status_(close,_open,_reopen_etc)"].astype(object)
    
    
    now = pd.Timestamp(datetime.datetime.now())
    a = claim['DOB_of_Driver'].str.split('/',1, expand=True)
    a.loc[:,1].replace('NaN', np.nan, inplace= True)
    a.loc[:,1] = a.loc[:,1].astype(float)
    
    claim['DOB_of_Driver'][(a.loc[:,1] > now.year)] = np.nan
    claim['age_of_Driver'] = (now - pd.to_datetime(claim['DOB_of_Driver'], format='%m/%Y')).astype('<m8[Y]') 
    

    claim_aggregations = {'Claim_Number': ['size', 'nunique'],
                          'Paid_Loss_Amount': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                          'paid_Expenses_Amount': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                          'Salvage_or_Subrogation?': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                          'Vehicle_identifier': ['size', 'nunique'],
                          'At_Fault?': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                          'Deductible': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                          'age_of_Driver': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                          'number_of_claimants': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                         
                           }
    
    
    number_of_claimants = claim['number_of_claimants']
    claim, claim_cat = one_hot_encoder(claim, exclude_list, nan_as_category)
    for col in claim_cat:
        claim_aggregations[col] = ['mean', 'size']
    claim['number_of_claimants'] = number_of_claimants 
    del number_of_claimants
    claim_agg = claim.groupby('Policy_Number').agg(claim_aggregations)
    claim_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in claim_agg.columns.tolist()])
    claim_agg['Is_Claim'] = 1
    
    del claim
    gc.collect()
    return claim_agg


#claim_agg = PreprocessClaim(num_rows, nan_as_category)


'''
    for i in claim.columns:
        print("length og unique column =>" + i +":"+str(len(set(claim.loc[:,i]))) + " == type: "+ str(claim[i].dtypes))
'''



# Preprocess policy_0702.csv
def PreprocessPolicy(num_rows = None, nan_as_category = False):
    # Read data and merge
    policy = pd.read_csv('../data/policy_0702.csv', nrows= num_rows)
    print("Unique policy number in policy data: {}".format(len(set(policy.loc[:,'Policy_Number']))))

    exclude_list = ["Next_Premium", "Policy_Number", "Insured's_ID",
                    'Prior_Policy_Number','Vehicle_identifier','Vehicle_Make_and_Model2',
                    'Coding_of_Vehicle_Branding_&_Type','Distribution_Channel',
                    'aassured_zip','ibirth','dbirth',
                    'Refund_Coverage_Deductible_if_applied',
                    'Partial_Refund_1st_Coverage_Deductible_if_applied',
                    'Partial_Refund_2nd_Coverage_Deductible_if_applied',
                    'Partial_Refund_3rd_Coverage_Deductible_if_applied',
                    'Partial_Refund_mean_Coverage_Deductible_if_applied',
                    'Amount_Refund_1st_Coverage_Deductible_if_applied',
                    'Amount_Refund_2nd_Coverage_Deductible_if_applied',
                    'Amount_Refund_3rd_Coverage_Deductible_if_applied',
                    'Amount_Refund_mean_Coverage_Deductible_if_applied',
                    'Paid_Coverage_Deductible_if_applied',
                    'Partial_Paid_1st_Coverage_Deductible_if_applied',
                    'Partial_Paid_2nd_Coverage_Deductible_if_applied',
                    'Partial_Paid_3rd_Coverage_Deductible_if_applied',
                    'Partial_Paid_mean_Coverage_Deductible_if_applied',
                    'Amount_Paid_1st_Coverage_Deductible_if_applied',
                    'Amount_Paid_2nd_Coverage_Deductible_if_applied',
                    'Amount_Paid_3rd_Coverage_Deductible_if_applied',
                    'Amount_Paid_mean_Coverage_Deductible_if_applied',
                    ]# remove useless variable:
                    #'Vehicle_Make_and_Model1','fequipment1','fequipment2','fequipment3','fequipment4','fequipment5','fequipment6','fequipment9']



    now = pd.Timestamp(datetime.datetime.now())
    
    a = policy['dbirth'].str.split('/',1, expand=True)
    a.loc[:,1].replace('NaN', np.nan, inplace= True)
    a.loc[:,1] = a.loc[:,1].astype(float)
    
    policy['dbirth'][(a.loc[:,1] > now.year)] = np.nan
    policy['iage'] = (now - pd.to_datetime(policy['ibirth'], format='%m/%Y')).astype('<m8[Y]') 
    policy['dage'] = (now - pd.to_datetime(policy['dbirth'], format='%m/%Y')).astype('<m8[Y]') 


    policy['Insured_Amount_sum'] = policy['Insured_Amount1'] + policy['Insured_Amount2'] + policy['Insured_Amount3']
    policy['Insured_Amount_mean'] = policy[['Insured_Amount1','Insured_Amount2','Insured_Amount3']].replace(0,np.nan).apply(np.nanmean, axis=1)
    policy['Insured_Amount_sum_prmium_ratio'] = policy['Insured_Amount_sum']/policy['Premium']
    policy['Insured_Amount_mean_prmium_ratio'] = policy['Insured_Amount_mean']/policy['Premium']
    policy['vehicle_age'] = now.year - policy['Manafactured_Year_and_Month']
    policy['Replacement_cost_plia_acc'] = (1 + policy['plia_acc']) * policy['Replacement_cost_of_insured_vehicle']
    policy['Replacement_cost_pdmg_acc'] = (1 + policy['pdmg_acc']) * policy['Replacement_cost_of_insured_vehicle']
    policy['Premium_plia_acc'] = (1 + policy['plia_acc']) * policy['Premium']
    policy['Premium_pdmg_acc'] = (1 + policy['pdmg_acc']) * policy['Premium']


    Insurance_Coverage_dict = {'00I': '車損', '01A': '車損', '01J': '車損', '02K': '車損', '03L': '車損',
                               '04M': '車損', '05E': '車損', '05N': '竊盜', '06F': '車損', '07P': '車損',
                                '08H': '車損', '09@': '竊盜', '09I': '竊盜', '10A': '竊盜', '12L': '車責', 
                                '14E': '車責', '14N': '車損', '15F': '車責', '15O': '車責', '16G': '車責', 
                                '16P': '車責', '18@': '車責', '18I': '車責', '20B': '車損', '20K': '車損', 
                                '25G': '車責', '26H': '車責', '27I': '車責', '29B': '車責', '29K': '車責',
                                '32N': '車損', '33F': '車損', '33O': '車損', '34P': '車損', '35H': '車損', 
                                '36I': '車損', '37J': '車責', '40M': '車責', '41E': '車責', '41N': '車責', 
                                '42F': '車責', '45@': '車損', '46A': '車責', '47B': '車責', '51O': '車損',
                                '55J': '車損', '56B': '車損', '56K': '車損', '57C': '車損', '57L': '車責', 
                                '65K': '車責', '66C': '車損', '66L': '車損', '67D': '車損', '68E': '竊盜',
                                '68N': '竊盜', '70G': '車責', '70P': '車責', '71H': '車責', '72@': '車責'}
    
    policy['Insurance_Coverage_Name'] = policy['Insurance_Coverage'].map(Insurance_Coverage_dict)
    IsEmpty_Insurance_Coverage = ['01A','01J','03L','05E','06F','07P',
                                '08H','09@','12L','16G','16P','18@','18I','25G',
                                '26H','27I','29B','34P','35H','36I','40M','41E',
                                '41N','42F','45@','46A','47B','51O','55J','56B',
                                '57C','57L','67D','68N','70G','70P','71H','72@']
    IsCorrelated_Insurance_Coverage = ['09I','10A','14E','15F','15O',
                                        '20B','20K','29K','32N','33F',
                                        '33O','56K','65K'] 
    IsPartial_Insurance_Coverage = ['05N','14N','68E']
    IsAmount_Insurance_Coverage = ['00I','02K','04M','37J','66C','66L']
    
    ## empty/ not correlated/ correlated
    policy['IsEmpty_Coverage_Deductible_if_applied'] =  [a in IsEmpty_Insurance_Coverage for a in policy['Insurance_Coverage'] ]
    policy['IsCorrelated_Coverage_Deductible_if_applied'] =  [a in IsCorrelated_Insurance_Coverage for a in policy['Insurance_Coverage'] ]
    policy['IsPartial_Coverage_Deductible_if_applied'] =  [a in IsPartial_Insurance_Coverage for a in policy['Insurance_Coverage'] ]
    policy['IsAmount_Coverage_Deductible_if_applied'] =  [a in IsAmount_Insurance_Coverage for a in policy['Insurance_Coverage'] ]
    policy['IsPaid_Coverage_Deductible_if_applied'] =  [a in (IsPartial_Insurance_Coverage+IsAmount_Insurance_Coverage) for a in policy['Insurance_Coverage'] ]
    policy['IsRefund_Coverage_Deductible_if_applied'] = policy['Coverage_Deductible_if_applied'] <= 0
    

    Deductible_1st_dict = {1: 3000, 2: 5000, 3: 5000, 6: 15000, 7: 20000,
                           -1: -3000, -2: -5000, -3: -5000, -6: -15000, -7: -20000}
    Deductible_2nd_dict = {1: 5000, 2: 8000, 3: 8000, 6: 15000, 7: 20000,
                           -1: -5000, -2: -8000, -3: -8000, -6: -15000, -7: -20000}
    Deductible_3rd_dict = {1: 7000, 2: 8000, 3: 10000, 6:15000, 7:20000,
                           -1: -7000, -2: -8000, -3: -10000, -6: -15000, -7: -20000}
    Deductible_mean_dict = {1: 5000, 2: 6500, 3: 23000/3, 6:15000, 7:20000,
                            -1: -5000, -2: -6500, -3: -23000/3, -6: -15000, -7: -20000}
    

    for i in ['Refund','Paid']:
        if i == 'Refund':
            policy['{0}_Coverage_Deductible_if_applied'.format(i)] = -policy['Coverage_Deductible_if_applied'] 
        elif i == 'Paid':
            policy['{0}_Coverage_Deductible_if_applied'.format(i)] = policy['Coverage_Deductible_if_applied'] 
        policy['{0}_Coverage_Deductible_if_applied'.format(i)][policy['{0}_Coverage_Deductible_if_applied'.format(i)] <= 0] = np.nan        
        for j in ['Partial', 'Amount']:
            for k in ['1st','2nd','3rd','mean']:
                policy['{0}_{1}_{2}_Coverage_Deductible_if_applied'.format(j, i, k)]  = policy['{0}_Coverage_Deductible_if_applied'.format(i)]
                policy['{0}_{1}_{2}_Coverage_Deductible_if_applied'.format(j, i, k)][~policy['Is{0}_Coverage_Deductible_if_applied'.format(j)]] = np.nan
                policy['{0}_{1}_{2}_Coverage_Deductible_if_applied'.format(j, i, k)] = policy['{0}_{1}_{2}_Coverage_Deductible_if_applied'.format(j, i, k)].map(eval('Deductible_{0}_dict'.format(k)))

                

    
    policy['IsEmpty_Coverage_Deductible_if_applied'] = policy['IsEmpty_Coverage_Deductible_if_applied'].astype(object)
    policy['IsCorrelated_Coverage_Deductible_if_applied'] = policy['IsCorrelated_Coverage_Deductible_if_applied'].astype(object)
    policy['IsPartial_Coverage_Deductible_if_applied'] = policy['IsPartial_Coverage_Deductible_if_applied'].astype(object)
    policy['IsAmount_Coverage_Deductible_if_applied'] = policy['IsAmount_Coverage_Deductible_if_applied'].astype(object)
    policy['IsPaid_Coverage_Deductible_if_applied'] = policy['IsPaid_Coverage_Deductible_if_applied'].astype(object)
    policy['IsRefund_Coverage_Deductible_if_applied'] = policy['IsRefund_Coverage_Deductible_if_applied'].astype(object)


    ## TODO:
    # 年齡 *　性別
    # is 強制險
    # is 第三責任險
    


    policy_aggregations = {'Premium': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Manafactured_Year_and_Month': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Engine_Displacement_(Cubic_Centimeter)': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'qpt':  ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Insured_Amount1': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Insured_Amount2': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Insured_Amount3': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Premium': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Replacement_cost_of_insured_vehicle': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Multiple_Products_with_TmNewa_(Yes_or_No?)': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'lia_class': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'plia_acc': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'pdmg_acc':  ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'iage': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'dage': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Insured_Amount_sum': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Insured_Amount_mean' : ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Insured_Amount_sum_prmium_ratio': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Insured_Amount_mean_prmium_ratio': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'vehicle_age': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Replacement_cost_plia_acc' : ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Replacement_cost_pdmg_acc': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Premium_plia_acc': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Premium_pdmg_acc': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Refund_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Partial_Refund_1st_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Partial_Refund_2nd_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Partial_Refund_3rd_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Partial_Refund_mean_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Amount_Refund_1st_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Amount_Refund_2nd_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Amount_Refund_3rd_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Amount_Refund_mean_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Paid_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Partial_Paid_1st_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Partial_Paid_2nd_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Partial_Paid_3rd_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Partial_Paid_mean_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Amount_Paid_1st_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Amount_Paid_2nd_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Amount_Paid_3rd_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       'Amount_Paid_mean_Coverage_Deductible_if_applied': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum', 'nunique'],
                       "Insured's_ID": ['size', 'nunique'],
                       'Prior_Policy_Number': ['size', 'nunique'],
                       'Vehicle_identifier': [ 'size', 'nunique'],
                       }
    

    qpt = policy['qpt']
    pdmg_acc = policy['pdmg_acc']

    policy, policy_cat = one_hot_encoder(policy, exclude_list, nan_as_category)
    for col in policy_cat:
        policy_aggregations[col] = ['mean', 'size']
    policy['pdmg_acc'] = pdmg_acc
    policy['qpt'] = qpt
    del pdmg_acc, qpt
    policy_agg = policy.groupby('Policy_Number').agg(policy_aggregations)
    policy_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in policy_agg.columns.tolist()])


    del policy
    gc.collect()
    return policy_agg


df = PreprocessTrainTest(num_rows)
claim_agg = PreprocessClaim(num_rows, nan_as_category)
policy_agg = PreprocessPolicy(num_rows, nan_as_category)
df = df.join(policy_agg, how='left', on='Policy_Number')
df = df.join(claim_agg, how='left', on='Policy_Number', lsuffix='_number')



train = df[df['Next_Premium'].notnull()]
test = df[df['Next_Premium'].isnull()]
    

# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
# Class to extend XGboost classifer


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)



# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }



# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['Next_Premium'].ravel()
train = train.drop(['Next_Premium'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data



# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")


rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)



base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()

x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)


gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)






# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)





def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['Next_Premium'].notnull()]
    test_df = df[df['Next_Premium'].isnull()]
    
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    #del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)
    # Create arrays and dataframes to store results

    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['Policy_Number','Next_Premium',"Insured's_ID",
 'Vehicle_identifier','Prior_Policy_Number',
 'Cancellation',
 'Vehicle_identifier_number',
 'Vehicle_Make_and_Model1',
 'Vehicle_Make_and_Model2',
 'Imported_or_Domestic_Car',
 'Coding_of_Vehicle_Branding_&_Type',
 'qpt',
 'fpt',
 'Main_Insurance_Coverage_Group',
 'Insurance_Coverage',
 'Distribution_Channel',
 'pdmg_acc',
 'fassured',
 'ibirth',
 'fsex',
 'fmarriage',
 'aassured_zip',
 'iply_area',
 'dbirth',
 'fequipment1',
 'fequipment2',
 'fequipment3',
 'fequipment4',
 'fequipment5',
 'fequipment6',
 'fequipment9',
 'nequipment9',
 'Claim_Number',
 'Nature_of_the_claim',
 "Driver's_Gender",
 "Driver's_Relationship_with_Insured",
 'DOB_of_Driver',
 'Marital_Status_of_Driver',
 'Accident_Date',
 'Cause_of_Loss',
 'Coverage',
 'Vehicle_identifier_claim',
 'Claim_Status_(close,_open,_reopen_etc)',
 'Accident_area',
 'number_of_claimants',
 'Accident_Time']]
    
   # feats = [f for f in feats if f not in feature_importance_df[feature_importance_df.importance ==0].feature.tolist()]
    
    seed = 7
    test_size = 0.3
    n_fold = 1
    submission_file_name = "submission_kernel01.csv"
    submission_file_name_agg = "submission_kernel_agg.csv"
    
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Next_Premium'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['Next_Premium'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['Next_Premium'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMRegressor(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.03,
            num_leaves=32,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=40,
            silent=-1,
            verbose=-1)
    
        
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'mae', verbose= 100, early_stopping_rounds= 200)
    

        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)
        sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration_) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d MAE : %.6f' % (n_fold + 1, mean_absolute_error(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % mean_absolute_error(train_df['Next_Premium'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['Next_Premium'] = sub_preds
        NP_aggregations = {'Next_Premium': ['sum']}
        test_df_agg = test_df.groupby('Policy_Number').agg(NP_aggregations) 
        test_df_submit = pd.read_csv('../data/testing-set.csv')
        test_df_submit = test_df_submit.join(test_df_agg, how='left', on='Policy_Number',  rsuffix='_agg')
        test_df_submit.columns = ['Policy_Number','Next_Premium']
        test_df_submit['Next_Premium'][test_df_submit['Next_Premium']<0] = 0 
        test_df_submit.to_csv(submission_file_name_agg, index= False)
        #test_df[['Policy_Number', 'Next_Premium']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df



def main(debug = False):
    num_rows = 10000 if debug else None
    df = PreprocessTrainTest(num_rows)
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)

if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()