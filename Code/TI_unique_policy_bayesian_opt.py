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
import warnings
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from bayes_opt import BayesianOptimization
import datetime
from sklearn.metrics import mean_absolute_error
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

# Preprocess application_train.csv and application_test.csv
def PreprocessClaim(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../data/training-set.csv', nrows= num_rows)
    test_df = pd.read_csv('../data/testing-set.csv', nrows= num_rows)
    try:
        test_df = test_df.drop("Next_Premium", axis=1)
    except:
        0

        
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))

    df = df.append(test_df)#.reset_index()
    claim = pd.read_csv('../data/claim_0702.csv', nrows= num_rows)
    print("Unique policy number in training and testing data: {}".format(len(set(df.loc[:,'Policy_Number']))))
    print("Unique policy number in claim data: {}".format(len(set(claim.loc[:,'Policy_Number']))))

    exclude_list = ["Next_Premium", "Claim_Number", "Policy_Number", "DOB_of_Driver", "Vehicle_identifier"]
 
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
    



    
    claim_aggregations = {'Claim_Number': ['size'],
                          'Paid_Loss_Amount': ['min', 'max', 'mean', 'size'],
                          'paid_Expenses_Amount': ['min', 'max', 'mean', 'size'],
                          'Salvage_or_Subrogation?': ['min', 'max', 'mean', 'size'],
                          'Vehicle_identifier': ['size'],
                          'At_Fault?': ['min', 'max', 'mean', 'size'],
                          'Deductible': ['min', 'max', 'mean', 'size'],
                          #'number_of_claimants': ['min', 'max', 'mean', 'size'],
                          'age_of_Driver': ['min', 'max', 'mean', 'size'],
                           }

    claim_aggregations = {'Claim_Number': ['size'],
                          'Paid_Loss_Amount': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                          'paid_Expenses_Amount': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                          'Salvage_or_Subrogation?': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                          'Vehicle_identifier': ['size'],
                          'At_Fault?': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                          'Deductible': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                          'age_of_Driver': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                          #'number_of_claimants': ['min', 'max', 'mean', 'size'],
                           }
    
    

    

    
    claim, claim_cat = one_hot_encoder(claim, exclude_list, nan_as_category)
    for col in claim_cat:
        claim_aggregations[col] = ['mean']
    claim_agg = claim.groupby('Policy_Number').agg(claim_aggregations)
    claim_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in claim_agg.columns.tolist()])


    

    df = df.join(claim_agg, how='left', on='Policy_Number', rsuffix = "_claim")

    df = df[['Policy_Number','Next_Premium','Claim_Number_SIZE']]

    df['Is_Claim'] = np.isnan(df['Claim_Number_SIZE'])


    for i in claim.columns:
        print("length og unique column =>" + i +":"+str(len(set(claim.loc[:,i]))) + " == type: "+ str(claim[i].dtypes))


    
    del policy
    del claim
    del test_df
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    

    gc.collect()
    return df


# Preprocess application_train.csv and application_test.csv
def PrepreocessPolicy(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../data/training-set.csv', nrows= num_rows)
    test_df = pd.read_csv('../data/testing-set.csv', nrows= num_rows)
    try:
        test_df = test_df.drop("Next_Premium", axis=1)
    except:
        0

        
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))

    df = df.append(test_df)#.reset_index()
    policy = pd.read_csv('../data/policy_0702.csv', nrows= num_rows)
    print("Unique policy number in training and testing data: {}".format(len(set(df.loc[:,'Policy_Number']))))
    print("Unique policy number in policy data: {}".format(len(set(policy.loc[:,'Policy_Number']))))
    
    #policy =  policy_df.drop(["Insured's_ID",'Prior_Policy_Number','Vehicle_identifier','Vehicle_Make_and_Model2','Coding_of_Vehicle_Branding_&_Type','Distribution_Channel','aassured_zip','ibirth','dbirth'], axis = 1)
    #policy =  policy_df.drop(['Vehicle_Make_and_Model2','Coding_of_Vehicle_Branding_&_Type','Distribution_Channel','aassured_zip','ibirth','dbirth'], axis = 1)
    
    exclude_list = ["Next_Premium", "Policy_Number", "Insured's_ID",
                    'Prior_Policy_Number','Vehicle_identifier','Vehicle_Make_and_Model2',
                    'Coding_of_Vehicle_Branding_&_Type','Distribution_Channel',
                    'aassured_zip','ibirth','dbirth']
 
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






policy.columns
    policy['Manafactured_Year_and_Month']

Insured_Amount1
Insured_Amount2
Insured_Amount2

Premium

Engine_Displacement_(Cubic_Centimeter)                                            

Replacement_cost_of_insured_vehicle


    policy_aggregations = {'Premium': ['min', 'max', 'mean', 'size'],
                           'Manafactured_Year_and_Month': ['min', 'max', 'mean', 'size'],
                           'Engine_Displacement_(Cubic_Centimeter)': ['min', 'max', 'mean', 'size'],
                           #'qpt': ['min', 'max', 'mean', 'size'],
                           'Insured_Amount1': ['min', 'max', 'mean', 'size'],
                           'Insured_Amount2': ['min', 'max', 'mean', 'size'],
                           'Insured_Amount3': ['min', 'max', 'mean', 'size'],
                           'Premium': ['min', 'max', 'mean', 'size'],
                           'Replacement_cost_of_insured_vehicle': ['min', 'max', 'mean', 'size'],
                           'Multiple_Products_with_TmNewa_(Yes_or_No?)': ['min', 'max', 'mean', 'size'],
                           'lia_class': ['min', 'max', 'mean', 'size'],
                           'plia_acc': ['min', 'max', 'mean', 'size'],
                           #'pdmg_acc': ['min', 'max', 'mean', 'size']
                           "Insured's_ID": ['size'],
                           'Prior_Policy_Number': ['size'],
                           'Vehicle_identifier': [ 'size'],
                           }
    
    

    policy_aggregations = {'Premium': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'Manafactured_Year_and_Month': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'Engine_Displacement_(Cubic_Centimeter)': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       #'qpt': ['min', 'max', 'mean', 'size'],
                       'Insured_Amount1': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'Insured_Amount2': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'Insured_Amount3': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'Premium': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'Replacement_cost_of_insured_vehicle': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'Multiple_Products_with_TmNewa_(Yes_or_No?)': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'lia_class': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'plia_acc': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       #'pdmg_acc': ['min', 'max', 'mean', 'size']
                       'iage': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'dage': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'Insured_Amount_sum': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'Insured_Amount_mean' : ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'Insured_Amount_sum_prmium_ratio': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       'Insured_Amount_mean_prmium_ratio': ['min', 'max', 'mean', 'size', 'median', 'std', 'sum'],
                       "Insured's_ID": ['size'],
                       'Prior_Policy_Number': ['size'],
                       'Vehicle_identifier': [ 'size'],
                       }
    

    

    policy, policy_cat = one_hot_encoder(policy, exclude_list, nan_as_category)
    for col in policy_cat:
        policy_aggregations[col] = ['mean']
    policy_agg = policy.groupby('Policy_Number').agg(policy_aggregations)
    policy_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in policy_agg.columns.tolist()])





    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    

    

    df = df.join(policy_agg, how='left', on='Policy_Number')
    
    df = df.join(claim_agg, how='left', on='Policy_Number', lsuffix='_number')

    
    df['Is_Claim'] = np.isnan(df['Claim_Number_SIZE'])




    policy_aggregations_count = {'Policy_Number': ['size']}
    
    # Insured's_ID
    InsuredID_agg = policy_df.groupby("Insured's_ID").agg(policy_aggregations_count)

    # Prior_Policy_Number
    ## Nan -> if prior & prior renew policy count 
    PriorPolicy_agg = policy_df.groupby('Prior_Policy_Number').agg(policy_aggregations_count)
    ## TODO: prior premium and prior times
    
    # Cancellation
    ## TDOO: one hot encoding
    
    # Vehicle_identifier
    Vehicle_agg = policy_df.groupby('Vehicle_identifier').agg(policy_aggregations_count)

    # Vehicle_Make_and_Model1
    ## TDOO: one hot encoding
    
    # Vehicle_Make_and_Model2
    ## TDOO: one hot encoding or skip 
    
    # Manafactured_Year_and_Month
    ## new feature
    
    # Engine_Displacement_(Cubic_Centimeter)
    ## new feature

    # Imported_or_Domestic_Car
    ## TDOO: one hot encoding
    
    # Coding_of_Vehicle_Branding_&_Type
    ## TDOO: one hot encoding or skip 
    
    # apt
    ## new feature
    
    # fpt
    ## skip
    
    # Main_Insurance_Coverage_Group
    ## TDOO: one hot encoding
    
    # Insurance_Coverage
    ## TDOO: one hot encoding 
    
    # Insured_Amount1
    ## new feature
    
    # Insured_Amount2
    ## new feature
    
    # Insured_Amount3
    ## new feature
    
    # Coverage_Deductible_if_applied
    ## TDOO: one hot encoding 
    
    # Premium
    ## new feature
    
    # Replacement_cost_of_insured_vehicle
    ## new feature
    
    # Distribution_Channel
    ## TDOO: one hot encoding or skip 
    
    # Multiple_Products_with_TmNewa_(Yes_or_No?)
    ## new feature
    
    # lia_class
    ## new feature
    
    # plia_acc
    ## new feature
    
    # pdmg_acc
    ## new feature
    
    # fassured
    ## TDOO: one hot encoding
    
    # ibirth
    ## TODO: change to age
    
    # fsex
    ## TDOO: one hot encoding
    
    # fmarriage
    ## TDOO: one hot encoding
    
    # aassured_zip
    ## TDOO: one hot encoding or skip 
    
    # iply_area
    ## TDOO: one hot encoding 
    
    # dbirth
    ## TODO: change to age
     
    # fequipment1~9
     

lstpolicywithprior = list(set(policy['Policy_Number']) & set(policy['Prior_Policy_Number']))
v = [a in lstpolicywithprior for a in policy['Policy_Number'] ]
b = list(set(policy.loc[v]['Prior_Policy_Number']))

v2 = [a in b for a in policy['Policy_Number'] ]
b2 = list(set(policy.loc[v2]['Prior_Policy_Number']))


v3 = [a in b2 for a in policy['Policy_Number'] ]
b3 = list(set(policy.loc[v3]['Prior_Policy_Number']))




policy.loc[policy['Policy_Number'] == '00d2f2329cb04917c70cc30b3d44f3888452044f']['Prior_Policy_Number']





for i in policy.columns:
    print("length og unique column =>" + i +":"+str(len(set(policy.loc[:,i]))) + "type: "+ str(policy[i].dtypes))





    claim_aggregations = {'Paid_Loss_Amount': ['min', 'max', 'mean', 'size']}
    claim_agg = claim.groupby('Policy_Number').agg(claim_aggregations)


    df = df.set_index('Policy_Number')
    policy = policy.set_index('Policy_Number')
    claim = claim.set_index('Policy_Number')
    
    df = df.join(policy, how='left', on='Policy_Number', lsuffix='_number', rsuffix='_policy')
    df = df.join(claim, how='left', on='Policy_Number', lsuffix='_number', rsuffix='_claim')
    
    
    a = policy.loc['000061141b237e8619efedcd6939fddeff05b9a5']
    c = df.loc['000061141b237e8619efedcd6939fddeff05b9a5']
    b = policy.loc['8e31b2f7864ddeac1be3a31766b0bf2c54908d37'] # NP=4414

    
    
    del policy
    del claim
    del test_df
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    

    gc.collect()
    return df


train_df = df[df['Next_Premium'].notnull()].iloc[1:50]
X = train_df.drop('Next_Premium', axis=1)
X = train_df.drop('Policy_Number', axis=1)
y = train_df.Next_Premium



def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight, max_bin):
    params = {'objective': 'regression','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'mae'}
    params["num_leaves"] = int(round(num_leaves))
    params['feature_fraction'] = max(min(feature_fraction, 1), 0)
    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    params['max_depth'] = int(round(max_depth))
    params['lambda_l1'] = max(lambda_l1, 0)
    params['lambda_l2'] = max(lambda_l2, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    params['max_bin'] = int(round(max_bin))
    cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['mae'])
    return min(cv_result['l1-mean'])

init_round=15
opt_round=25
n_folds=5
random_seed=6
n_estimators=10000
learning_rate=0.05
output_process=False


lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50),
                                           'max_bin': (5, 50)}, random_state=0)

lgbBO.maximize(init_points=init_round, n_iter=opt_round)


def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05, output_process=False):
    # prepare data
    categorical_feats = None
    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight, max_bin):
        params = {'objective': 'regression','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'mae'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['max_bin'] = int(round(max_bin))
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['mae'])
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


train_df = df[df['Next_Premium'].notnull()]    
X = train_df.drop('Next_Premium', axis=1)
X = train_df.drop('Policy_Number', axis=1)
y = train_df.Next_Premium




opt_params = bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=10, n_folds=3, random_seed=6, n_estimators=100, learning_rate=0.05, output_process = True)
print(opt_params)





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
 'Prior_Policy_Number',
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

    seed = 7
    test_size = 0.3
    n_fold = 1
    submission_file_name = "submission_kernel01.csv"
    submission_file_name_agg = "submission_kernel_agg.csv"
    train_x, valid_x, train_y, valid_y = cross_validation.train_test_split(train_df[feats], train_df['Next_Premium'], test_size=test_size, random_state=seed)


    clf = LGBMRegressor(
        nthread=4,
        n_estimators=10000,
        learning_rate=0.01,
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
    
    
    clf = LGBMRegressor(
        nthread=4,
        n_estimators=10000,
        learning_rate=0.03,
        num_leaves=32,
        colsample_bytree=0.18692095349417298,
        subsample=0.8010822837533205,
        max_depth=7,
        reg_alpha=0.7823971863704173,
        reg_lambda=0.02471099928294551,
        min_split_gain=0.009640523204557641,
        min_child_weight=36,
        silent=-1,
        verbose=-1)
    
    {'num_leaves': 33.255805943929886, 
     'feature_fraction': 0.18692095349417298, 
     'bagging_fraction': 0.8010822837533205, 
     'max_depth': 7.349330616930791, 
     'lambda_l1': 0.7823971863704173, 
     'lambda_l2': 0.02471099928294551, 
     'min_split_gain': 0.009640523204557641, 
     'min_child_weight': 36.328299627807326, 
     'max_bin': 30.62253929738296}
    

    
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
        eval_metric= 'mae', verbose= 100, early_stopping_rounds= 500)




    oof_preds = clf.predict(valid_x, num_iteration=clf.best_iteration_)
    sub_preds = clf.predict(test_df[feats], num_iteration=clf.best_iteration_)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    display_importances(feature_importance_df)



    
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, mean_absolute_error(valid_y, oof_preds)))
    test_df['Next_Premium'] = sub_preds
    #est_df[['Policy_Number', 'Next_Premium']].to_csv(submission_file_name, index= False)
    test_df.to_csv(submission_file_name)


    NP_aggregations = {'Next_Premium': ['mean']}
    test_df_agg = test_df.groupby('Policy_Number').agg(NP_aggregations)


    
    test_df_submit = pd.read_csv('../data/testing-set.csv')
    test_df_submit = test_df_submit.join(test_df_agg, how='left', on='Policy_Number',  rsuffix='_agg')
    test_df_submit.to_csv(submission_file_name_agg, index= False)

    

   oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    
     for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Next_Premium'])):
         print(n_fold)
         time.sleep(1)
         
         
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
        NP_aggregations = {'Next_Premium': ['mean']}
        test_df_agg = test_df.groupby('Policy_Number').agg(NP_aggregations) 
        test_df_submit = pd.read_csv('../data/testing-set.csv')
        test_df_submit = test_df_submit.join(test_df_agg, how='left', on='Policy_Number',  rsuffix='_agg')
        test_df_submit.columns = ['Policy_Number','Next_Premium']
        test_df_submit['Next_Premium'][test_df_submit['Next_Premium']<0] = 0 
        test_df_submit.to_csv(submission_file_name_agg, index= False)
        #test_df[['Policy_Number', 'Next_Premium']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df





# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('../input/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('../input/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': [ 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': [ 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        #'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('../input/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('../input/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('../input/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': [ 'mean',  'var'],
        'PAYMENT_DIFF': [ 'mean', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('../input/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
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
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)

if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()