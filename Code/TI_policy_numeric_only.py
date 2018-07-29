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
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object' or len(df[col].unique().tolist()) < 20]
    if ("Next_Premium" in categorical_columns): categorical_columns.remove('Next_Premium')
    if ("Policy_Number" in categorical_columns): categorical_columns.remove('Policy_Number')
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

'''
num_rows = None
nan_as_category = False
num_folds = 5
stratified = False
debug = False
'''


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../data/training-set.csv', nrows= num_rows)
    test_df = pd.read_csv('../data/testing-set.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df)#.reset_index()
    policy = pd.read_csv('../data/policy_0702.csv', nrows= num_rows)
    claim = pd.read_csv('../data/claim_0702.csv', nrows= num_rows)
    
    df = df.set_index('Policy_Number')
    policy = policy.set_index('Policy_Number')
    claim = claim.set_index('Policy_Number')
    
    df = df.join(policy, how='left', on='Policy_Number', lsuffix='_number', rsuffix='_policy')
    df = df.join(claim, how='left', on='Policy_Number', lsuffix='_number', rsuffix='_claim')
    
    del policy
    del claim
    del test_df
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    gc.collect()
    return df

def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['Next_Premium'].notnull()]
    test_df = df[df['Next_Premium'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
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
        learning_rate=0.06,
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


    oof_preds = clf.predict(valid_x, num_iteration=clf.best_iteration_)
    sub_preds = clf.predict(test_df[feats], num_iteration=clf.best_iteration_)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('Fold %2d AUC : %.6f' % (n_fold + 1, mean_absolute_error(valid_y, oof_preds)))
    test_df['Next_Premium'] = sub_preds
    #est_df[['Policy_Number', 'Next_Premium']].to_csv(submission_file_name, index= False)
    test_df.to_csv(submission_file_name)


    NP_aggregations = {'Next_Premium': ['mean']}
    test_df_agg = test_df.groupby('Policy_Number').agg(NP_aggregations)

    
    test_df_submit = pd.read_csv('../data/testing-set.csv')
    test_df_submit = test_df_submit.join(test_df_agg, how='left', on='Policy_Number',  rsuffix='_agg')
    test_df_submit.to_csv(submission_file_name_agg, index= False)


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
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)

if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()