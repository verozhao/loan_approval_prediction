#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/gauravbrills/kaggle-fiddle/blob/main/amex/rapids_cudf_feature_engineering_xgb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ### In this notebook, we use [RAPIDS cudf](https://github.com/rapidsai/cudf) with XGB

import os,random 
import tqdm
#import dask.dataframe as cudf
import pandas as cudf
import numpy as cupy 
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import joblib
from joblib import Parallel, delayed
import pathlib
import tqdm
from sklearn.model_selection import StratifiedKFold

class CFG:
  seed = 42
  INPUT = "./input"
  TRAIN = True
  OPTIMIZE = False
  PERM_IMP = True
  INFER = True
  n_folds = 5
  target ='target'
  DEBUG= False
  ADD_CAT = True
  ADD_LAG = True
  COMPUTE_Z = False
  ADD_DIFF_1 = True
  ADD_DIFF =  [2]
  ADD_PCTDIFF = []
  TG_ENCODING = False
  SPLIT_CAT = False
  KURT = False
  MAD = False
  TRIM= True
  ADD_RATIOS = False
  ADD_MIDDLE = True
  ADD_SIRU = True

  model_dir = "./output/models/xgb"
  folds_to_train = [0,1,2,3,4]

path = f'{CFG.INPUT}/amex-data-integer-dtypes-parquet-format'   

# ====================================================
# Seed everything
# ====================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(CFG.seed)    


# ### Feature Engineering
def agg_df_num(df):
    df_agg = df.groupby('customer_ID').agg(f_names)
    df_agg.columns = [str(c[0])+'_'+str(c[1]) for c in df_agg.columns]
    return df_agg

# ====================================================
# Get the difference  --> capture fluctuations, can capture diff(1),diff(2),diff(3) and consider adding features
# ====================================================
def get_difference(data, num_features,period=1): 
    df1 = []
    customer_ids = []
    for customer_id, df in  data.groupby(['customer_ID']):
        # Get the differences
        diff_df1 = df[num_features].diff(period).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    # Concatenate
    df1 = np.concatenate(df1, axis = 0)
    # Transform to dataframe
    df1 = pd.DataFrame(df1, columns = [col + f'_diff{period}' for col in df[num_features].columns])
    # Add customer id
    df1['customer_ID'] = customer_ids
    return df1

def get_pct_change(data, num_features,period=1): 
    df1 = []
    customer_ids = []
    for customer_id, df in  data.groupby(['customer_ID']):
        # Get the differences
        diff_df1 = df[num_features].pct_change(period,fill_method=None).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    # Concatenate
    df1 = np.concatenate(df1, axis = 0)
    # Transform to dataframe
    df1 = pd.DataFrame(df1, columns = [col + f'_pct_chg{period}' for col in df[num_features].columns])
    # Add customer id
    df1['customer_ID'] = customer_ids
    return df1


def kurtosis(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)    
 

CID ="customer_ID"
TIME = "S_2"
TARGET = "target"
def pivot_data(df, train=True):
    cols = [c for c in df.columns if c not in [CID, TIME, TARGET]]
    tmp = df.copy()
    tmp['max'] = tmp.groupby([CID])[TIME].transform('max')
    tmp['size'] = tmp.groupby([CID])[TIME].transform('size')
    tmp['rank'] = tmp.groupby([CID])[TIME].transform('rank')
    tmp['statement'] = (tmp['size']-tmp['rank']).astype(np.int8)
    pivot_pd = tmp.pivot(index=CID,columns=['statement'],values=cols)
    pivot_pd.columns = [('{0}__TE{1}'.format(*tup)) for tup in pivot_pd.columns]
    pivot_pd = pivot_pd.reset_index()
    return pivot_pd

def agg_pct_rank_by_cat(df,main_features_last,cat_features_last):
    df_list = [] 
    for c in cat_features_last:
        df_agg = df[main_features_last].groupby(df[c]).transform('rank')/df[main_features_last].groupby(df[c]).transform('count')
        df_agg.columns = [f+'_pct_rank_by_'+c for f in df_agg.columns]
        df_list.append(df_agg.astype('float16')) 
    return pd.concat([df,pd.concat(df_list,axis=1).astype('float16')], axis=1)
def agg_global_rank(df,main_features_last):
    df_rank = df[main_features_last].transform('rank')
    df_rank.columns = [s+'_global_rank' for s in df_rank.columns]
    return pd.concat([df,(df_rank/len(df)).astype('float16')],axis=1)

def agg_standardize_by_cat(df,main_features_last,cat_features_last):
    df_list = []
    for c in cat_features_last:
        df_agg = df[main_features_last].groupby(df[c]).transform(lambda x: (x - x.mean()) / x.std())
        df_agg.columns = [f+'_standardized_by_'+c for f in df_agg.columns]
        df_list.append(df_agg.astype('float16'))

    return pd.concat([df,pd.concat(df_list,axis=1).astype('float16')],axis=1)

def get_not_used():  
  return ['row_id', 'customer_ID', 'target', 'cid', 'S_2','D_103','D_139']

def get_not_used():  
  return ['row_id', 'customer_ID', 'target', 'cid', 'S_2','D_103','D_139']

stats = ['mean', 'min', 'max','std']
features_avg = ['S_2_wk','B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18',
                'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_28', 'B_29', 'B_30', 'B_32', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42',
                'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_50', 'D_51', 'D_53', 'D_54', 'D_55', 'D_58', 'D_59', 'D_60', 'D_61', 
                'D_62', 'D_65', 'D_66', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_75', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_86', 'D_91', 
                'D_92', 'D_94', 'D_96', 'D_103', 'D_104', 'D_108', 'D_112', 'D_113', 'D_114', 'D_115', 'D_117', 'D_118', 'D_119', 'D_120', 'D_121', 'D_122', 'D_123',
                'D_124', 'D_125', 'D_126', 'D_128', 'D_129', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135', 'D_136', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145',
                'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_14', 'R_15', 'R_16', 'R_17', 'R_20', 'R_21', 'R_22', 'R_24', 
                'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_9', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_18', 'S_22', 'S_23', 'S_25', 'S_26']
features_min = ['B_2', 'B_4', 'B_5', 'B_9', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_19', 'B_20', 'B_28', 'B_29', 'B_33', 'B_36', 'B_42', 'D_39',
                'D_41', 'D_42', 'D_45', 'D_46', 'D_48', 'D_50', 'D_51', 'D_53', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70', 'D_71', 'D_74', 
                'D_75', 'D_78', 'D_83', 'D_102', 'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_128', 'D_132', 'D_140', 'D_141', 'D_144',
                'D_145', 'P_2', 'P_3', 'R_1', 'R_27', 'S_3', 'S_5', 'S_7', 'S_9', 'S_11', 'S_12', 'S_23', 'S_25']
features_std = ["B_4","R_1","B_38","B_40","B_3","D_39","D_44","S_25","D_75","D_43","B_23","P_2","D_70","S_26","R_3","B_19","D_65",
                "S_15","B_28","B_30","B_6","B_9","S_12","D_55","S_23","D_48","D_61","D_133","B_17","D_144","D_42","D_74",
                "S_11","S_7","S_13","D_58","D_104","S_3","P_3","D_121","B_13","B_20","S_22","B_22","B_16","B_5","D_45","D_62",
                "B_12","B_24","D_117","D_47","B_8","S_5","D_60","D_59","P_4","B_11","D_119","D_115","D_113","D_46",
                "R_16","S_2_wk","B_2","B_1","R_27","B_14","D_78","B_18","D_124","D_126","B_21","D_142","D_131","D_136","D_71",
                "B_37","D_53","S_9","D_112","D_118","D_77","B_15","R_10","D_80","R_11",
                "D_120","S_16","D_128","B_33","S_6","B_10","D_122","D_132","D_69","B_25","D_41","D_141"]                 
features_max = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19',
                'B_21', 'B_23', 'B_24', 'B_25', 'B_29', 'B_30', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44',
                'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_52', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_61', 'D_63', 'D_64', 'D_65', 'D_70',
                'D_71', 'D_72', 'D_73', 'D_74', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_91', 'D_102', 'D_105', 'D_107', 'D_110', 'D_111', 'D_112',
                'D_115', 'D_116', 'D_117', 'D_118', 'D_119', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_128', 'D_131', 'D_132', 'D_133',
                'D_134', 'D_135', 'D_136', 'D_138', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_3', 'R_5', 'R_6', 'R_7',
                'R_8', 'R_10', 'R_11', 'R_14', 'R_17', 'R_20', 'R_26', 'R_27', 'S_3', 'S_5', 'S_7', 'S_8', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_22',
                'S_23', 'S_24', 'S_25', 'S_26', 'S_27']
features_last = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 
                 'B_18', 'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_26', 'B_28', 'B_29', 'B_30', 'B_32', 'B_33', 'B_36', 'B_37', 'B_38',
                 'B_39', 'B_40', 'B_41', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_51', 'D_52',
                 'D_53', 'D_54', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_61', 'D_62', 'D_63', 'D_64', 'D_65', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_75', 'D_76', 'D_77', 'D_78', 'D_79', 'D_80', 'D_81', 'D_82', 'D_83', 'D_86', 'D_91', 'D_96', 'D_105', 'D_106', 'D_112', 'D_114', 'D_119', 'D_120', 'D_121', 'D_122', 'D_124', 'D_125', 'D_126', 'D_127', 'D_130', 'D_131', 'D_132', 'D_133', 'D_134', 'D_138', 'D_140', 'D_141', 'D_142', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_12', 'R_13', 'R_14', 'R_15', 'R_19', 'R_20', 'R_26', 'R_27', 'S_3',
                 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_11', 'S_12', 'S_13', 'S_16', 'S_19', 'S_20', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']
features_mad = ['D_43','B_4','D_39','P_3','B_3','S_3','B_40','D_61','D_48','D_49','S_7','B_17','D_77','P_2','S_27','B_7','D_62','D_44','S_12',
                'D_105','D_52','S_26','D_46',
                'S_8','S_9','B_23','D_65','D_47','D_55','S_15','S_11','D_75','D_53','B_13','D_121','S_24','D_115','D_50','B_28','S_22','D_58',
                'S_5','D_59','B_5','S_23','D_45','D_70','S_13','D_74','D_60','D_119']
features_mad = ['D_43','B_4','D_39','P_3','B_3','S_3','B_40','D_61','D_48','D_49','S_7','B_17','D_77','P_2','S_27','B_7','D_62','D_44','S_12',
                'D_105','D_52','S_26','D_46']                
# Feature Engineering on credit risk
spend_p=[ 'S_3',  'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_17', 'S_18', 'S_19', 'S_20', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']
balance_p = ['B_1', 'B_2', 'B_3',  'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15',  'B_17', 'B_18',  'B_21',   'B_23', 'B_24', 'B_25', 'B_26', 'B_27', 'B_28',  'B_36', 'B_37',  'B_40',    ]
payment_p = ['P_2', 'P_3', 'P_4']
cat_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120',
            'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
delq = ['D_39',
                'D_41', 'D_42', 'D_45', 'D_46', 'D_48', 'D_50', 'D_51', 'D_53', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70', 'D_71', 'D_74', 
                'D_75', 'D_78', 'D_83', 'D_102', 'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_128', 'D_132', 'D_140', 'D_141', 'D_144',
                'D_145']           
zero_imp = ['B_42_pct_chg5','D_108_first','D_109','D_109-mean','D_109_diff1','D_109_mean','D_111_diff1','D_111_first','D_111_pct_chg1','D_111_pct_chg5',
            'D_116_first','D_64_count','D_68_count','D_82_diff1','D_86_diff1','D_87','D_87-mean','D_87_diff1','D_87_first','D_87_mean','D_88','D_88-mean',
            'D_88_diff1','D_92_diff1','D_93','D_93-mean','D_93_diff1','D_93_first','D_94_diff1','R_17_first','R_18','R_18-mean','R_18_diff1','R_18_first','R_18_mean',
            'R_19_diff1','R_23','R_23-mean',
            'R_23_diff1','R_23_first','R_23_mean','R_25_first','R_26_diff1','R_28','R_28_diff1','R_28_first','S_18'] 

cat_cols_avg = [col for col in cat_cols if col in features_avg]
not_used = get_not_used()            
g_num_cols = []

def calc_mad(col,df):
  train_stat = df.groupby(CID)[col].agg(['mad'])
  train_stat.columns = [f"{col}_mad" for x in train_stat.columns]
  print("Mad Calculated for ",train_stat.columns)
  train_stat.reset_index(inplace = True)
  return train_stat    
  #dgs.append(train_stat) 

def preprocess(df):
    df['row_id'] = cupy.arange(df.shape[0])
    not_used = get_not_used()
    # Drop cols https://www.kaggle.com/code/raddar/redundant-features-amex/notebook
    df=df.drop(["D_103","D_139"],axis=1)
    num_cols = [col for col in df.columns if col not in cat_cols+not_used]   

    globals()['g_num_cols'] = num_cols
    for col in df.columns:
        if col not in not_used+cat_cols:
           df[col] = df[col].astype('float32').round(decimals=2).astype('float16') 
    print(f"Starting fe [{len(df.columns)}]") 
    dgs=add_stats_step(df, num_cols) 
    #custom stats    
    if CFG.MAD:
      results = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(calc_mad)(col,df) for col in features_mad )  
      for result in results:
        dgs.append( result)  
      print(f"Stats MAD calculated [{len(features_mad)}]")   

    train_stat = df.groupby("customer_ID")[spend_p+payment_p+delq+balance_p].agg('sum')
    train_stat.columns = [x+'_sum' for x in train_stat.columns]
    print(train_stat.columns)
    train_stat.reset_index(inplace = True)    
    dgs.append(train_stat)
    del train_stat; gc.collect() 
    print(f"Stats Sum calc [{len(df.columns)}]")    
    # END Custom cherry picked Stats
    print(f"Stats added and calculated [{len(df.columns)}]")     

    df["P_SUM"] = df[payment_p].sum(axis=1) 
    df["S_SUM"] = df[spend_p].sum(axis=1) 
    df["B_SUM"] = df[balance_p].sum(axis=1)
    df["P-S"] = df.P_SUM - df.S_SUM       
    df["P-B"] = df.P_SUM - df.B_SUM
    df=df.drop(["S_SUM","P_SUM","B_SUM"],axis=1)
    print(f"P-S feature added")      
    # Agg on P-S sum 
    #train_stat = df.groupby("customer_ID")["P-S"].agg(['sum','mean','std'])
    #train_stat.columns = [x+'_sum' for x in train_stat.columns]
    #print(train_stat.columns)
    #train_stat.reset_index(inplace = True)    
    #dgs.append(train_stat)
    #del train_stat; gc.collect() 
    #print(f"Stats Sum calc [{len(df.columns)}]")   
    if CFG.ADD_MIDDLE:
      df_middle = df.groupby([CID])[num_cols+cat_cols].apply(lambda x: x.iloc[(len(x)+1)//2 if len(x) > 1 else 0])   
      df_middle.columns = [x+'_mid' for x in df_middle.columns]  
      dgs.append(df_middle)
      print(f"Mid Cols added [{len(df_middle.columns)}]")    
      del df_middle; gc.collect() 

    # Add Lag Columns 
    if CFG.ADD_LAG:
      train_num_agg = df.groupby("customer_ID")[num_cols].agg(['first', 'last'])#payment_p+balance_p+spend_p
      train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
      train_num_agg.reset_index(inplace = True) 
      for col in train_num_agg:
        if 'last' in col and col.replace('last', 'first') in train_num_agg:
                    train_num_agg[col + '_lag_sub'] = train_num_agg[col] - train_num_agg[col.replace('last', 'first')]
                    train_num_agg[col + '_lag_div'] = train_num_agg[col] / train_num_agg[col.replace('last', 'first')]            
      train_num_agg.drop([col for col in train_num_agg.columns if "last" in col],axis=1, inplace=True)
      dgs.append(train_num_agg)
      del train_num_agg
      # get_difference():
      print(f"Computing diff 1 features ,curr cols [{len(df.columns)}]") 
      dff_cols =  payment_p+balance_p+spend_p+delq ## Replace with num_cols
      if CFG.ADD_DIFF_1:
        train_diff = df.loc[:,num_cols+['customer_ID']].groupby(['customer_ID']).apply(lambda x: cupy.diff(x.values[-2:,:], axis = 0).squeeze().astype(cupy.float32))
        index = train_diff.index
        cols = [col + '_diff1' for col in df[num_cols].columns]
        train_diff = pd.DataFrame(train_diff.values.tolist(), columns=cols)   
        train_diff['customer_ID'] = index    
        train_diff.reset_index(inplace = True) 
        print(f"Computing diff 1 features ,curr cols [{ train_diff.columns}]")
        dgs.append(train_diff) 
      for pdf in CFG.ADD_DIFF:
        train_diff = get_difference(df, dff_cols,period=pdf)
        print(f"Computing Diff {pdf} ,curr cols [{ train_diff.columns}]") 
        dgs.append(train_diff)    
        del train_diff; gc.collect()        
      for pdf in CFG.ADD_PCTDIFF:
        train_diff = get_pct_change(df, dff_cols,period=pdf)
        print(f"Computing pct change {pdf} ,curr cols [{ train_diff.columns}]") 
        dgs.append(train_diff)    
        del train_diff; gc.collect() 
      print(f"Lag Features added [{len(df.columns)}]")          
    
    # compute "after pay" features
    for bcol in [f'B_{i}' for i in [11,14,17]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]:
        for pcol in ['P_2','P_3']:
            if bcol in df.columns:
                df[f'{bcol}-{pcol}'] = df[bcol] - df[pcol]
    #
    df['S_2'] = cudf.to_datetime(df['S_2'])
    df['cid'], _ = df.customer_ID.factorize()    

    # Add sundays count as a feature 
    s2_count = df[df.S_2.dt.dayofweek == 6].groupby("customer_ID")['S_2'].agg(['count']) 
    s2_count.columns = ['S_2_Sun_Count']
    s2_count.reset_index(inplace = True)     
    dgs.append(s2_count)
    print(f"sundays count added and calculated [{len(s2_count.columns)}]")    
    # Add week of the month correlation
    df['S_2_wk'] =  df['S_2'].dt.week
    s2_count = df.groupby("customer_ID")['S_2_wk'].agg(['std'])  
    s2_count.columns = ['S_2_wk_std']
    s2_count.reset_index(inplace = True)     
    dgs.append(s2_count)
    df=df.drop(["S_2_wk"],axis=1 )
    print(f"sundays count added and calculated [{len(s2_count.columns)}]")        
    del s2_count; gc.collect()     

    ## Flatten Categoricals
    if CFG.SPLIT_CAT:
      for CAT_COL in cat_cols:
        X_train_cat_pd = df[[CAT_COL,TIME,CID]] 
        te_fold = df.groupby([CID,TIME]).cumcount().to_dict() 
        X_train_pivot = pivot_data(X_train_cat_pd[[CID, TIME, CAT_COL]])
        display(X_train_pivot.head(1))
        ##  change types 
        cat_features= [col for col in X_train_pivot.columns if "__TE" in col ]
        X_train_pivot[cat_features] = X_train_pivot[cat_features].values.astype(int)
        dgs.append(X_train_pivot)
        print(f"Adding flattened cat_col {CAT_COL}")
        del X_train_pivot; gc.collect()   
      del X_train_cat_pd


    if CFG.ADD_CAT:
      train_cat_agg = df.groupby("customer_ID")[cat_cols].agg(['count', 'nunique', 'std','first']) 
      train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]
      train_cat_agg.reset_index(inplace = True)     
      dgs.append(train_cat_agg)
      del train_cat_agg; gc.collect() 
      train_cat_mean = df.groupby("customer_ID")[cat_cols_avg].agg(['mean']) 
      train_cat_mean.columns = ['_'.join(x) for x in train_cat_mean.columns]
      train_cat_mean.reset_index(inplace = True)    
      print(f"Added cat mean cols [{train_cat_mean.columns}]")   
      dgs.append(train_cat_mean)
      del train_cat_mean; gc.collect() 
      print(f"CAT features added {len(df.columns)}") 
    # Add s2 count as a feature ( Number of spends)
    s2_count = df.groupby("customer_ID")['S_2'].agg(['count']) 
    s2_count.columns = ['S_2_Count']
    s2_count.reset_index(inplace = True)     
    df = df.merge(s2_count, on='customer_ID', how='inner')
    print(f"Stats added and calculated [{len(s2_count.columns)}]")    
    if CFG.ADD_MIDDLE:
      df_middle = df[df.S_2_Count > 2].groupby(['customer_ID'])[balance_p+payment_p+delq+spend_p].apply(lambda x: x.iloc[(len(x)+1)//2])   
      df_middle.columns = [x+'_mid' for x in df_middle.columns]  
      dgs.append(df_middle) 
      print(f"Mid Cols added [{len(df_middle.columns)}]")    
      del df_middle; gc.collect()  
    del s2_count; gc.collect()   
    # cudf merge changes row orders
    # restore the original row order by sorting row_id
    df = df.sort_values('row_id')
    df = df.drop(['row_id'],axis=1)
    return df, dgs

def add_stats_step(df, cols):
    n = 50
    dgs = []
    for i in range(0,len(cols),n):
        s = i
        e = min(s+n, len(cols))
        dg = add_stats_one_shot(df, cols[s:e])
        dgs.append(dg)
    return dgs

def add_stats_one_shot(df, cols):
    
    dg = df.groupby('customer_ID').agg({col:stats for col in cols})
    out_cols = []
    for col in cols:
        out_cols.extend([f'{col}_{s}' for s in stats])
    dg.columns = out_cols
    dg = dg.reset_index()
    return dg

def load_test_iter(path, chunks=4):
    
    test_rows = 11363762
    chunk_rows = test_rows // chunks
    
    test = cudf.read_parquet(f'{path}/test.parquet',
                             columns=['customer_ID','S_2'],
                             num_rows=test_rows)
    test = get_segment(test)
    start = 0
    while start < test.shape[0]:
        if start+chunk_rows < test.shape[0]:
            end = test['cus_count'].values[start+chunk_rows]
        else:
            end = test['cus_count'].values[-1]
        end = int(end)
        df = cudf.read_parquet(f'{path}/test.parquet',
                               num_rows = end-start, skiprows=start)
        start = end
        yield process_data(df)
    

def load_train(path):
    train = cudf.read_parquet(f'{path}/train.parquet')
    train = process_data(train)
    if CFG.ADD_SIRU:
      ds_s = cudf.read_csv(f"{CFG.INPUT}/feature0809/train_flat_1_train.csv", index_col = False)
      print("Merge Sirius data",len(ds_s),len(train))
      train = train.merge(ds_s, on='customer_ID', how='left')
      
      
    trainl = cudf.read_csv(f'{CFG.INPUT}/train_labels.csv')
    trainl.target = trainl.target.astype('int8')
    train = train.merge(trainl, on='customer_ID', how='left')
    
    return train

def process_data(df):
    df,dgs = preprocess(df) 
    df = df.drop_duplicates('customer_ID',keep='last')
    for dg in dgs:
        df = df.merge(dg, on='customer_ID', how='left')
        # drop specific non impactful cols 
    del dgs; gc.collect()    
    if CFG.TRIM:
      #drop_col = [col for  col in df.columns if (("std" in col) and (col.replace("_std","") not in features_std))]
      #print(f"Dropping {drop_col}")
      #df=df.drop(drop_col,axis=1)      
      drop_col = [col for  col in df.columns if (("min" in col) and (col.replace("_min","") not in features_min))]
      print(f"Dropping {drop_col}")
      df=df.drop(drop_col,axis=1)
      drop_col = [col for  col in df.columns if  (("max" in col) and (col.replace("_max","") not in features_max))]
      print(f"Dropping {drop_col}")
      df=df.drop(drop_col,axis=1)
      #drop_col = [col for  col in df.columns if  (("median" in col) and (col.replace("_median","") not in features_avg))]
      #print(f"Dropping {drop_col}")
      #df=df.drop(drop_col,axis=1)      
      #drop_col = [col for  col in df.columns if  (("mean" in col) and (col.replace("_mean","") not in features_avg))]
      #print(f"Dropping {drop_col}")
      #df=df.drop(drop_col,axis=1)
      #drop_col = [col for  col in df.columns if  (("sum" in col) and (col.replace("_sum","") not in features_min))]
      #print(f"Dropping {drop_col}")       
      #df=df.drop(drop_col,axis=1)                 
    diff_cols = [col for col in df.columns if col.endswith('_diff')]
    df = df.drop(diff_cols,axis=1)
    print(f"All stats merged {len(df.columns)}")   
  
    math_col = globals()['g_num_cols'] 
    print("Added more lags")
    # More Lag Features
    for col in spend_p+payment_p+balance_p:
        for col_2 in ['min','max']: 
          if f"{col}_{col_2}" in df.columns:
              df[f'{col}_{col_2}_lag_sub'] = df[f"{col}_{col_2}"] - df[col]
              df[f'{col}_{col_2}_lag_div'] = df[f"{col}_{col_2}"] / df[col] 
    print("Added more lags")
    df["P2B9"] = df["P_2"] / df["B_9"] 
    math_col = globals()['g_num_cols']
    for pcol in math_col:
      if pcol+"_mean" in df.columns:
        df[f'{pcol}-mean'] = df[pcol] - df[pcol+"_mean"]
        df[f'{pcol}-div-mean'] = df[pcol] /df[pcol+"_mean"]
      if (pcol+"_min" in df.columns) and (pcol+"_max" in df.columns):  
        df[f'{pcol}_min_div_max'] = df[pcol+"_min"] / df[pcol+"_max"]
        df[f'{pcol}_min-max'] = df[pcol+"_min"] - df[pcol+"_max"]
      # compute z score 
      if CFG.COMPUTE_Z:
        if (pcol+"_mean" in df.columns) and (pcol+"_std" in df.columns):
          df[f'{pcol}-zscore'] = (df[pcol] - df[pcol+"_mean"])/df[pcol+"_std"]
          # Remove last as standardized
          #df = df.drop(pcol,axis=1)
    print(f"Addding col-mean {len(df.columns)} cols {math_col}")     
    # compute (spend-pay/balance) ratios
    if CFG.ADD_RATIOS:
      for scol in [f'S_{i}' for i in [16,23]]:
          for pcol in ['P_2','P_3']:
              for bcol in [f'B_{i}' for i in [11,14,17]]:
                  print(f"Addding (spend-pay/balance) ratios {scol}-{pcol}div{bcol}")     
                  df[f'{scol}-{pcol}div{bcol}'] = df[f'{scol}-{pcol}']/df[f'{bcol}']  

      
      for scol in [f'S_{i}' for i in [16,23]]:
          for pcol in ['P_2','P_3']:
              for bcol in [f'B_{i}' for i in [11,14,17]]:
                  print(f"Addding (spend-pay/balance)_sum ratios {scol}-{pcol}div{bcol}")     
                  df[f'{scol}s-{pcol}s_d_{bcol}'] = (df[f'{scol}_sum']-df[f'{pcol}_sum'])/df[f'{bcol}']  

    # Dropping Sum
    drop_col = [col for  col in df.columns if  (("sum" in col))]
    print(f"Dropping {drop_col}")
    df=df.drop(drop_col,axis=1)   

    print(f"Addding col-mean + custom features {len(features_avg)} cols {globals()['g_num_cols']}") 
    main_features = [f'B_{i}' for i in [11,14,17]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]+['P_2','P_3']
    cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'] + ['B_31', 'D_87']
   
    if CFG.SPLIT_CAT:
      df = df.drop(cat_cols,axis=1) 
    # Impute missing values
    df.fillna(value=-1, inplace=True)
    # Replace inf with zeros 
    df.replace([np.inf, -np.inf], -1, inplace=True)
    # Reduce memory
    for c in df.columns:
      if c in get_not_used(): continue
      if str( df[c].dtype )=='int64':
          df[c] = df[c].astype('int32')
      if str(df[c].dtype )=='float64':
          df[c] = df[c].astype('float32')
    return df

def get_segment(test):
    dg = test.groupby('customer_ID').agg({'S_2':'count'})
    dg.columns = ['cus_count']
    dg = dg.reset_index()
    dg['cid'],_ = dg['customer_ID'].factorize()
    dg = dg.sort_values('cid')
    dg['cus_count'] = dg['cus_count'].cumsum()
    
    test = test.merge(dg, on='customer_ID', how='left')
    test = test.sort_values(['cid','S_2'])
    assert test['cus_count'].values[-1] == test.shape[0]
    return test


# ### XGB Params and utility functions
params = {
        #'booster': 'dart',
        'objective': 'binary:logistic', 
        'tree_method': 'gpu_hist', 
        'max_depth': 8,
        'subsample':0.88,
        'colsample_bytree': 0.5,
        'gamma':1.5,
        'min_child_weight':8,
        'lambda':70,
        'eta':0.02, 
}

def xgb_train(x, y, xt, yt,_params= params):
    print("# of features:", x.shape[1])
    assert x.shape[1] == xt.shape[1]
    dtrain = xgb.DMatrix(data=x, label=y)
    dvalid = xgb.DMatrix(data=xt, label=yt)
 
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(_params, dtrain=dtrain,
                num_boost_round=4000,evals=watchlist,
                early_stopping_rounds=600, feval=xgb_amex, maximize=True,
                verbose_eval=100)
    print('best ntree_limit:', bst.best_ntree_limit)
    print('best score:', bst.best_score)
    return bst.predict(dvalid, iteration_range=(0,bst.best_ntree_limit)), bst

# Metrics
def xgb_amex(y_pred, y_true):
    return 'amex', amex_metric_np(y_pred,y_true.get_label())
# Created by https://www.kaggle.com/yunchonggan
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/328020
def amex_metric_np(preds: np.ndarray, target: np.ndarray) -> float:
    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d)

# we still need the official metric since the faster version above is slightly off
import pandas as pd
def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)

def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric(y_true, y_pred), True

# ### Load data and add feature
import torch, gc 
import time
torch.cuda.empty_cache()
gc.collect() 

if CFG.TRAIN or CFG.OPTIMIZE:
  fe = f"{CFG.model_dir}/train_fe.pickle"
  if os.path.exists(fe):
    train = pd.read_pickle(fe)
    print(train.shape)
  else:  
    path = f'{CFG.INPUT}/amex-data-integer-dtypes-parquet-format'
    train = load_train(path)    
    features = [col for col in train.columns if col not in  get_not_used()] 
    print("Saving FE to file")
    print(train.shape)
    train.to_pickle(f"{CFG.model_dir}/train_fe.pickle")

# train.head()
#train[[col for col in train.columns if "_std" in col ]]  
if CFG.DEBUG:
  train = train.sample(n=2000, random_state=42).reset_index(drop=True)
  features = [col for col in train.columns if col not in  get_not_used()]

if CFG.INFER:
  test = process_data(cudf.read_parquet(f'{CFG.INPUT}/amex-data-integer-dtypes-parquet-format/test.parquet' ))
  ds_s = cudf.read_csv(f"{CFG.INPUT}/feature0809/test_flat_1_train.csv")
  print("Merge Sirius data",len(ds_s),len(train))
  test = test.merge(ds_s, on='customer_ID', how='left')

  ds_norm_fe_test = cudf.read_csv(f'{CFG.INPUT}/pca_target_encoding/test_norm_fe.csv')
  print("Merge Sirius norm_fe data",len(ds_norm_fe_test ),len(test))
  test = test.merge(ds_norm_fe_test , on='customer_ID', how='left')
  # ======================= 将测试数据存在本地 ======================
  print("Saving FE to file")    
  test.to_pickle(f"{CFG.model_dir}/test_fe.pickle")

  print("Initial shape ",test.shape) 
  # drop_col = [col for  col in test.columns if  col in zero_imp]
  # print(f"Dropping zero importance {drop_col}")
  # test = test.drop(drop_col,axis=1)    
  print(f"Test size {len(test)}")
  features = [col for col in test.columns if col not in  get_not_used()]  
_ = gc.collect()

# ### Train XGB in K-folds
not_used = get_not_used()
msgs = {}
folds = CFG.n_folds
score = 0

kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed) 

def train_fn(i,x,y,xt,yt,_params= params):   
    yp, bst = xgb_train(x, y, xt, yt,_params)
    bst.save_model(f'{CFG.model_dir}/xgb_{i}.json')
    amex_score =  amex_metric(yt.values,yp) 
    return amex_score, bst,yp

if CFG.TRAIN: 
    oof_predictions = np.zeros(len(train)) 
    kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)
    features = [col for col in train.columns if col not in  get_not_used()]
 
    not_used = get_not_used()
    not_used = [i for i in not_used if i in train.columns]
    msgs = {}
    score = 0 
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[CFG.target])):
        x, y = train[features].iloc[trn_ind], train[CFG.target].iloc[trn_ind]
        xt, yt= train[features].iloc[val_ind], train[CFG.target].iloc[val_ind]
        if os.path.exists(f"{CFG.model_dir}/xgb_{fold}.json"):
          bst = xgb.Booster()          
          dtrain = xgb.DMatrix(data=x, label=y)
          dvalid = xgb.DMatrix(data=xt, label=yt)    
          bst.load_model(f"{CFG.model_dir}/xgb_{fold}.json")  
          print('best ntree_limit:', bst.best_ntree_limit)    
          yp = bst.predict(dvalid, iteration_range=(0,bst.best_ntree_limit))
          amex_score = amex_metric(yt.values,yp)   
          del dtrain,dvalid;gc.collect()
        else: 
          amex_score, model,yp = train_fn(fold,x,y,xt,yt)
          # joblib.dump(model, f'{CFG.model_dir}/xgb_{fold}.json')
        print(f'Our fold {fold} CV score is {amex_score}')
        oof_predictions[val_ind] = yp
        score += amex_score
        del x,y,xt,yt,yp; gc.collect()
        torch.cuda.empty_cache()
    score /= CFG.n_folds
    oof_df = cudf.DataFrame({'customer_ID': train['customer_ID'], 'target': train[CFG.target], 'prediction': oof_predictions})
    #display(oof_df.head())
    oof_df.to_csv(f'{CFG.model_dir}/xg_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)   
    print(f"Average amex score: {score:.4f}")
      
if CFG.INFER:
  test_predictions = np.zeros(len(test))
  yps = []
  not_used = [i for i in not_used if i in test.columns]
  yp=0
  test_predictions = np.zeros(len(test))
  for fold  in range(CFG.n_folds):
    bst = xgb.Booster()           
    bst.load_model(f"{CFG.model_dir}/xgb_{fold}.json") 
    dx = xgb.DMatrix(test.drop(not_used, axis=1))
    print('best ntree_limit:', bst.best_ntree_limit)
    test_predictions += bst.predict(dx, iteration_range=(0,bst.best_ntree_limit))/CFG.n_folds 
  test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
  test_df.to_csv(f'{CFG.model_dir}/test_xg_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False) 

# ### Optuna PO
import optuna 


def optunaOpt(model_name,t_params,n_trials=100, callbacks=(lambda trial: [])):
    """ Best model eval util using Optuna
    """
    def run(trials):
        """ Optima trials lambda"""  
        trial_params = {param:param_fn(trials) for param,param_fn in t_params.items()}  
        if trial_params["bootstrap_type"] == "Bayesian":
          trial_params["bagging_temperature"] =trials.suggest_float("bagging_temperature", 0, 10)
        if trial_params["bootstrap_type"] == "Bernoulli":  
          trial_params["subsample"] =trials.suggest_float("subsample", 0.1, 1)
        not_used = get_not_used()
        not_used = [i for i in not_used if i in train.columns]
        for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[CFG.target])):   
            x, y = train[features].iloc[trn_ind], train[CFG.target].iloc[trn_ind]
            xt, yt= train[features].iloc[val_ind], train[CFG.target].iloc[val_ind] 
            val_pred,model=train_fn(fold,x,y,xt,yt,trial_params) 
            break
        
        amex_score = amex_metric(yt.values,val_pred)
        return amex_score
    
    study = optuna.create_study(direction="maximize",
                                study_name=f"{model_name}-study")
    study.optimize(run, n_trials)
    print('\n Best Trial:')
    print(study.best_trial)
    print('\n Best value')
    print(study.best_value)
    print('\n Best hyperparameters:')
    print(study.best_params)
    return study 
# 0.7932
catb_params = {
    "iterations":lambda trial :trial.suggest_int("iterations", 6000, 11000), 
    #"learning_rate":lambda trial :trial.suggest_loguniform("learning_rate", 0.1,1.0), 
    'l2_leaf_reg' : lambda trial :trial.suggest_categorical('l2_leaf_reg',[0.2,0.5,1,3]),
    "bootstrap_type": lambda trial:trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli" ]
        ),
    #"colsample_bylevel": lambda trial:trial.suggest_float("colsample_bylevel", 0.01, 0.1),
    "depth":lambda trial :trial.suggest_int("max_depth", 7, 12),   
}

if CFG.OPTIMIZE:
  optunaOpt("Catboost",catb_params,n_trials=100, callbacks=(lambda trial: []))
