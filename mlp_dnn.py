#!/usr/bin/env python
# coding: utf-8

import os,sys,random
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import tensorflow as tf 
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    BATCH_SIZE = tpu_strategy.num_replicas_in_sync * 64
    print("Running on TPU:", tpu.master())
    print(f"Batch Size: {BATCH_SIZE}")
    
except ValueError:
    strategy = tf.distribute.get_strategy()
    BATCH_SIZE = 512
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    print(f"Batch Size: {BATCH_SIZE}")


# Utils
# Function to get hardware strategy
def get_hardware_strategy():
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        tf.config.optimizer.set_jit(True)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    return tpu, strategy
tpu, strategy = get_hardware_strategy()    

import os,random,time,datetime,math
import pandas as cudf
import pandas as pd
import numpy as cupy  
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder, PowerTransformer
import joblib
import pathlib
import tqdm
import pickle
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from sklearn.model_selection import StratifiedKFold
from adabelief_tf import AdaBeliefOptimizer
from colorama import Fore, Back, Style
# tf.config.threading.set_inter_op_parallelism_threads(4)
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, load_model,model_from_json
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input, InputLayer, Add, Concatenate, Dropout, BatchNormalization, Conv1D, Reshape, Flatten, AveragePooling1D, MaxPool1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, LSTM,GRU,  Conv1D,Dropout,Bidirectional,Multiply

import seaborn as sns 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16,9)
plt.rcParams["figure.facecolor"] = '#FFFACD'
plt.rcParams["axes.facecolor"] = '#FFFFE0'
plt.rcParams["axes.grid"] = True 
plt.rcParams["grid.alpha"] = 0.5
plt.rcParams["grid.linestyle"] = '--'


class CFG:
  seed = 42
  INPUT = "./input"
  TRAIN = True 
  INFER = True
  n_folds = 5
  target ='target'
  DEBUG= False 
  ADD_CAT = True
  ADD_LAG = True 
  COMPUTE_Z = False
  ADD_DIFF_1 = True
  ADD_DIFF =  [] 
  ADD_PCTDIFF = []
  ADD_RATIOS = False
  KURT = False
  TRIM = True 
  APPLY_CAT_EMB = False
  SPLIT_CAT = False
  max_epochs = 500
  batch_size = 2*1024  
  model_dir = "./output/models/dnn"

path = f'{CFG.INPUT}/amex-data-integer-dtypes-parquet-format'   
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(CFG.seed)    


# ====================================================
# Seed everything
# ====================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(CFG.seed)

# Feature Engineering
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
 
from numpy import mean, absolute

def mad(data, axis=None):
    return mean(absolute(data - mean(data, axis)), axis)

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
  return ['row_id', 'customer_ID', 'target', 'cid', 'S_2',"D_103","D_139"]

stats = ['mean', 'min', 'max','std']
features_avg = ['S_2_wk','B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18',
                'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_28', 'B_29', 'B_30', 'B_32', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42',
                'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_50', 'D_51', 'D_53', 'D_54', 'D_55', 'D_58', 'D_59', 'D_60', 'D_61', 
                'D_62', 'D_65', 'D_66', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_75', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_86', 'D_91', 
                'D_92', 'D_94', 'D_96', 'D_103', 'D_104', 'D_108', 'D_112', 'D_113', 'D_114', 'D_115', 'D_117', 'D_118', 'D_119', 'D_120', 'D_121', 'D_122', 'D_123',
                'D_124', 'D_125', 'D_126', 'D_128', 'D_129', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135', 'D_136', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145',
                'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_14', 'R_15', 'R_16', 'R_17', 'R_20', 'R_21', 'R_22', 'R_24', 
                'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_9', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_18', 'S_22', 'S_23', 'S_25', 'S_26']
#features_std = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_28', 'B_29', 'B_30', 'B_32', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_50', 'D_51', 'D_53', 'D_54', 'D_55', 'D_58', 'D_59', 'D_60', 'D_61', 'D_62', 'D_65', 'D_66', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_75', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_86', 'D_91', 'D_92', 'D_94', 'D_96', 'D_103', 'D_104', 'D_108', 'D_112', 'D_113', 'D_114', 'D_115', 'D_117', 'D_118', 'D_119', 'D_120', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_128', 'D_129', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135', 'D_136', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_14', 'R_15', 'R_16', 'R_17', 'R_20', 'R_21', 'R_22', 'R_24', 'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_9', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_18', 'S_22', 'S_23', 'S_25', 'S_26']
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
features_mad = ['D_43','B_4','D_39','P_3','B_3','S_3','B_40','D_61','D_48','R_1','D_49','S_7','B_17','D_77','P_2','S_27','B_7','D_62','D_44','S_12','D_105','D_52','S_26','D_46',
                'S_8','S_9','B_23','D_65','D_47','R_3','D_55','S_15','S_11','D_75','D_53','B_13','D_121','S_24','D_115','D_50','B_28','S_22','D_58',
                'S_5','D_59','B_5','S_23','D_45','D_70','S_13','D_74','D_60','R_11','D_119']
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
        train_diff = df.loc[:,dff_cols+['customer_ID']].groupby(['customer_ID']).apply(lambda x: cupy.diff(x.values[-2:,:], axis = 0).squeeze().astype(cupy.float32))
        index = train_diff.index
        cols = [col + '_diff1' for col in df[dff_cols].columns]
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
    del s2_count; gc.collect() 

    if CFG.ADD_MIDDLE:
      df_middle = df[df.S_2_Count > 2].groupby(['customer_ID'])[balance_p+payment_p+delq+spend_p].apply(lambda x: x.iloc[(len(x)+1)//2])   
      df_middle.columns = [x+'_mid' for x in df_middle.columns]  
      dgs.append(df_middle) 
      print(f"Mid Cols added [{len(df_middle.columns)}]")    
      del df_middle; gc.collect() 
      
    # cudf merge changes row orders
    # restore the original row order by sorting row_id
    df = df.sort_values('row_id')
    df = df.drop(['row_id'],axis=1)
    # Impute missing values
    df.fillna(value=-1, inplace=True)
    ## Replace inf with zeros 
    df.replace([np.inf, -np.inf], -1, inplace=True)
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

      ds_norm_fe = cudf.read_csv(f'{CFG.INPUT}/pca_target_encoding/train_norm_fe.csv')
      ds_norm_fe = ds_norm_fe.drop(['target'], axis=1)
      print("Merge Sirius norm_fe data",len(ds_norm_fe),len(train))
      train = train.merge(ds_norm_fe, on='customer_ID', how='left')


    trainl = cudf.read_csv(f'{CFG.INPUT}/train_labels.csv')
    trainl.target = trainl.target.astype('int8')
    train = train.merge(trainl, on='customer_ID', how='left')
    return train

def process_data(df, infer = False):
    df,dgs = preprocess(df) 
    df = df.drop_duplicates('customer_ID',keep='last')
    for dg in dgs:
        df = df.merge(dg, on='customer_ID', how='left')
        # drop specific non impactful cols 
    del dgs; gc.collect()    
    if CFG.TRIM:
      drop_col = [col for  col in df.columns if (("std" in col) and (col.replace("_std","") not in features_std))]
      print(f"Dropping {drop_col}")
      df=df.drop(drop_col,axis=1)      
      drop_col = [col for  col in df.columns if (("min" in col) and (col.replace("_min","") not in features_min))]
      print(f"Dropping {drop_col}")
      df=df.drop(drop_col,axis=1)
      drop_col = [col for  col in df.columns if  (("max" in col) and (col.replace("_max","") not in features_max))]
      print(f"Dropping {drop_col}")
      df=df.drop(drop_col,axis=1)
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
    # More Lag Features
    for col in spend_p+payment_p+balance_p:
        for col_2 in ['min','max']: 
          if f"{col}_{col_2}" in df.columns:
              df[f'{col}_{col_2}_lag_sub'] = df[f"{col}_{col_2}"] - df[col]
              df[f'{col}_{col_2}_lag_div'] = df[f"{col}_{col_2}"] / df[col] 

    print("Added more lags")
    df["P2B9"] = df["P_2"] / df["B_9"] 
    
    for pcol in math_col:
      if pcol+"_mean" in df.columns: 
        df[f"{pcol}_lag_mean"] = df[pcol] - df[pcol+"_mean"]
        df[f"{pcol}_div_mean"] = df[pcol] /df[pcol+"_mean"]
      if (pcol+"_min" in df.columns) and (pcol+"_max" in df.columns):  
        df[f'{pcol}_min_div_max'] = df[pcol+"_min"] / df[pcol+"_max"]
        df[f'{pcol}_min-max'] = df[pcol+"_min"] - df[pcol+"_max"] 

     
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
    _ = gc.collect()
    if not CFG.APPLY_CAT_EMB:
      cat_oh= cat_cols
      df_categorical = df[cat_oh] 
      if not infer:
        ohe = OneHotEncoder(drop='first', sparse=False, dtype=np.float32, handle_unknown='ignore')
        ohe.fit(df_categorical)
        with open(f"{CFG.model_dir}/ohe.pickle", 'wb') as f: pickle.dump(ohe, f)          
      else:
        ohe = pickle.load(open(f"{CFG.model_dir}/ohe.pickle","rb"))
      df_categorical = cudf.DataFrame(ohe.transform(df_categorical).astype(np.float16),
                                    index=df_categorical.index).rename(columns=str)
      df.drop(cat_cols,axis=1,inplace=True)                                    
      df = cudf.concat([df, df_categorical ], axis=1)
      del df_categorical,ohe
    _ = gc.collect()
    # Reduce memory
    for c in df.columns:
      if c in get_not_used(): continue
      if str( df[c].dtype )=='int64':
          df[c] = df[c].astype('int32')
      if str(df[c].dtype )=='float64':
          df[c] = df[c].astype('float32')   
    # Impute missing values
    df.fillna(value=-1, inplace=True)
    ## Replace inf with zeros 
    df.replace([np.inf, -np.inf], -1, inplace=True)
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


# ### Model
def my_model(n_inputs):
    """Sequential neural network with a skip connection.
    
    Returns a compiled instance of tensorflow.keras.models.Model.
    """
    activation = 'swish'
    inputs = Input(shape=n_inputs)
    x = Reshape((n_inputs, 1))(inputs)
    # 15 agg features per main feature, size = 15, step = 15.
    x = Conv1D(24,15,strides=15, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Conv1D(12,1, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Conv1D(4,1, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.33)(x)
    x = Dense(32, activation = activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation = activation)(x)
    x = Dropout(0.1)(x)
    x = Dense(16, activation = activation)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    gc.collect()
    return Model(inputs, outputs)

def my_model(n_inputs):
    """Sequential neural network with a skip connection.
    
    Returns a compiled instance of tensorflow.keras.models.Model.
    """
    activation = 'swish'
    l1 = 1e-7
    l2 = 4e-4
    inputs = Input(shape=(n_inputs, ))
    x0 = BatchNormalization()(inputs)
    x0 = Dense(256, 
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1,l2=l2),
#                activity_regularizer=tf.keras.regularizers.L1L2(l1=l1,l2=l2),
              activation=activation,
             )(x0)
    x0 = Dropout(0.1)(x0)
    x = Dense(64, 
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1,l2=l2),
#               activity_regularizer=tf.keras.regularizers.L1L2(l1=l1,l2=l2),
              activation=activation,
             )(x0)
    x = Dense(64, 
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1,l2=l2),
#               activity_regularizer=tf.keras.regularizers.L1L2(l1=l1,l2=l2),
              activation=activation,
             )(x)
    x = Concatenate()([x, x0])
    x = Dropout(0.1)(x)
    x = Dense(16, 
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1,l2=l2),
#               activity_regularizer=tf.keras.regularizers.L1L2(l1=l1,l2=l2),
              activation=activation,
             )(x)
    x = Dense(1,
              activation='sigmoid',
             )(x)
    return Model(inputs, x)    


def my_model1(n_inputs):
    
    x_input = Input(shape=(n_inputs,))
    
    x1 = Bidirectional(LSTM(units=768, return_sequences=True))(x_input)
    x2 = Bidirectional(LSTM(units=512, return_sequences=True))(x1)
    x3 = Bidirectional(LSTM(units=384, return_sequences=True))(x2)
    x4 = Bidirectional(LSTM(units=256, return_sequences=True))(x3)
    x5 = Bidirectional(LSTM(units=128, return_sequences=True))(x4)
    
    z2 = Bidirectional(GRU(units=384, return_sequences=True))(x2)
    
    z31 = Multiply()([x3, z2])
    z31 = BatchNormalization()(z31)
    z3 = Bidirectional(GRU(units=256, return_sequences=True))(z31)
    
    z41 = Multiply()([x4, z3])
    z41 = BatchNormalization()(z41)
    z4 = Bidirectional(GRU(units=128, return_sequences=True))(z41)
    
    z51 = Multiply()([x5, z4])
    z51 = BatchNormalization()(z51)
    z5 = Bidirectional(GRU(units=64, return_sequences=True))(z51)
    
    x = Concatenate(axis=2)([x5, z2, z3, z4, z5])
    
    x = Dense(units=128, activation='selu')(x)
    
    x_output = Dense(units=1)(x)

    model = Model(inputs=x_input, outputs=x_output, 
                  name='DNN_Model')
    return model            


# #### Metrics
def amex_metric(y_true, y_pred, return_components=False) -> float:
    """Amex metric for ndarrays"""
    def top_four_percent_captured(df) -> float:
        """Corresponds to the recall for a threshold of 4 %"""
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(df) -> float:
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(df) -> float:
        """Corresponds to 2 * AUC - 1"""
        df2 = pd.DataFrame({'target': df.target, 'prediction': df.target})
        df2.sort_values('prediction', ascending=False, inplace=True)
        return weighted_gini(df) / weighted_gini(df2)

    df = pd.DataFrame({'target': y_true.ravel(), 'prediction': y_pred.ravel()})
    df.sort_values('prediction', ascending=False, inplace=True)
    g = normalized_weighted_gini(df)
    d = top_four_percent_captured(df)

    if return_components: return g, d, 0.5 * (g + d)
    del df
    gc.collect()
    return 0.5 * (g + d)
#Keras
def DiceBCELoss(targets, inputs, smooth=1e-6):    
    BCE =  binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))    
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE
ALPHA = 0.8
GAMMA = 2
def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss
#Keras
def DiceBCELoss(targets, inputs, smooth=1e-6):    
    BCE =  binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))    
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE
ALPHA = 0.8
GAMMA = 2
def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss

## Scalars
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted


class GaussRankScaler(BaseEstimator, TransformerMixin):
    """Transform features by scaling each feature to a normal distribution.
    Parameters
        ----------
        epsilon : float, optional, default 1e-4
            A small amount added to the lower bound or subtracted
            from the upper bound. This value prevents infinite number
            from occurring when applying the inverse error function.
        copy : boolean, optional, default True
            If False, try to avoid a copy and do inplace scaling instead.
            This is not guaranteed to always work inplace; e.g. if the data is
            not a NumPy array, a copy may still be returned.
        n_jobs : int or None, optional, default None
            Number of jobs to run in parallel.
            ``None`` means 1 and ``-1`` means using all processors.
        interp_kind : str or int, optional, default 'linear'
           Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
            refer to a spline interpolation of zeroth, first, second or third
            order; 'previous' and 'next' simply return the previous or next value
            of the point) or as an integer specifying the order of the spline
            interpolator to use.
        interp_copy : bool, optional, default False
            If True, the interpolation function makes internal copies of x and y.
            If False, references to `x` and `y` are used.
        Attributes
        ----------
        interp_func_ : list
            The interpolation function for each feature in the training set.
    """

    def __init__(
        self,
        epsilon=1e-4,
        copy=True,
        n_jobs=None,
        interp_kind="linear",
        interp_copy=False,
    ):
        self.epsilon = epsilon
        self.copy = copy
        self.interp_kind = interp_kind
        self.interp_copy = interp_copy
        self.fill_value = "extrapolate"
        self.n_jobs = n_jobs
        self.bound = 1.0 - self.epsilon

    def fit(self, X, y=None):
        """Fit interpolation function to link rank with original data for future scaling
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to fit interpolation function for later scaling along the features axis.
        y
            Ignored
        """
        X = check_array(
            X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True
        )

        self.interp_func_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit)(x) for x in X.T
        )
        return self

    def _fit(self, x):
        x = self.drop_duplicates(x)
        rank = np.argsort(np.argsort(x))
        factor = np.max(rank) / 2.0 * self.bound
        scaled_rank = np.clip(rank / factor - self.bound, -self.bound, self.bound)
        return interp1d(
            x,
            scaled_rank,
            kind=self.interp_kind,
            copy=self.interp_copy,
            fill_value=self.fill_value,
        )

    def transform(self, X, copy=None):
        """Scale the data with the Gauss Rank algorithm
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        check_is_fitted(self, "interp_func_")

        copy = copy if copy is not None else self.copy
        X = check_array(
            X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True
        )

        X = np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._transform)(i, x) for i, x in enumerate(X.T)
            )
        ).T
        return X

    def _transform(self, i, x):
        clipped = np.clip(self.interp_func_[i](x), -self.bound, self.bound)
        return erfinv(clipped)

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        check_is_fitted(self, "interp_func_")

        copy = copy if copy is not None else self.copy
        X = check_array(
            X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True
        )

        X = np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._inverse_transform)(i, x) for i, x in enumerate(X.T)
            )
        ).T
        return X

    def _inverse_transform(self, i, x):
        inv_interp_func = interp1d(
            self.interp_func_[i].y,
            self.interp_func_[i].x,
            kind=self.interp_kind,
            copy=self.interp_copy,
            fill_value=self.fill_value,
        )
        return inv_interp_func(erf(x))

    @staticmethod
    def drop_duplicates(x):
        is_unique = np.zeros_like(x, dtype=bool)
        is_unique[np.unique(x, return_index=True)[1]] = True
        return x[is_unique]

# ### Load data and add feature
import torch, gc 
torch.cuda.empty_cache()
_ = gc.collect()  

if CFG.TRAIN :
  fe = f"{CFG.model_dir}/train_fe_v1659103981.4445105.pickle"
  if os.path.exists(fe):
    train = cudf.read_pickle(fe)
  else:  
    path = f'{CFG.INPUT}/amex-data-integer-dtypes-parquet-format'
    train = load_train(path)       
    print("Saving FE to file")
    train.to_pickle(f"{CFG.model_dir}/train_fe_v{time.time()}.pickle")
  gc.collect()
  drop_col = [col for col in train.columns if (("_diff" in col) or ("_TE11" in col) or ("_TE10" in col) or ("_TE9" in col) or ("_TE8" in col) or ("_TE7" in col)) ]
  print(f"Dropping zero importance {drop_col}")
  train = train.drop(drop_col,axis=1)       
  print(train.shape)  
  #display(train[cat_cols].head(4))  
  features = [col for col in train.columns if col not in  get_not_used()]
  plot_model(my_model(len(features)), show_layer_names=False, show_shapes=True)
if CFG.DEBUG:
  train = train.sample(n=2000, random_state=42).reset_index(drop=True) 
  features = [col for col in train.columns if col not in  get_not_used()]

if CFG.INFER:
  test = process_data(cudf.read_parquet(f'{CFG.INPUT}/amex-data-integer-dtypes-parquet-format/test.parquet'),True)
  if CFG.ADD_SIRU:
    ds_s = cudf.read_csv(f"{CFG.INPUT}/feature0809/test_flat_1_train.csv", index_col = False)
    print("Merge Sirius data",len(ds_s),len(test))
    test = test.merge(ds_s, on='customer_ID', how='left')

    ds_norm_fe_test = cudf.read_csv(f'{CFG.INPUT}/pca_target_encoding/test_norm_fe.csv')
    print("Merge Sirius norm_fe data",len(ds_norm_fe_test ),len(test))
    test = test.merge(ds_norm_fe_test , on='customer_ID', how='left')
  

  drop_col = [col for col in test.columns if (("_diff" in col) or ("_TE11" in col) or ("_TE10" in col) or ("_TE9" in col) or ("_TE8" in col) or ("_TE7" in col)) ]
  print(f"Dropping zero importance {drop_col}")
  test = test.drop(drop_col,axis=1)
  features = [col for col in test.columns if col not in  get_not_used()]


# ### Train
VERBOSE = 0
CYCLES = 1
EPOCHS = 400
USE_PLATEAU = True
BATCH_SIZE = 2*1024 
LR_START = 0.01
EPOCHS_EXPONENTIALDECAY = 100
LR_END = 1e-5 # learning rate at the end of training
oof_predictions = np.zeros(len(train))
score_list = []
history_list = []

def fit_model(X_tr, X_va, y_tr, y_va=None, fold=0):
    start_time = datetime.datetime.now() 
    scaler = StandardScaler()
    scaler = GaussRankScaler() 
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    with open(f"{CFG.model_dir}/scaler_{fold}.pickle", 'wb') as f: pickle.dump(scaler, f)

    es = EarlyStopping(monitor="val_loss",
                       patience=24, 
                       verbose=VERBOSE,
                       mode="min", 
                       restore_best_weights=True) 
    if USE_PLATEAU and X_va is not None: # use early stopping
        epochs = EPOCHS
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, 
                               patience=4, verbose=VERBOSE)
        es = EarlyStopping(monitor="val_loss",
                           patience=12, 
                           verbose=VERBOSE,
                           mode="min", 
                           restore_best_weights=True)
        callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN(),TqdmCallback(verbose=1)]

    else: # use exponential learning rate decay rather than early stopping
        print("using Exp decay")
        epochs = EPOCHS_EXPONENTIALDECAY

        def exponential_decay(epoch):
            # v decays from e^a to 1 in every cycle
            # w decays from 1 to 0 in every cycle
            # epoch == 0                  -> w = 1 (first epoch of cycle)
            # epoch == epochs_per_cycle-1 -> w = 0 (last epoch of cycle)
            # higher a -> decay starts with a steeper decline
            a = 3
            epochs_per_cycle = epochs // CYCLES
            epoch_in_cycle = epoch % epochs_per_cycle
            if epochs_per_cycle > 1:
                v = math.exp(a * (1 - epoch_in_cycle / (epochs_per_cycle-1)))
                w = (v - 1) / (math.exp(a) - 1)
            else:
                w = 1
            return w * LR_START + (1 - w) * LR_END

        lr = LearningRateScheduler(exponential_decay, verbose=0)
        callbacks = [lr, tf.keras.callbacks.TerminateOnNaN(),TqdmCallback(verbose=1)]
    checkpoint_filepath = f"{CFG.model_dir}/model_fold{fold}_seed{CFG.seed}.h5"   
    if os.path.exists(checkpoint_filepath):
      model = tf.keras.models.load_model(checkpoint_filepath) 
    else:
      model = my_model(X_tr.shape[1])
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_START),
                loss=tf.keras.losses.BinaryCrossentropy())
      #model.compile(optimizer=AdaBeliefOptimizer(learning_rate=0.02,
      #                                           weight_decay = 1e-5,
      #                                           epsilon = 1e-7,print_change_log = False
      #    ), loss='binary_crossentropy', )
      gc.collect() 
      history = model.fit(X_tr, y_tr, 
              validation_data=(X_va, y_va),
              epochs=EPOCHS,
              verbose=VERBOSE,
              batch_size=BATCH_SIZE,
              shuffle=True,
              callbacks=callbacks)
      history_list.append(history.history)
      del X_tr, y_tr, callbacks,history, es, lr; gc.collect()
    oof_pred = model.predict(X_va).reshape( (len(X_va), )) 
    score = amex_metric(y_va, oof_pred)
    if os.path.exists(checkpoint_filepath):
      print(f"{Fore.GREEN}{Style.BRIGHT}Fold {fold} | {str(datetime.datetime.now() - start_time)[-12:-7]}" 
          f" |  Score: {score:.5f}{Style.RESET_ALL}")
    else:  
      lastloss = f"Training loss: {history_list[-1]['loss'][-1]:.4f} | Val loss: {history_list[-1]['val_loss'][-1]:.4f}"
      print(f"{Fore.GREEN}{Style.BRIGHT}Fold {fold} | {str(datetime.datetime.now() - start_time)[-12:-7]}"
        f" | {len(history_list[-1]['loss']):3} ep"
        f" | {lastloss} | Score: {score:.5f}{Style.RESET_ALL}")
      
    score_list.append(score)
    oof_predictions[idx_va]+=oof_pred
    
    del scaler;gc.collect()
    model.save(f"{CFG.model_dir}/model_fold{fold}_seed{CFG.seed}.h5") 

def fit_train_models(X_tr, X_va, y_tr, y_va=None, fold=0):
    print(f'{Fore.GREEN}{Style.BRIGHT}Training KFOLD {fold} WITH SEED {CFG.seed} {Style.RESET_ALL}')
    oof_pred = fit_model(X_tr, X_va, y_tr, y_va, fold) 
    gc.collect()
#with strategy.scope():

# TRAIN
tf.keras.backend.clear_session()
if CFG.TRAIN:
  num_cols = [col for col in train.columns if col not in cat_cols+not_used]
  kf = StratifiedKFold(n_splits=CFG.n_folds, shuffle= True, random_state= CFG.seed)
  print(train.shape)
  for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train.target)):
    fit_train_models(train.iloc[idx_tr][features],train.iloc[idx_va][features],train.target.iloc[idx_tr], train.target.iloc[idx_va], fold)
  print(f"{Fore.GREEN}{Style.BRIGHT}OOF Score: {np.mean(score_list):.5f}{Style.RESET_ALL}")
  # SAVE OOF
  oof_df = pd.DataFrame({'customer_ID': train['customer_ID'], 'target': train[CFG.target], 'prediction': oof_predictions})
  oof_df.to_csv(f'{CFG.model_dir}/mlp_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)    


# ### INFER
if CFG.INFER:
  test_predictions = np.zeros(len(test))
  not_used = [i for i in not_used if i in test.columns]  
  for fold in range(CFG.n_folds):
    scalar_path = f"{CFG.model_dir}/scaler_{fold}.pickle" 
    scaler = pickle.load(open(scalar_path,"rb"))
    _test = scaler.transform(test[features])
    checkpoint_filepath = f"{CFG.model_dir}/model_fold{fold}_seed{CFG.seed}.h5" 
    model = tf.keras.models.load_model(checkpoint_filepath) 
    preds = model.predict(_test).reshape( (len(_test), )) 
    test_predictions += preds / CFG.n_folds 
test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
test_df.to_csv(f'{CFG.model_dir}/test_lgbm_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False) 

