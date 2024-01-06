from test import detect_outlier
from time import time
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA, TruncatedSVD, KernelPCA
import numpy as np
from os import listdir

report ={}
# outliers = {}
files = ["./data/houses_to_rent_v2","./data/winequalityN","./data/segmented"]
files = ["./data/winequalityN"]

# Kernel PCA: This variation of PCA uses a kernel trick to transform the data into a 
# higher-dimensional space where it is more easily linearly separable. 
# This can be useful for handling non-linearly separable data.

# Incremental PCA: This variation of PCA allows for the analysis of large datasets that 
# cannot be fit into memory all at once. It is useful for handling big data problems.

# Sparse PCA: This variation of PCA adds a sparsity constraint to the PCA problem, which 
# encourages the algorithm to find a lower-dimensional representation of the data with fewer 
# non-zero components.

# KernelPCA : Kernel Principal Component Analysis.
# SparsePCA : Sparse Principal Component Analysis.
# TruncatedSVD : Dimensionality reduction using truncated SVD.
# IncrementalPCA : Incremental Principal Component Analysis.

name = ["PCA", "IncrementalPCA", "SparsePCA", "TruncatedSVD", "KernelPCA"]
# FUNCTION TO DO PCA 
def get_pca_df(filename,name,reduction_factor=.8,method=PCA):
    compressed_filename = filename+'_compressed_%d_%s.csv'%(reduction_factor*100,name)
    compressed_df = None
    if compressed_filename not in listdir():
        df = pd.read_csv(filename+'.csv')
        df = df.select_dtypes([np.number])
        df = df.dropna()

        columns = df.columns
        no_cols = len(df.columns)

        # pca = PCA(n_components=int(reduction_factor*no_cols))
        pca = method(n_components=int(reduction_factor*no_cols))
        pca.fit(df)
        transformed = pca.transform(df)

        compressed_df = pd.DataFrame(transformed)
    else :
        compressed_df = pd.read_csv(compressed_filename)
    # compressed_df.to_csv(compressed_filename)
    return compressed_df,compressed_filename[:-4]

# WITHOUT ANY DIMENTIONALITY REDUCITON OR DATA COMPRESSION
def gen_report_CODWC(files):
    for filename in files:
        start = time()
        df = pd.read_csv(filename+'.csv')
        result = detect_outlier(df,on_cluster=True,plot=False)
        end = time()
        report[filename] = {
            'dataset':filename,
            'time':end-start,
            'outlier(%)':(sum(result[result['outlier']==True]['count'])*100)/sum(result['count']),
            'method':'Without DC/DR',
            # 'no.of rows/objects':result.shape[0],
            '#features':result.shape[1]-3,
        }
    # return list_of_dfs

def gen_report_CODPCA(files,factor=.8):
    for filename in files:
        # name = ["PCA"]
        for i,method in enumerate([PCA, IncrementalPCA, SparsePCA, TruncatedSVD, KernelPCA]):
        # for i,method in enumerate([PCA]):
            compressed_df,compressed_filename = get_pca_df(filename,name=name[i],reduction_factor=factor,method=method)
            features = len(compressed_df.columns)
            # compressed_filename =filename+'_compressed_%d_%s'%(factor*100,name[i]) 
            # compressed_df.to_csv(compressed_filename+'.csv',index=False)
            start = time()
            result = detect_outlier(compressed_df,on_cluster=True,plot=False)
            end = time()
            report[compressed_filename] = {
                'dataset':filename,
                'time':end-start,
                'outlier(%)':(sum(result[result['outlier']==True]['count'])*100)/sum(result['count']),
                'method':(name[i]),
                # 'no.of rows/objects':result.shape[0],
                # '#features':result.shape[1]-3,
                '#features':features,
            }

comparision = None
from pickle import load,dump
if 'comparison.pkl' in listdir():
    with open('comparison.pkl','rb') as f:
        comparision = load(f)
else :
    comparision = {}
    for _ in name:
        comparision[_]=[]
    with open('comparison.pkl','wb') as f:
        dump(comparision,f)

def df_report(report):
    df = pd.DataFrame(report)
    df = df.transpose()

    # ERROR CALCULATION
    squared_error = []
    for row in df.itertuples():
        for filename in files :
            if (row.Index).startswith(filename):
                squared_error.append(df.at[filename,"outlier(%)"])
    df['squared error'] = squared_error
    df['squared error'] = (df['squared error']-df['outlier(%)'])**2
    df = df.sort_values(by=["squared error","time"],ascending=[True,True])
    # df.to_csv('final.csv')
    # df = pd.read_csv('final.csv')

    for row in df.itertuples():
        # print (row)
        if row.method in name:
            comparision[row.method].append(row._6)
    with open('comparison.pkl','wb') as f:
        dump(comparision,f)

    # print (comparision)
    result = {}
    for _ in comparision.keys():
        result[_] = [sum(comparision[_])/len(comparision[_])]
    # print (result)
    comparision_df = pd.DataFrame(result)
    comparision_df = comparision_df.transpose()
    comparision_df = comparision_df.sort_values(by=0,ascending=True)
    print("Following are the best to worst dimentionality reduction techniques based on min mean squared error[Result over %d iterations.]"%(len(comparision['PCA'])))
    print (comparision_df.head())

    grouped = df.groupby('dataset')
    # for dataset_df in grouped:
    #     print (dataset_df)
    #     print ()
    print (grouped.head(30))

for i in range(1):
    gen_report_CODWC(files)
    gen_report_CODPCA(files,.5)
    df_report(report)