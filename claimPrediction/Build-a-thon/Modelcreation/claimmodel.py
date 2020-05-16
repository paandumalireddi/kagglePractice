# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:26:20 2020

@author: mpandavu
"""
import pandas as pd
import os
import seaborn as sns
from sklearn import impute,preprocessing,pipeline,decomposition,compose,feature_selection,ensemble,model_selection,tree
dir="C:/Users/mpandavu/Desktop/Build-a-thon/umr-claim-data"
claim_train=pd.read_csv(os.path.join(dir,"aos0001t-after20200101.csv"))
claim_train1=claim_train.copy();
claim_train1.info()
claim_train2=claim_train1.iloc[:-2,:]

labels_to_drop=['CREATE_DT','CREATE_USER_ID', 'EMPE_FIRST_NM', 'EMPE_LAST_NM','INDIV_FIRST_NM',
       'INDIV_LAST_NM','PRVD_NM                                PRVD_NM2                               PRVD_NM3                               PRVD_NM4',
       'ORGNL_CREATE_DT','DATE','SERV_RCVD_DT','SERV_CMPL_DT','INDIV_BIRTH_DT','SERV_RCVD_DT','SERV_CMPL_DT','PRVD_PTNT_ACCT_CD                      PRVD_PTNT_ACCT_CD2                     PRVD_PTNT_ACCT_CD3                     PRVD_PTNT_ACCT_CD4','CLM_PMT_AMT', 'CLM_PMT_AMT2',
       'OTH_INS_PAID_AMT', 'PTNT_RSPNS_AMT', 'ASSGN_TRANS_NBR', 
       'SERV_RCVD_', 'PAY_RE', 'MIT_NBR', 'PAY_T', 'RACE_NBR','EOB_SITE_XREF_CODE','EOB_CMPL_DT','PRVD_HLTH_ORGN_CD','CLM_NTWRK_STTS']
tmp = claim_train2.isnull().sum()
labels_to_drop.extend(list(tmp[tmp/float(claim_train2.shape[0]) > 0.25].index))

claim_train2.drop(labels_to_drop,axis=1,inplace=True)
claim_train1.count()
claim_train2.columns
claim_train2.drop(['GRP_NM'],axis=1,inplace=True)
claim_train2=claim_train2.iloc[1:,:]
claim_train2[['TOT_BLNG_AMT']]=claim_train2[['TOT_BLNG_AMT']].apply(pd.to_numeric)
claim_train2.info()

sns.countplot(x='MCRFM_ROLL_CD',data=claim_train2)
claim_train2['MCRFM_ROLL_CD'].groupby('MCRFM_ROLL_CD').count()
sns.countplot(x='GRP_NBR',data=claim_train2)
sns.countplot(x='ENRL_CERT_NBR',data=claim_train2)
sns.countplot(x='ENRL_CERT_SUB',data=claim_train2) # categorical
sns.countplot(x='INDIV_SEQ_NBR',data=claim_train2)
#INDIV_BIRTH_DT
sns.countplot(x='PRVD_IRS_NBR',data=claim_train2)
sns.countplot(x='CLM_TYPE_CD',data=claim_train2)
sns.countplot(x='CLM_STTS_CD',data=claim_train2)
sns.countplot(x='CLM_STTS_DESC_CD',data=claim_train2)
#SERV_RCVD_DT,SERV_CMPL_DT
sns.countplot(x='PRVD_PTNT_ACCT_CD',data=claim_train2)
sns.countplot(x='PRVD_PTNT_ACCT_CD2',data=claim_train2)
sns.countplot(x='PRVD_PTNT_ACCT_CD3',data=claim_train2)
sns.countplot(x='PRVD_PTNT_ACCT_CD4',data=claim_train2)
sns.countplot(x='TOT_BLNG_AMT',data=claim_train2)
sns.countplot(x='CLM_PMT_AMT',data=claim_train2)
sns.countplot(x='CLM_PMT_AMT2',data=claim_train2)
sns.countplot(x='OTH_INS_PAID_AMT',data=claim_train2)
sns.countplot(x='PTNT_RSPNS_AMT',data=claim_train2)
sns.countplot(x='EOB_CMPL_DT',data=claim_train2)
sns.countplot(x='ADJ_DESC_CD',data=claim_train2)
sns.countplot(x='TRANS_ID_CD',data=claim_train2)
sns.countplot(x='ASSGN_TRANS_NBR',data=claim_train2)
sns.countplot(x='HISTORY_IND',data=claim_train2)
sns.countplot(x='SERV_RCVD_',data=claim_train2)
#DATE
sns.countplot(x='PAY_RE',data=claim_train2)
sns.countplot(x='RACE_NBR',data=claim_train2)
sns.countplot(x='PRVD_HLTH_ORGN_CD',data=claim_train2)
sns.countplot(x='CLM_NTWRK_STTS',data=claim_train2) # categorical data

sns.factorplot(x='GRP_NBR',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='ENRL_CERT_NBR',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='ENRL_CERT_SUB',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='INDIV_SEQ_NBR',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
#INDIV_BIRTH_DT
sns.factorplot(x='PRVD_IRS_NBR',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='CLM_TYPE_CD',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='CLM_STTS_CD',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='CLM_STTS_DESC_CD',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
#SERV_RCVD_DT,SERV_CMPL_DT
sns.factorplot(x='PRVD_PTNT_ACCT_CD',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='PRVD_PTNT_ACCT_CD2',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='PRVD_PTNT_ACCT_CD3',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='PRVD_PTNT_ACCT_CD4',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='TOT_BLNG_AMT',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='CLM_PMT_AMT',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='CLM_PMT_AMT2',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='OTH_INS_PAID_AMT',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='PTNT_RSPNS_AMT',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='EOB_CMPL_DT',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='ADJ_DESC_CD',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='TRANS_ID_CD',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='ASSGN_TRANS_NBR',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='HISTORY_IND',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='SERV_RCVD_',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
#DATE
sns.factorplot(x='PAY_RE',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='PRVD_HLTH_ORGN_CD',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
sns.factorplot(x='CLM_NTWRK_STTS',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)

#continuous to categorical
sns.FacetGrid(claim_train2, hue="CLM_STTS_CD",size=8).map(sns.kdeplot, "CLM_PMT_AMT").add_legend()
sns.FacetGrid(claim_train2, hue="CLM_STTS_CD",size=8).map(sns.kdeplot, "CLM_PMT_AMT2").add_legend()



claim_train2_independent=claim_train2.loc[:,claim_train2.columns!='CLM_STTS_CD']
claim_train2_dependent=claim_train2['CLM_STTS_CD']

X_train,X_test,y_train,y_test=model_selection.train_test_split(claim_train2_independent, claim_train2_dependent, test_size=0.25, random_state=1)

X_train.columns
num_features_list=X_train.select_dtypes(include=['number']).columns
features_to_cast=X_train.select_dtypes(include=['object']).columns
for feature in features_to_cast:
    X_train[feature] = X_train[feature].astype('category')
cat_features_list=X_train.select_dtypes(include=['category']).columns

claim_train2.info()
cat_feature_pipeline=pipeline.Pipeline([('imputation',impute.SimpleImputer(strategy="most_frequent")),
                                        ('ohe',preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore'))])
transformed_data=cat_feature_pipeline.fit_transform(X_train[['ENRL_CERT_NBR']])
num_feature_pipeline=pipeline.Pipeline([('imputation',impute.SimpleImputer()),
                                        ('standardscalar',preprocessing.StandardScaler())])

transformed_data=num_feature_pipeline.fit_transform(X_train[['TOT_BLNG_AMT']])
feature_preprocessing=compose.ColumnTransformer([('cat_feature_pipeline',cat_feature_pipeline,cat_features_list),
                                                 ('num_feature_pipeline',num_feature_pipeline,num_features_list)],n_jobs=10)


#viz_pipeline = pipeline.Pipeline([
#                     ('preprocess', feature_preprocessing),
#                     ('pca', decomposition.PCA(n_components=0.95))
#                ])
features_pipeline = pipeline.FeatureUnion([
                    ('pca_selector', decomposition.PCA(n_components=0.95) ),
                    ('et_selector', feature_selection.SelectFromModel(ensemble.ExtraTreesClassifier()) )
                ],n_jobs=20)

classifier = tree.DecisionTreeClassifier()
#build complete pipeline with feature selection and ml algorithms
complete_pipeline = pipeline.Pipeline([  
                    ('preprocess', feature_preprocessing),
                    ('zv_filter', feature_selection.VarianceThreshold() ),
                    ('features', features_pipeline ),
                    ('tree', classifier)
                ])
    
pipeline_grid  = {}
grid_estimator = model_selection.GridSearchCV(complete_pipeline, pipeline_grid, scoring="accuracy", cv=5,verbose=3,n_jobs=20)
grid_estimator.fit(X_train, y_train)

