# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:07:14 2020

@author: mpandavu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:23:38 2020

@author: mpandavu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 00:15:30 2020

@author: mpandavu
"""


 

import pandas as pd
from sklearn.externals import joblib
import pyodbc
cnxn = pyodbc.connect(r'Driver={SQL Server};Server=10.106.186.45;Database=AOS001DB;Trusted_Connection=yes;')
cursor = cnxn.cursor()
#result=cursor.execute("select * from AOS.COR0001T where CL_CD='P' and CL_TYPE_CD='CLM_NTWRK_STTS'")

#df=pd.DataFrame(cursor.fetchall())
#df.columns = cursor.keys()
claim_train=pd.read_sql("select aos1t.MCRFM_ROLL_CD,GRP_NBR,ENRL_CERT_NBR,ENRL_CERT_SUB,INDIV_SEQ_NBR,aos1t.PRVD_IRS_NBR,"+
"CLM_TYPE_CD,CLM_STTS_CD,CLM_STTS_DESC_CD,aos2t.PRVD_PTNT_ACCT_CD,TOT_BLNG_AMT,aos2t.ADJ_IND,ADJ_DESC_CD,aos1t.TRANS_ID_CD,aos1t.HISTORY_IND,CLM_SERV_SEQ"+
",SERV_CD,PRCD_OCCUR_CNT,SERV_TYPE_DESC_CD,PRCD_MOD_TXT,SERV_BLNG_AMT,ELIG_AMT,DISC_AMT,DEDUC_AMT,COINS_AMT,aos2t.PTNT_RSPNS_AMT,aos2t.OTH_INS_PAID_AMT,"+
"NON_ELIG_BLNG_AMT,NON_ELIG_BLNG_AMT1,NON_ELIG_BLNG_AMT2,NON_ELIG_BLNG_AMT3,NON_ELIG_DESC_CD,NON_ELIG_DESC_CD1,NON_ELIG_DESC_CD2,NON_ELIG_DESC_CD3,NON_ELIG_ANSI_CD,"+
"NON_PYBL_AMT,SERV_PMT_AMT,SERV_PMT_AMT2,CHK_NBR,PAYEE_PMT_CD,SORT_SEQ,CHK_NBR_INT "+
" from aos.AOS0001T aos1t,aos.AOS0002T aos2t where aos1t.MCRFM_ROLL_CD=aos2t.MCRFM_ROLL_CD and aos1t.CLM_STTS_CD  in ('C','D') and aos1t.CREATE_DT>'20200101'",cnxn)

cnxn.close();

claim_train1=claim_train.copy()
claim_train1.info()
claim_train2=claim_train1.copy()
labels_to_drop=['CLM_STTS_DESC_CD','MCRFM_ROLL_CD','GRP_NBR','PRVD_PTNT_ACCT_CD','ADJ_DESC_CD','ENRL_CERT_NBR']
tmp = claim_train2.isnull().sum()
labels_to_drop.extend(list(tmp[tmp/float(claim_train2.shape[0]) > 0.25].index))
claim_train2.drop(labels_to_drop,axis=1,inplace=True)
claim_train2.columns


claim_train2_independent=claim_train2.loc[:,claim_train2.columns!='CLM_STTS_CD']
claim_train2_dependent=claim_train2['CLM_STTS_CD']

from sklearn import impute,preprocessing,pipeline,decomposition,compose,feature_selection,ensemble,model_selection,naive_bayes
from sklearn2pmml.pipeline import PMMLPipeline
import seaborn as sns

X_train,X_test,y_train,y_test=model_selection.train_test_split(claim_train2_independent, claim_train2_dependent, test_size=0.50, random_state=1)

X_train.columns
#EDA work:
sns.countplot(x='TRANS_ID_CD', data=X_train)
sns.factorplot(x='TRANS_ID_CD',hue='CLM_STTS_CD',data=claim_train2,kind="count", size=6)
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
num_features_list=X_train.select_dtypes(include=['number']).columns
print("numeric features list",num_features_list)
features_to_cast=X_train.select_dtypes(include=['object']).columns
for feature in features_to_cast:
    X_train[feature] = X_train[feature].astype('category')
cat_features_list=X_train.select_dtypes(include=['category']).columns
print("Category features list is",cat_features_list)
claim_train2.info()

lencoder = preprocessing.LabelEncoder()
for i in cat_features_list:
    lencoder.fit(X_train[i])
    X_train[i] = lencoder.transform(X_train[i])


cat_feature_pipeline=pipeline.Pipeline([('imputation',impute.SimpleImputer(strategy="most_frequent")),
                                        #('label',preprocessing.LabelEncoder())
                                        ])
#transformed_data=cat_feature_pipeline.fit_transform(X_train[['ENRL_CERT_NBR']])
num_feature_pipeline=pipeline.Pipeline([('imputation',impute.SimpleImputer()),
                                        ('standardscalar',preprocessing.StandardScaler())])

#transformed_data=num_feature_pipeline.fit_transform(X_train[['TOT_BLNG_AMT']])
feature_preprocessing=compose.ColumnTransformer([('cat_feature_pipeline',cat_feature_pipeline,cat_features_list),
                                                 ('num_feature_pipeline',num_feature_pipeline,num_features_list)],n_jobs=10)


features_pipeline = pipeline.FeatureUnion([
                    ('pca_selector', decomposition.PCA(n_components=0.90) ),
                    ('et_selector', feature_selection.SelectFromModel(ensemble.ExtraTreesClassifier()) )
                ],n_jobs=20)

classifier = naive_bayes.GaussianNB()
#build complete pipeline with feature selection and ml algorithms
complete_pipeline =  PMMLPipeline([  
                    ('preprocess', feature_preprocessing),
                    ('zv_filter', feature_selection.VarianceThreshold() ),
                    ('features', features_pipeline ),
                    ('tree', classifier)
                ])
    
pipeline_grid  = {}
grid_estimator = model_selection.GridSearchCV(complete_pipeline, pipeline_grid, scoring="accuracy", cv=5,verbose=10,n_jobs=20)
grid_estimator.fit(X_train, y_train)
print(grid_estimator.best_estimator_)
print(grid_estimator.best_params_)
print(grid_estimator.best_score_)


num_features_list_test=X_test.select_dtypes(include=['number']).columns
print("numeric features list",num_features_list_test)
features_to_cast_test=X_test.select_dtypes(include=['object']).columns
for feature in features_to_cast_test:
    X_test[feature] = X_test[feature].astype('category')
cat_features_list_test=X_test.select_dtypes(include=['category']).columns
print("Category features list is",cat_features_list_test)
claim_train2.info()

lencoder = preprocessing.LabelEncoder()
for i in cat_features_list:
    lencoder.fit(X_test[i])
    X_test[i] = lencoder.transform(X_test[i])
    
    
    
    
    
print(grid_estimator.score(X_test,y_test))
predict_proba=grid_estimator.predict_proba(X_test)
y_test=y_test.to_frame()
y_test['predicted']=grid_estimator.predict(X_test)


import pickle
filename = 'finalized_model_RF.sav'
pickle.dump(grid_estimator.best_estimator_, open(filename, 'wb'))

#checking for real time data
import pandas as pd
import pyodbc
cnxn = pyodbc.connect(r'Driver={SQL Server};Server=10.106.186.45;Database=AOS001DB;Trusted_Connection=yes;')
cursor = cnxn.cursor()
#result=cursor.execute("select * from AOS.COR0001T where CL_CD='P' and CL_TYPE_CD='CLM_NTWRK_STTS'")

#df=pd.DataFrame(cursor.fetchall())
#df.columns = cursor.keys()
claim_inprogress_data=pd.read_sql("select top 1000 aos1t.MCRFM_ROLL_CD,GRP_NBR,ENRL_CERT_NBR,ENRL_CERT_SUB,INDIV_SEQ_NBR,aos1t.PRVD_IRS_NBR,"+
"CLM_TYPE_CD,CLM_STTS_CD,CLM_STTS_DESC_CD,aos2t.PRVD_PTNT_ACCT_CD,TOT_BLNG_AMT,aos2t.ADJ_IND,ADJ_DESC_CD,aos1t.TRANS_ID_CD,aos1t.HISTORY_IND,CLM_SERV_SEQ"+
",SERV_CD,PRCD_OCCUR_CNT,SERV_TYPE_DESC_CD,PRCD_MOD_TXT,SERV_BLNG_AMT,ELIG_AMT,DISC_AMT,DEDUC_AMT,COINS_AMT,aos2t.PTNT_RSPNS_AMT,aos2t.OTH_INS_PAID_AMT,"+
"NON_ELIG_BLNG_AMT,NON_ELIG_BLNG_AMT1,NON_ELIG_BLNG_AMT2,NON_ELIG_BLNG_AMT3,NON_ELIG_DESC_CD,NON_ELIG_DESC_CD1,NON_ELIG_DESC_CD2,NON_ELIG_DESC_CD3,NON_ELIG_ANSI_CD,"+
"NON_PYBL_AMT,SERV_PMT_AMT,SERV_PMT_AMT2,CHK_NBR,PAYEE_PMT_CD,SORT_SEQ,CHK_NBR_INT "+
" from aos.AOS0001T aos1t,aos.AOS0002T aos2t where aos1t.MCRFM_ROLL_CD=aos2t.MCRFM_ROLL_CD and aos1t.CLM_STTS_CD  in ('D') and aos1t.CREATE_DT>'20200101'",cnxn)

cnxn.close();
claim_inprogress1=claim_inprogress_data
sns.factorplot(x='TRANS_ID_CD',hue='CLM_STTS_CD',data=claim_inprogress1,kind="count", size=6)
labels_to_drop_inprogress=['CLM_STTS_DESC_CD','MCRFM_ROLL_CD','GRP_NBR','PRVD_PTNT_ACCT_CD','ADJ_DESC_CD','CLM_STTS_CD','ENRL_CERT_NBR']
tmp = claim_inprogress_data.isnull().sum()
labels_to_drop_inprogress.extend(list(tmp[tmp/float(claim_inprogress_data.shape[0]) > 0.25].index))
claim_inprogress_data.drop(labels_to_drop_inprogress,axis=1,inplace=True)

num_features_list_test_inprogress=claim_inprogress_data.select_dtypes(include=['number']).columns
print("numeric features list",num_features_list_test_inprogress)
features_to_cast_test_inprogress=claim_inprogress_data.select_dtypes(include=['object']).columns
for feature in features_to_cast_test_inprogress:
    claim_inprogress_data[feature] = claim_inprogress_data[feature].astype('category')
cat_features_list_test_inprogress=claim_inprogress_data.select_dtypes(include=['category']).columns
print("Category features list is",cat_features_list_test_inprogress)
claim_inprogress_data.info()


for i in cat_features_list_test_inprogress:
    lencoder.fit(claim_inprogress_data[i])
    claim_inprogress_data[i] = lencoder.transform(claim_inprogress_data[i])
    
    
    
    


#y_test=y_test.to_frame()
y_final_test=grid_estimator.predict(claim_inprogress_data)



loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)