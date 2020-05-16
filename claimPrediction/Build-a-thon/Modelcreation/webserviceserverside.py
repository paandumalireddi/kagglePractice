# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:02:13 2020

@author: mpandavu
"""

from flask import Flask,request,jsonify
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import os
import json

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/checkClaimStatus', methods=['POST'])
def getClaimStatus():
    data=request.get_json()
    #claim_inprogress_data_client=pd.read_json(os.path.join("C:/Users/mpandavu/Desktop/Build-a-thon/Modelcreation","claimmodeljson.txt"))
    print("data is ",data)
    claim_inprogress_data_client=pd.DataFrame(data)
    print("input df is ",claim_inprogress_data_client)
    claim_inprogress_data_client.info
   # filename = 'C:/Users/mpandavu/finalized_model.sav'
   # loaded_model = pickle.load(open(filename, 'rb'))
    X_inprogress_data=claim_inprogress_data_client
    path="C:/Users/mpandavu/Desktop/Build-a-thon/Modelcreation"
    objects_map = joblib.load(os.path.join(path, 'deployment.pkl') )
    cat_features_list_test_inprogress_client=objects_map.get('cat-features')
    for i in cat_features_list_test_inprogress_client:
        print("feature is"+i)
        X_inprogress_data[i] = objects_map.get('lencoder').fit_transform(claim_inprogress_data_client['ENRL_CERT_SUB'])
    #tmp1=claim_inprogress_data_client[cat_features_list_test_inprogress_client].values
    #cont_features = objects_map.get('cont-features')
    #tmp2=claim_inprogress_data_client[cont_features].values
    #print("type of tmp2 is ",type(tmp2),"  tmp2 values are",tmp2)
    #X_test = np.concatenate((tmp1,tmp2),axis=1)
    #print("X_test",X_test)
    estimator = objects_map.get('estimator')
    return_value=estimator.predict(X_inprogress_data)
    return_value_list=return_value.tolist()
    #type(return_value)
    json_string=json.dumps(return_value_list)
    return jsonify(claimstatus=json_string)
app.run()