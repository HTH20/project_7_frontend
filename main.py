import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import requests
import pickle
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import shap

df = pd.read_csv('data_request.csv')
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)

def test_test():
    json_test={
      "CODE_GENDER_F": 0,
      "CODE_GENDER_M": 0,
      "CODE_REJECT_REASON_CLIENT_mean": 0,
      "CODE_REJECT_REASON_HC_mean": 0,
      "CODE_REJECT_REASON_LIMIT_mean": 0,
      "CODE_REJECT_REASON_SCOFR_mean": 0,
      "CODE_REJECT_REASON_SCO_mean": 0,
      "CODE_REJECT_REASON_SYSTEM_mean": 0,
      "CODE_REJECT_REASON_VERIF_mean": 0,
      "CODE_REJECT_REASON_XAP_mean": 0,
      "CODE_REJECT_REASON_XNA_mean": 0,
      "EXT_SOURCE_2": 0,
      "EXT_SOURCE_3": 0,
      "FLAG_OWN_CAR_N": 0,
      "FLAG_OWN_CAR_Y": 0,
      "NAME_EDUCATION_TYPE_Academic_degree": 0,
      "NAME_EDUCATION_TYPE_Higher_education": 0,
      "NAME_EDUCATION_TYPE_Incomplete_higher": 0,
      "NAME_EDUCATION_TYPE_Lower_secondary": 0,
      "NAME_EDUCATION_TYPE_Secondary_secondary_special": 0
    }

    response = requests.post('http://127.0.0.1:8000/predict', json=json_test)
    return response.content
    
def graph(data, result):

    features = df.columns[:-1]
    X = df[features]
    pca = PCA(n_components=0.97)
    X_pca = pca.fit_transform(X)
    
    tree = KDTree(X_pca)
    X = pca.transform(data)
    ds, inds =  tree.query((X), 50)

    df_copy = df.loc[inds[0].tolist()].copy()
    data[0].append(result)
    df_copy.loc[len(df_copy)] = data[0]
    for col,values in zip(df_copy.columns[:-1], df_copy.iloc[-1]):
        if values:
            if df[col].dtypes == np.dtype('int64'):
                fig = plt.figure(figsize=(10, 4))
                sb.barplot(x=df_copy.loc[df[col] == 1].TARGET.value_counts().index, 
                           y=df_copy.loc[df[col] == 1].TARGET.value_counts())
                plt.title(col)
                st.pyplot(fig)       
    cols = [col for col in df_copy if 'CODE_REJECT' in col or 'EXT_SOURCE' in col]
    fig, axs = plt.subplots(2, 9, figsize=(40, 20))
    for index_e, col_EXT in enumerate(cols[-2:]):
        for index_c, col_CODE_REJECT in enumerate(cols[:-2]):
            sb.scatterplot(x = df_copy[col_CODE_REJECT], 
                           y = df_copy[col_EXT], 
                           hue = df_copy.TARGET,
                           ax=axs[index_e, index_c]
                           )
                           
            axs[index_e, index_c].scatter(x = df_copy.iloc[-1][col_CODE_REJECT], 
                        y = df_copy.iloc[-1][col_EXT],
                        color= 'red')
            axs[index_e, index_c].set_xlim(-0.1, 1.1)
    st.pyplot(fig)
    
    features = df.columns[:-1]
    X = df[features]
    shap.initjs()
    explainer = shap.TreeExplainer(model, data=X)
    shap_values = explainer(data)
    fig = shap.plots.bar(shap_values)
    st.pyplot(fig)

def main():

    st.title('Scoring Credit')
        
    CODE_GENDER = st.radio('Select gender:', ['Female', 'Male'], horizontal = True)
    
    st.write('code reject reason :')
    
    CODE_REJECT_REASON_CLIENT_mean = st.number_input('CLIENT', min_value=0, value=0, step=1)
    CODE_REJECT_REASON_HC_mean = st.number_input('HC', min_value=0, value=0, step=1)
    CODE_REJECT_REASON_LIMIT_mean = st.number_input('LIMIT', min_value=0, value=0, step=1)
    CODE_REJECT_REASON_SCOFR_mean = st.number_input('SCOFR', min_value=0, value=0, step=1)
    CODE_REJECT_REASON_SCO_mean = st.number_input('SCO', min_value=0, value=0, step=1)
    CODE_REJECT_REASON_SYSTEM_mean = st.number_input('SYSTEM', min_value=0, value=0, step=1)
    CODE_REJECT_REASON_VERIF_mean = st.number_input('VERIF', min_value=0, value=0, step=1)
    CODE_REJECT_REASON_XAP_mean = st.number_input('XAP', min_value=0, value=0, step=1)
    CODE_REJECT_REASON_XNA_mean = st.number_input('XNA', min_value=0, value=0, step=1)
    
    EXT_SOURCE_2 = st.number_input('EXT_SOURCE_2', min_value=0., value=df['EXT_SOURCE_2'].mean(), step=0.01)
                              
    EXT_SOURCE_3 = st.number_input('EXT_SOURCE_3', min_value=0., value=df['EXT_SOURCE_3'].mean(), step=0.01)
    
    FLAG_OWN_CAR = st.radio('Own car:', ['No', 'Yes'], horizontal = True)
    
    NAME_EDUCATION_TYPE = st.radio('Education Type:', ['Academic degree', 'Higher education', 'Incomplete higher',
                                                       'Lower secondary','Secondary / secondary special'],
                                                       horizontal = True)

    predict_btn = st.button('Prédire')

    if predict_btn:
        if CODE_GENDER == 'Female':
            CODE_GENDER = [1,0]
        else:
            CODE_GENDER = [0,1]
            
        CODE_REJECT_TOTAL = (CODE_REJECT_REASON_CLIENT_mean + CODE_REJECT_REASON_HC_mean + CODE_REJECT_REASON_LIMIT_mean +
                             CODE_REJECT_REASON_SCOFR_mean + CODE_REJECT_REASON_SCO_mean + CODE_REJECT_REASON_SYSTEM_mean +
                             CODE_REJECT_REASON_VERIF_mean + CODE_REJECT_REASON_XAP_mean + CODE_REJECT_REASON_XNA_mean)
                             
        if CODE_REJECT_TOTAL > 0:
            CODE_REJECT_REASON_CLIENT_mean = ( CODE_REJECT_REASON_CLIENT_mean / CODE_REJECT_TOTAL )
            CODE_REJECT_REASON_HC_mean = ( CODE_REJECT_REASON_HC_mean / CODE_REJECT_TOTAL )
            CODE_REJECT_REASON_LIMIT_mean = ( CODE_REJECT_REASON_LIMIT_mean / CODE_REJECT_TOTAL )
            CODE_REJECT_REASON_SCOFR_mean = ( CODE_REJECT_REASON_SCOFR_mean / CODE_REJECT_TOTAL )
            CODE_REJECT_REASON_SCO_mean = ( CODE_REJECT_REASON_SCO_mean / CODE_REJECT_TOTAL )
            CODE_REJECT_REASON_SYSTEM_mean = ( CODE_REJECT_REASON_SYSTEM_mean / CODE_REJECT_TOTAL )
            CODE_REJECT_REASON_VERIF_mean = ( CODE_REJECT_REASON_VERIF_mean / CODE_REJECT_TOTAL )
            CODE_REJECT_REASON_XAP_mean = ( CODE_REJECT_REASON_XAP_mean / CODE_REJECT_TOTAL )
            CODE_REJECT_REASON_XNA_mean = ( CODE_REJECT_REASON_XNA_mean / CODE_REJECT_TOTAL )
        
        if FLAG_OWN_CAR == 'No':
            FLAG_OWN_CAR = [1,0]
        else:
            FLAG_OWN_CAR = [0,1]
            
        if NAME_EDUCATION_TYPE == 'Academic degree':
            NAME_EDUCATION_TYPE = [1,0,0,0,0]
            
        elif NAME_EDUCATION_TYPE == 'Higher education':
            NAME_EDUCATION_TYPE = [0,1,0,0,0]
            
        elif NAME_EDUCATION_TYPE == 'Incomplete higher':
            NAME_EDUCATION_TYPE = [0,0,1,0,0]
            
        elif NAME_EDUCATION_TYPE == 'Lower secondary':
            NAME_EDUCATION_TYPE = [0,0,0,1,0]
            
        else:
            NAME_EDUCATION_TYPE = [0,0,0,0,1]

        data = [CODE_GENDER + [CODE_REJECT_REASON_CLIENT_mean, CODE_REJECT_REASON_HC_mean, CODE_REJECT_REASON_LIMIT_mean, CODE_REJECT_REASON_SCOFR_mean, CODE_REJECT_REASON_SCO_mean, CODE_REJECT_REASON_SYSTEM_mean,
                CODE_REJECT_REASON_VERIF_mean, CODE_REJECT_REASON_XAP_mean, CODE_REJECT_REASON_XNA_mean, EXT_SOURCE_2, EXT_SOURCE_3] + FLAG_OWN_CAR + NAME_EDUCATION_TYPE]
        
        pred = test_test()
        
        if pred[0] == 0:
            st.success('credit accorder')
            graph(data, 0)
        else:
            st.success('credit non accorder')
            graph(data, 1)
            
if __name__ == '__main__':
    main()