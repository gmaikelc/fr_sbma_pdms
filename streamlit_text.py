# -*- coding: utf-8 -*-
"""
Created on Fri July 19 17:41:37 2024

@author: Gerardo Casanola
"""


#%% Importing libraries

from pathlib import Path
import pandas as pd
import pickle
from molvs import Standardizer
from rdkit import Chem
from openbabel import openbabel
#from mordred import Calculator, descriptors
from multiprocessing import freeze_support
import numpy as np
from rdkit.Chem import AllChem
import plotly.graph_objects as go
import networkx as nx

#Import Libraries
import math 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# packages for streamlit
import streamlit as st
from PIL import Image
import io
import base64

from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdFingerprintGenerator, Descriptors, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataStructs import cDataStructs
from io import StringIO
#from mordred import Calculator, descriptors
import numpy as np
import pandas as pd
import seaborn as sns
import sys, os, shutil
import matplotlib.pyplot as plt
import streamlit as st
#from streamlit_ketcher import st_ketcher
import time
import subprocess
from PIL import Image
import uuid
#from filelock import Timeout, FileLock

#%% PAGE CONFIG

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Coating fouling release ML-predictor', page_icon=":computer:", layout='wide')

######
# Function to put a picture as header   
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

image = Image.open('cropped_header_fr2.png')
st.image(image)

# Create a layout with 3 columns
col1, col2, col3 = st.columns(3)

# Input for 'SBMA molecular weight'
sbma_mw = col1.number_input('SBMA molecular weight',max_value=6000.0,value=None)

if sbma_mw < 280.41 or sbma_mw > 6000.0:
    st.error("SBMA molecular weight must be between 280.41 and 6000.0.")

# Input for 'Percent'
pdms_mw = col2.number_input('PDMS molecular weight',max_value=11000.0,value = None)
if pdms_mw < 92.12 or pdms_mw > 11000.0:
    st.error("PDMS molecular weight must be between 92.12 and 11000.0.")

# Dropdown for 'PM' and 'DP'
options = ['0.2%', '1%','5%']
choice = col3.selectbox('Additive percent to add to the coating', options)

st.write(f"You entered {sbma_mw} for SBMA molecular weight, {pdms_mw} for PDMS molecular weight, and selected {choice} as percent to be added.")

sbma_mw_unit = 280.41
pdms_mw_unit = 92.12

fraction_sbma = sbma_mw/sbma_mw_unit
fraction_pdms = pdms_mw/pdms_mw_unit

st.write('fraction sbma:', fraction_sbma)
st.write('fraction pdms:', fraction_pdms)

descriptors_sbma_pdms = pd.read_csv("data/" + "descriptors_sbma_pdms.csv")

def reading_reorder(data):
        
    #Select the specified columns from the DataFrame
    df_selected = data[loaded_desc]
    df_id = data.reset_index()
    df_id.rename(columns={'index': 'NAME'}, inplace=True)
    id = df_id['NAME'] 
    # Order the DataFrame by the specified list of columns
    test_data = df_selected.reindex(columns=loaded_desc)
    #descriptors_total = data[loaded_desc]


def mixture_descriptors(data1, data2):
    # Extract component fractions
    sbma_mw_unit = 280.41
    pdms_mw_unit = 92.12

    fraction_sbma = sbma_mw/sbma_mw_unit
    fraction_pdms = pdms_mw/pdms_mw_unit
    
    component1 = fraction_sbma  #data['Component1']
    component2 = fraction_pdms #data['Component2']

    # Multiply corresponding rows in data1 and data2 for all columns
    df_mixture_left = component1.values[:, None] * test_data1.values
    df_mixture_right = component2.values[:, None] * test_data2.values

    # Create a new DataFrame using the result and set column names from data1 and data2
    df_mixture_left = pd.DataFrame(df_mixture_left, columns=test_data1.columns)
    df_mixture_right = pd.DataFrame(df_mixture_right, columns=test_data2.columns)

    # Initialize DataFrame for the final result
    df_sum_mixture = pd.DataFrame(index=test_data1.index)

    # Check if Component2 is 0, if so, only use the result from df_mixture_left
    for value in data['Component2']:
        if value == 0:
            df_sum_mixture = df_mixture_left
        else:
            # Sum the DataFrames row-wise by column name
            df_sum_mixture = df_mixture_left.add(df_mixture_right)
            # Set the index of df1 to match the index of df2
            df_sum_mixture.set_index(test_data1.index, inplace=True)

    return df_sum_mixture


def normalize_data(train_data, test_data):
    # Normalize the training data
    df_train = pd.DataFrame(train_data)
    saved_cols = df_train.columns
    min_max_scaler = preprocessing.MinMaxScaler().fit(df_train)
    np_train_scaled = min_max_scaler.transform(df_train)
    df_train_normalized = pd.DataFrame(np_train_scaled, columns=saved_cols)

    # Normalize the test data using the scaler fitted on training data
    np_test_scaled = min_max_scaler.transform(test_data)
    df_test_normalized = pd.DataFrame(np_test_scaled, columns=saved_cols)

    return df_train_normalized, df_test_normalized


def applicability_domain(x_test_normalized, x_train_normalized):
    y_train=data_train['pLC50_sw']
    X_train = x_train_normalized.values
    X_test = x_test_normalized.values
    # Calculate leverage and standard deviation for the training set
    hat_matrix_train = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T
    leverage_train = np.diagonal(hat_matrix_train)
    leverage_train=leverage_train.ravel()
    
    # Calculate leverage and standard deviation for the test set
    hat_matrix_test = X_test @ np.linalg.inv(X_train.T @ X_train) @ X_test.T
    leverage_test = np.diagonal(hat_matrix_test)
    leverage_test=leverage_test.ravel()


    from sklearn.metrics import mean_squared_error

    # Train a linear regression model
    lr = LinearRegression()
    lr.fit(df_train_normalized, y_train)
    y_pred_train = lr.predict(df_train_normalized)
    
    std_dev_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    std_residual_train = (y_train - y_pred_train) / std_dev_train
    std_residual_train = std_residual_train.ravel()
    
    # threshold for the applicability domain
    
    h3 = 3*((x_train_normalized.shape[1]+1)/x_train_normalized.shape[0])  
    
    diagonal_compare = list(leverage_test)
    h_results =[]
    for valor in diagonal_compare:
        if valor < h3:
            h_results.append(True)
        else:
            h_results.append(False)         
    return h_results, leverage_train, leverage_test, std_residual_train 

# Function to assign colors based on confidence values
def get_color(confidence):
    """
    Assigns a color based on the confidence value.

    Args:
        confidence (float): The confidence value.

    Returns:
        str: The color in hexadecimal format (e.g., '#RRGGBB').
    """
    # Define your color logic here based on confidence
    if confidence == "HIGH" or confidence == "Inside AD":
        return 'green'
    elif confidence == "MEDIUM":
        return 'yellow'
    else:
        confidence ==  "LOW"
        return 'red'



#%% Predictions   

def predictions(loaded_model, loaded_desc, df_test_normalized):
    scores = []
    h_values = []
    std_resd = []
    idx = data['ID']
    

    descriptors_model = loaded_desc
    # Placeholder for the spinner
    with st.spinner('CALCULATING PREDICTIONS (STEP 2 OF 3)...'):
        # Simulate a long-running computation
        time.sleep(1)  # Sleep for 5 seconds to mimic computation
     
        X = df_test_normalized[descriptors_model]
        predictions = loaded_model.predict(X)
        scores.append(predictions)
        
        # y_true and y_pred are the actual and predicted values, respectively
    
        # Create y_true array with all elements set to mean value and the same length as y_pred
        y_pred_test = predictions
        y_test = np.full_like(y_pred_test, mean_value)
        residuals_test = y_test -y_pred_test

        std_dev_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        std_residual_test = (y_test - y_pred_test) / std_dev_test
        std_residual_test = std_residual_test.ravel()
          
        std_resd.append(std_residual_test)
        
        h_results, leverage_train, leverage_test, std_residual_train  = applicability_domain(df_test_normalized, df_train_normalized)
        h_values.append(h_results)
    

        dataframe_pred = pd.DataFrame(scores).T
        dataframe_pred.index = idx
        dataframe_pred.rename(columns={0: "pLC50"},inplace=True)
    
        dataframe_std = pd.DataFrame(std_resd).T
        dataframe_std.index = idx
          
        
        h_final = pd.DataFrame(h_values).T
        h_final.index = idx
        h_final.rename(columns={0: "Confidence"},inplace=True)

        std_ensemble = dataframe_std.iloc[:,0]
        # Create a mask using boolean indexing
        std_ad_calc = (std_ensemble >= 3) | (std_ensemble <= -3) 
        std_ad_calc = std_ad_calc.replace({True: 'Outside AD', False: 'Inside AD'})
   
    
        final_file = pd.concat([std_ad_calc,h_final,dataframe_pred], axis=1)
    
        final_file.rename(columns={0: "Std_residual"},inplace=True)
    
        h3 = 3*((df_train_normalized.shape[1]+1)/df_train_normalized.shape[0])  ##  Mas flexible

        final_file.loc[(final_file["Confidence"] == True) & ((final_file["Std_residual"] == 'Inside AD' )), 'Confidence'] = 'HIGH'
        final_file.loc[(final_file["Confidence"] == True) & ((final_file["Std_residual"] == 'Outside AD')), 'Confidence'] = 'LOW'
        final_file.loc[(final_file["Confidence"] == False) & ((final_file["Std_residual"] == 'Outside AD')), 'Confidence'] = 'LOW'
        final_file.loc[(final_file["Confidence"] == False) & ((final_file["Std_residual"] == 'Inside AD')), 'Confidence'] = 'MEDIUM'


            
        df_no_duplicates = final_file[~final_file.index.duplicated(keep='first')]
        styled_df = df_no_duplicates.style.apply(lambda row: [f"background-color: {get_color(row['Confidence'])}" for _ in row],subset=["Confidence"], axis=1)
    
        return final_file, styled_df,leverage_train,std_residual_train, leverage_test, std_residual_test

#Data C. lytica at 10 psi
data_clyt10psi = pd.read_csv("data/" + "dataset_clytica10psi_norm_ascending_Series_p1.csv")
data_train_clyt10psi = data_clyt10psi[data_clyt10psi['Series_p1'] == 'Training'] 
mean_value_clyt10psi = data_train_clyt10psi['c_lytica_removal_at_10psi'].mean()
loaded_model = pickle.load(open("models/" + "model_clyt_10psi_rf.pickle", 'rb'))
loaded_desc = pickle.load(open("models/" + "descriptor_clyt_10psi_rf.pickle", 'rb'))


#Data C. lytica at 20 psi
data_clyt20psi = pd.read_csv("data/" + "dataset_clytica10psi_norm_ascending_Series_p1.csv")
data_train_clyt20psi = data_clyt20psi[data_clyt20psi['Series_p1'] == 'Training'] 
mean_value_clyt20psi = data_train_clyt10psi['c_lytica_removal_at_10psi'].mean()
loaded_model = pickle.load(open("models/" + "model_clyt_10psi_rf.pickle", 'rb'))
loaded_desc = pickle.load(open("models/" + "descriptor_clyt_10psi_rf.pickle", 'rb'))

#data_train = pd.read_csv("data/" + "data_126c_15var_pLC50_train_sw.csv")
#mean_value = data_train['pLC50_sw'].mean()
#loaded_model = pickle.load(open("models/" + "ml_model_rotifer_sw.pickle", 'rb'))
#loaded_desc = pickle.load(open("models/" + "ml_descriptor_rotifer_sw.pickle", 'rb'))


#Footer edit

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made in  🐍 and <img style='display: ; 
' href="https://streamlit.io" src="https://i.imgur.com/iIOA6kU.png" target="_blank"></img> Developed by <a style='display: ;
 text-align: center' href="https://www.linkedin.com/in/gerardo-m-casanola-martin-27238553/" target="_blank">Gerardo M. Casanola</a> for <a style='display: ; 
 text-align: center;' href="http://www.rasulev.org" target="_blank">Rasulev Research Group</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
