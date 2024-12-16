# -*- coding: utf-8 -*-
"""
Created on Fri December 13 14:41:37 2024

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
col4, col5, col6 = st.columns(3)

# Input for 'SBMA molecular weight'
sbma_mw = col1.number_input('SBMA molecular weight',max_value=6000.0)

if sbma_mw < 280.41 or sbma_mw > 6000.0:
    col4.error("SBMA molecular weight must be between 280.41 and 6000.0.")

# Input for 'Percent'
pdms_mw = col2.number_input('PDMS molecular weight',max_value=11000.0)
if pdms_mw < 92.12 or pdms_mw > 11000.0:
    col5.error("PDMS molecular weight must be between 92.12 and 11000.0.")

# Dropdown for 'PM' and 'DP'
options = ['0.2%', '1%','5%']
choice = col3.selectbox('Additive percent to add to the coating', options)

# Convert the selected percentage to a float
percentage = float(choice.strip('%'))

st.write(f"You entered {sbma_mw} for SBMA molecular weight, {pdms_mw} for PDMS molecular weight, and selected {choice} as percent to be added.")

sbma_mw_unit = 280.41
pdms_mw_unit = 92.12

fraction_sbma = sbma_mw/sbma_mw_unit
fraction_pdms = pdms_mw/pdms_mw_unit

st.write('fraction sbma:', fraction_sbma)
st.write('fraction pdms:', fraction_pdms)

descriptors_sbma_pdms = pd.read_csv("data/" + "descriptors_sbma_pdms.csv")

def reading_reorder(data, loaded_desc):
        
    #Select the specified columns from the DataFrame
    df_selected = data[loaded_desc]
    df_id = data.reset_index()
    df_id.rename(columns={'MolID': 'NAME'}, inplace=True)
    id = df_id['NAME'] 
    # Order the DataFrame by the specified list of columns
    test_data = df_selected.reindex(columns=loaded_desc)
    # Fill missing values with 0
    #test_data = test_data.fillna(0)
    #descriptors_total = data[loaded_desc]

    return test_data, id


def mixture_descriptors(data1, data2):
    # Extract component fractions
    #sbma_mw_unit = 280.41
    #pdms_mw_unit = 92.12

    #fraction_sbma = sbma_mw/sbma_mw_unit
    #fraction_pdms = pdms_mw/pdms_mw_unit
    
    component1 = fraction_sbma  #data['Component1']
    component2 = fraction_pdms #data['Component2']

    # Multiply corresponding rows in data1 and data2 for all columns
    df_mixture_left = component1* test_data1
    df_mixture_right = component2* test_data2

    st.dataframe(df_mixture_left)
    st.dataframe(df_mixture_right)

    df_mixture_left = df_mixture_left.reset_index(drop=True)
    df_mixture_right = df_mixture_right.reset_index(drop=True)
    
    # Sum the DataFrames row-wise by column name
    df_sum_mixture_ini = df_mixture_left.add(df_mixture_right)
    # Remove the column index from the dataframe 
    df_sum_mixture_ini = df_sum_mixture_ini.iloc[:,0:]
    st.write('dataframe mixture descriptors')
    st.dataframe(df_sum_mixture_ini)
    st.write(choice)
    # Multiply the DataFrame by the selected percentage
    df_sum_mixture = df_sum_mixture_ini * percentage
    st.write('dataframe  by percent added')
    st.dataframe(df_sum_mixture)

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

    st.write('Normalized data')
    st.dataframe(df_test_normalized)

    st.write('Train normalized')
    st.dataframe(df_train_normalized.head(5))
                 
    return df_train_normalized, df_test_normalized


def applicability_domain(x_test_normalized, x_train_normalized):
    y_train=data_train_1['c_lytica_removal_at_10psi']
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

    # Train a random forest model
    from sklearn.ensemble import RandomForestRegressor


    lr = RandomForestRegressor(
        n_estimators=100,        # Number of trees in the forest
        max_depth=10,           # Max depth of each tree
        min_samples_split=2,    # Minimum samples required to split an internal node
        random_state=42         # For reproducibility
    )
    
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

#Calculating the William's plot limits
def calculate_wp_plot_limits(leverage_train,std_residual_train, x_std_max=4, x_std_min=-4):
    
    with st.spinner('CALCULATING APPLICABILITY DOMAIN (STEP 3 OF 3)...'):
        # Simulate a long-running computation
        time.sleep(1)  # Sleep for 5 seconds to mimic computation
        # Getting maximum std value
        if std_residual_train.max() < 4:
            x_lim_max_std = x_std_max
        elif std_residual_train.max() > 4:
            x_lim_max_std = round(std_residual_train.max()) + 1

        # Getting minimum std value
        if std_residual_train.min() > -4:
            x_lim_min_std = x_std_min
        elif std_residual_train.min() < 4:
            x_lim_min_std = round(std_residual_train.min()) - 1

    
        #st.write('x_lim_max_std:', x_lim_max_std)
        #st.write('x_lim_min_std:', x_lim_min_std)

        # Calculation H critical
        n = len(leverage_train)
        p = df_train_normalized.shape[1]
        h_value = 3 * (p + 1) / n
        h_critical = round(h_value, 4)
        #st.write('Number of cases training:', n)
        #st.write('Number of variables:', p)
        #st.write('h_critical:', h_critical)

        # Getting maximum leverage value
        if leverage_train.max() < h_critical:
            x_lim_max_lev = h_critical + h_critical * 0.5
        elif leverage_train.max() > h_critical:
            x_lim_max_lev = leverage_train.max() + (leverage_train.max()) * 0.1

        # Getting minimum leverage value
        if leverage_train.min() < 0:
            x_lim_min_lev = x_lev_min - x_lev_min * 0.05
        elif leverage_train.min() > 0:
            x_lim_min_lev = 0

        #st.write('x_lim_max_lev:', x_lim_max_lev)

        return x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def williams_plot(leverage_train, leverage_test, std_residual_train, std_residual_test,id_list_1,
                  plot_color='cornflowerblue', show_plot=True, save_plot=False, filename=None, add_title=False, title=None):
    fig = go.Figure()

    # Add training data points
    fig.add_trace(go.Scatter(
        x=leverage_train,
        y=std_residual_train,
        mode='markers',
        marker=dict(color='cornflowerblue', size=10, line=dict(width=1, color='black')),
        name='Training'
    ))

    # Add test data points
    fig.add_trace(go.Scatter(
        x=leverage_test,
        y=std_residual_test,
        mode='markers',
        marker=dict(color='orange', size=10, line=dict(width=1, color='black')),
        name='Prediction',
        text = id_list_1, # Add compounds IDs for hover
        hoverinfo = 'text' #Show only the text when hovering
    ))

    # Add horizontal and vertical dashed lines
    fig.add_shape(type='line', x0=h_critical, y0=x_lim_min_std, x1=h_critical, y1=x_lim_max_std,
                  line=dict(color='black', dash='dash'))
    fig.add_shape(type='line', x0=x_lim_min_lev, y0=3, x1=x_lim_max_lev, y1=3,
                  line=dict(color='black', dash='dash'))
    fig.add_shape(type='line', x0=x_lim_min_lev, y0=-3, x1=x_lim_max_lev, y1=-3,
                  line=dict(color='black', dash='dash'))

    # Add rectangles for outlier zones
    fig.add_shape(type='rect', x0=x_lim_min_lev, y0=x_lim_min_std, x1=h_critical, y1=-3,
                  fillcolor='lightgray', opacity=0.4, line_width=0)
    fig.add_shape(type='rect', x0=x_lim_min_lev, y0=3, x1=h_critical, y1=x_lim_max_std,
                  fillcolor='lightgray', opacity=0.4, line_width=0)
                      
    fig.add_shape(type='rect', x0=h_critical, y0=x_lim_min_std, x1=x_lim_max_lev, y1=-3,
                  fillcolor='lightgray', opacity=0.4, line_width=0)
    fig.add_shape(type='rect', x0=h_critical, y0=3, x1=x_lim_max_lev, y1=x_lim_max_std,
                  fillcolor='lightgray', opacity=0.4, line_width=0)

    # Add annotations for outlier zones
    fig.add_annotation(x=(h_critical + x_lim_min_lev) / 2, y=-3.5, text='Outlier zone', showarrow=False,
                       font=dict(size=15))
    fig.add_annotation(x=(h_critical + x_lim_min_lev) / 2, y=3.5, text='Outlier zone', showarrow=False,
                       font=dict(size=15))
    fig.add_annotation(x=(h_critical + x_lim_max_lev) / 2, y=-3.5, text='Outlier zone', showarrow=False,
                       font=dict(size=15))
    fig.add_annotation(x=(h_critical + x_lim_max_lev) / 2, y=3.5, text='Outlier zone', showarrow=False,
                       font=dict(size=15))

    # Update layout
    fig.update_layout(
        width=600,
        height=600,
        xaxis=dict(title='Leverage', range=[x_lim_min_lev, x_lim_max_lev], tickfont=dict(size=15)),
        yaxis=dict(title='Std Residuals', range=[x_lim_min_std, x_lim_max_std], tickfont=dict(size=15)),
        legend=dict(x=0.99, y=0.825, xanchor='right', yanchor='top', font=dict(size=20)),
        showlegend=True
    )

    if add_title and title:
        fig.update_layout(title=dict(text=title, font=dict(size=20)))

    if save_plot and filename:
        fig.write_image(filename)

    if show_plot:
        fig.show()

    return fig

run = st.button("Click to make prediction for the drawn structure")
    if run == True

        ID='1'
        data = pd.DataFrame({'ID': [ID],})

        #Data C. lytica at 10 psi
        #data_clyt10psi = pd.read_csv("data/" + "dataset_clytica10psi_original_asc_Series_p1_traininig.csv")
        #data_train_clyt10psi = data_clyt10psi[data_clyt10psi['Series_p1'] == 'Training'] 
        data_train_1 = pd.read_csv("data/" + "dataset_clytica10psi_original_asc_Series_p1_traininig.csv")
        mean_value_1 = data_train_1['c_lytica_removal_at_10psi'].mean()
        loaded_model = pickle.load(open("models/" + "model_clyt_10psi_rf.pickle", 'rb'))
        loaded_desc = pickle.load(open("models/" + "descriptor_clyt_10psi_rf.pickle", 'rb'))

        train_data = data_train_1[loaded_desc]
        #Selecting the descriptors based on model for first component
        #st.dataframe(descriptors_sbma_pdms)
        descriptors_sbma = descriptors_sbma_pdms.iloc[0:1,:]
        descriptors_pdms = descriptors_sbma_pdms.iloc[1:2,:]
        #st.dataframe(descriptors_sbma)
        test_data1, id_list_1 =  reading_reorder(descriptors_sbma,loaded_desc)
        #Selecting the descriptors based on model for first component
        test_data2, id_list_2 =  reading_reorder(descriptors_pdms,loaded_desc)

        # Display the dataframe in Streamlit
        #st.dataframe(test_data1)
        #st.dataframe(test_data2)

         #Calculating mixture descriptors    
        test_data_mix= mixture_descriptors(test_data1,test_data2)
        #st.dataframe(test_data_mix)
        #test_data_mix.fillna(0,inplace=True)
        #st.markdown(filedownload4(test_data_mix), unsafe_allow_html=True)

        X_final2= test_data_mix
        df_train_normalized, df_test_normalized = normalize_data(train_data, X_final2)               

        #X_final1, id = all_correct_model(test_data_mix,loaded_desc, id_list)

        st.dataframe(data_train_1.head(5))

        #st.dataframe(data.head(5))

        final_file, styled_df,leverage_train,std_residual_train, leverage_test, std_residual_test= predictions(loaded_model, loaded_desc, df_test_normalized)
        #final_file2, styled_df2,leverage_train2,std_residual_train2, leverage_test2, std_residual_test2= predictions2(loaded_model2, loaded_desc2, df_test_normalized2)
        
        x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev = calculate_wp_plot_limits(leverage_train,std_residual_train, x_std_max=4, x_std_min=-4)
        #x_lim_max_std2, x_lim_min_std2, h_critical2, x_lim_max_lev2, x_lim_min_lev2 = calculate_wp_plot_limits2(leverage_train2,std_residual_train2, x_std_max2=4, x_std_min2=-4)
        
        figure  = williams_plot(leverage_train, leverage_test, std_residual_train, std_residual_test,id_list_1)
        #figure2  = williams_plot2(leverage_train2, leverage_test2, std_residual_train2, std_residual_test2,id_list_2)


#st.markdown(filedownload5(df_test_normalized), unsafe_allow_html=True)
#final_file, styled_df = predictions(loaded_model, loaded_desc, df_test_normalized)
#figure  = final_plot(final_file)  
#col1, col2 = st.columns(2)


#Data C. lytica at 20 psi
#data_clyt20psi = pd.read_csv("data/" + "dataset_clytica10psi_norm_ascending_Series_p1.csv")
#data_train_clyt20psi = data_clyt20psi[data_clyt20psi['Series_p1'] == 'Training'] 
#mean_value_clyt20psi = data_train_clyt10psi['c_lytica_removal_at_10psi'].mean()
#loaded_model = pickle.load(open("models/" + "model_clyt_10psi_rf.pickle", 'rb'))
#loaded_desc = pickle.load(open("models/" + "descriptor_clyt_10psi_rf.pickle", 'rb'))

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
