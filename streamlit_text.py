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

image = Image.open('cropped_header_fr.png')
st.image(image)

# Create a layout with 3 columns
col1, col2, col3 = st.columns(3)

# Input for 'SBMA molecular weight'
sbma_mw = col1.number_input('SBMA molecular weight', min_value=0, max_value = 6000)

# Input for 'Percent'
pdms_mw = col2.number_input('PDMS molecular weight', min_value=0, max_value = 11000)

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


def generate_si_oil_pattern(choice, sbma_mw, pdms_mw, DM='DM', DP='DP', PM='PM'):
    # Calculate Number_Rep_Unit_1
    st.write('percent:',percent)
    Number_Rep_Unit_2 = round(percent / 100 * degree_of_polymerization)

    

    # Perform the division and get the integer quotient and remainder
    Number_Rep_Unit_1 = degree_of_polymerization - Number_Rep_Unit_2 - 2  # Assuming a value for DiMethyl for the code to run
    ratio_rep_unit, remainder = divmod(Number_Rep_Unit_1, Number_Rep_Unit_2 + 1)
    num_it = Number_Rep_Unit_2

    # Specify the number of times to repeat the pattern
    num_repeats = num_it

    # Define name_dm, name_pm, and name_dp based on ratio_rep_unit and num_1
    name_dm = f'-[{DM}({ratio_rep_unit})]'
    name_pm = f'-[{PM}({num_1})]'
    name_dp = f'-[{DP}({num_1})]'

    # Construct the pattern based on the choice
    if choice == 'DP':
        pattern = f'{name_dm}{name_dp}' * num_repeats
    elif choice == 'PM':
        pattern = f'{name_dm}{name_pm}' * num_repeats
    else:
        pattern = ''  # Handle invalid choice gracefully

    # Print the pattern along with left and right ends
    si_oil_pattern = f'{name_left_end}{pattern}{name_dm}{name_right_end}'

    return si_oil_pattern

def generate_si_oil(choice, percent, degree_of_polymerization):
    # Calculate Number_Rep_Unit_2
    st.write('percent:',percent)
    Number_Rep_Unit_2 = round(percent/100 * degree_of_polymerization)
    st.write(f'Number of repeating units for {choice} :', Number_Rep_Unit_2)

    # Perform the division and get the integer quotient and remainder
    Number_Rep_Unit_1 = degree_of_polymerization - Number_Rep_Unit_2 - 2  # Assuming a value for DiMethyl for the code to run
    ratio_rep_unit, remainder = divmod(Number_Rep_Unit_1, Number_Rep_Unit_2 + 1)

    # Print the results
    st.write("Ratio (Integer part):", ratio_rep_unit)
    st.write("Remainder:", remainder)

    # Number of iterations
    num_it = Number_Rep_Unit_2
    st.write('Number of iterations:', num_it)
    st.write('RU:', choice)

    # Perform the concatenation for the specified number of iterations

    # Initialize the final string with the left end
    end_ru = DM * ratio_rep_unit
    f_ru = ''

    # Depending on the choice, add the string DP or PM at the end of f_ru after each iteration
    if choice == 'PM':
        for i in range(num_it):
            # Add the DM string ratio_DM times
            f_ru += (DM * ratio_rep_unit) + PM

    if choice == 'DP':
        for i in range(num_it):
            # Add the DM string ratio_DM times
            f_ru += (DM * ratio_rep_unit) + DP
            
            
    n_ru = f_ru + end_ru

    si_oil = left_end + n_ru + right_end

    # Remove the patterns
    si_oil_final = si_oil.replace('**', '').replace('I*', '')

    return si_oil_final


if st.button('Press to see the pattern for the assembled silicon oil'):
  # Example usage:
  si_oil_pattern_generated = generate_si_oil_pattern(choice, percent, degree_of_polymerization)
  st.write(si_oil_pattern_generated)

if st.button('Press to generate the silicon oil structure based on the parameters'):

# Example usage:
  si_oil_generated = generate_si_oil(choice, percent, degree_of_polymerization)
  st.write(si_oil_generated)


  mol = Chem.MolFromSmiles(si_oil_generated)
  mol = Chem.AddHs(mol)
  AllChem.EmbedMolecule(mol, AllChem.ETKDG())

 # Save as a mol file
 #save_filename = st.text_input("Enter filename for MOL file")
 #if st.button("Save MOL"):
   #with open(save_filename, "w") as f:
     #writer = Chem.SDWriter(f)
     #writer.write(mol)
     #writer.close()
   #st.success(f"Mol file saved as {save_filename}")
 #writer = Chem.SDWriter(filename)
 #writer.write(mol)
 #writer.close()


#def smiles_to_mol(smiles):
 #   mol = Chem.MolFromSmiles(smiles)
  #  if mol:
   #     mol = Chem.AddHs(mol)
    #    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
     #   return mol
    #else:
     #   return None

#def save_mol_file(mol, filename):
 #   writer = Chem.SDWriter(filename)
  #  writer.write(mol)
   # writer.close()

# Convert SMILES to mol
#mol = smiles_to_mol(si_oil_final)

# Save as a mol file
#save_filename = "output.mol"
#save_mol_file(mol, save_filename)
