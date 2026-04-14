# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 14:43:40 2025

@author: Edwin
"""

#%% Extract parameters
# to-do bshape
#| echo: false

import os
import sys
import pickle
import numpy as np
import pandas as pd
from great_tables import GT
from pathlib import Path

# Paths
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

#%% Extract parameters
params = np.array([
    ['A0',        '$A_0$',            'm',        'Musculoskeletal geometry',  '@van_soest_which_2000',               r'MTC length at $\phi_{knee}=0$'],
    ['A1',        '$A_1$',            'm/rad',    'Musculoskeletal geometry',  '@van_soest_which_2000',               r'Linear coefficient of MTC-$\phi_{knee}$ relation'],
    ['arel',      '$a^{rel}$',        '-',        'Contraction dynamics',      '@van_soest_which_2000',               r'Hill constant'],
    ['brel',      '$b^{rel}$',        '1/s',      'Contraction dynamics',      '@van_soest_which_2000',               r'Hill constant'],
    ['fmax',      '$F_{CE}^{max}$',   'N',        'Contraction dynamics',      '@van_soest_which_2000',               r'Maximum isometric CE force'],
    ['kpee',      '$k_{PEE}$',        'N/m^2^',   'Contraction dynamics',  '@van_soest_contribution_1993',            r'PEE stiffness scaling parameter'],
    ['ksee',      '$k_{SEE}$',        'N/m^2^',   'Contraction dynamics',      '@van_soest_which_2000',               r'SEE stiffness scaling parameter'],
    ['lce_opt',   '$L_{CE}^{opt}$',   'm',        'Contraction dynamics',      '@van_soest_which_2000',               r'CE optimum length'],
    ['lpee0',     '$L_{PEE}^0$',      'm',        'Contraction dynamics',  '@van_soest_contribution_1993',            r'PEE slack length'],
    ['lsee0',     '$L_{SEE}^0$',      'm',        'Contraction dynamics',      '@van_soest_which_2000',               r'SEE slack length'],
    ['w',         '$w$',              '-',        'Contraction dynamics',      '@van_soest_which_2000',               r'Determines CE force-length relation width'],
    ['a_act',     '$a_{act}$',        '-',        'Excitation dynamics',       '@hatze_myocybernetic_1981',           r'Determines steepness $\gamma$-$q$ relation'],
    ['b_act1',    '$b_{act,1}$',      '-',        'Excitation dynamics',       '@hatze_myocybernetic_1981',           r'Determines $pCa^{2+}$ level at which $q=0.5$'],
    ['b_act2',    '$b_{act,2}$',      '-',        'Excitation dynamics',       '@hatze_myocybernetic_1981',           r'Determines $pCa^{2+}$ level at which $q=0.5$'],
    ['b_act3',    '$b_{act,3}$',      '-',        'Excitation dynamics',       '@hatze_myocybernetic_1981',           r'Determines $pCa^{2+}$ level at which $q=0.5$'],
    ['kCa',       '$kCa$',            'mol/L',    'Excitation dynamics',       '@kistemaker_length-dependent_2005',   r'Relates $\gamma$ to $pCa^{2+}$'],
    ['tact',      '$\\tau_{act}$',    's',        'Excitation dynamics',       '@hatze_myocybernetic_1981',           r'Activation time constant'],
    ['tdeact',    '$\\tau_{deact}$',  's',        'Excitation dynamics',       '@hatze_myocybernetic_1981',           r'De-activation time constant'],
])
variables,symbols,units,partypes,references,descriptions = list(zip(*params))  # This transposes the list of lists

musparms = []
muspar = pickle.load(open(os.path.join(dataDir,'VAS_muspar.pkl'), 'rb'))
muspar['A2'] = 0
muspar['b_act1'], muspar['b_act2'], muspar['b_act3'] = muspar['b_act']
musparms = [muspar[k] for k in params[:,0]]
# musparms = ['0.21', '0.042', '0', '0.41', '5.2', '5250', r'$1.28 \cdot 10^8$', '0.093', '0.16', '0.56', '-4.59', '5.17', '1.08', '-0.19', r'$8.00 \cdot 10^{-6}$', r'$88.9 \cdot 10^{-3}$', r'$88.9 \cdot 10^{-3}$']

#%% Create pandas dataframe
df_or = pd.DataFrame(musparms, columns=['Value'], index=params[:,0])

df_or.insert(0, "Description", descriptions) 
df_or.insert(1, "Symbol", symbols) 
df_or.insert(2, "Unit", units) 
df_or.insert(4, "Reference", references) 
df_or.insert(5, "Partype", partypes) 

#%% Great table
from gt_tex import make_latex, insert_rows, fix_reference, delete_rows

tex_df = df_or.copy()
tex_df = tex_df.drop('Partype', axis=1)

gt_table = (GT(tex_df)
    .cols_align(align='left', columns=["Description","Parameter"])
)

citation_map = {
    "van_soest_which_2000": "Soest and Casius (2000)",
    "kistemaker_length-dependent_2005": "Kistemaker et al. (2005)",
    "hatze_myocybernetic_1981": "Hatze (1981)"
}

# Transform to LateX table
latex_str = make_latex(gt_table.as_latex())
latex_str = delete_rows(latex_str, row_numbers=[0])
add_rows = {
    0: r"  \bfseries Description & \bfseries Symbol & \bfseries Unit & \bfseries Value & \bfseries Reference \\ \hline",
    1: r"  \multicolumn{5}{|r|}{\itshape Musculoskeletal geometry} \\ \hline",
    5: r"  \multicolumn{5}{|r|}{\itshape Contraction dynamics} \\ \hline",
    13: r"  \multicolumn{5}{|r|}{\itshape Excitation dynamics} \\ \hline",
}

latex_str = insert_rows(latex_str, add_rows)
latex_str = fix_reference(latex_str,citation_map)
latex_str += (r"\break\hfill\footnotesize{"+ 
              r"Parameter values are depicted only for those key to predict the maximally attainable AMPO. }")
print(latex_str)

# Write to a .tex file
with open("tbl-s-parms.tex", "w", encoding="utf-8") as f:
    f.write(latex_str)