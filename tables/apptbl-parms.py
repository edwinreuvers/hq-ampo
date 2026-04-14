import os
import sys
import pickle
import numpy as np
import pandas as pd
from great_tables import GT, style, loc
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
    ['kpee',      '$k_{PEE}$',        'N/m^2^',   'Contraction dynamics',       '@van_soest_contribution_1993',            r'PEE stiffness scaling parameter'],
    ['ksee',      '$k_{SEE}$',        'N/m^2^',   'Contraction dynamics',      '@van_soest_which_2000',               r'SEE stiffness scaling parameter'],
    ['lce_opt',   '$L_{CE}^{opt}$',   'm',        'Contraction dynamics',      '@van_soest_which_2000',               r'CE optimum length'],
    ['lpee0',     '$L_{PEE}^0$',      'm',        'Contraction dynamics',       '@van_soest_contribution_1993',            r'PEE slack length'],
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
muspar['b_act1'], muspar['b_act2'], muspar['b_act3'] = muspar['b_act']
musparms = [muspar[k] for k in params[:,0]]
# musparms = ['0.21', '0.042', '0', '0.41', '5.2', '5250', r'$1.28 \cdot 10^8$', '0.093', '0.16', '0.56', '-4.59', '5.17', '1.08', '-0.19', r'$8.00 \cdot 10^{-6}$', r'$88.9 \cdot 10^{-3}$', r'$88.9 \cdot 10^{-3}$']

#%% Create pandas dataframe
df = pd.DataFrame(musparms, columns=['Value'], index=params[:,0])

df.insert(0, "Description", descriptions) 
df.insert(1, "Symbol", symbols) 
df.insert(2, "Unit", units) 
df.insert(4, "Reference", references) 
df.insert(5, "Partype", partypes) 

#%% Some are in other units so..
for parm in ['A0', 'A1', 'lce_opt', 'lpee0', 'lsee0']: # from m to cm
    df.loc[parm, 'Value'] = df.loc[parm, 'Value'] * 1e2
    df.loc[parm, 'Unit'] = 'cm'
for parm in ['tact', 'tdeact']: # from s to ms
    df.loc[parm, 'Value'] = df.loc[parm, 'Value'] * 1e3
    df.loc[parm, 'Unit'] = 'ms'
df.loc['A0', 'Unit'] = 'cm/rad'

#%% Great table
from gt_tex import make_latex, insert_rows, fix_reference, delete_rows

df_tex = df.copy()
df_tex = df_tex.drop('Partype', axis=1)

gt_table = (GT(df_tex)
    .cols_align(align='left', columns=["Description","Parameter"])
    .fmt_number(columns=["Value"], n_sigfig=3)
    .fmt_number(columns=["Value"], n_sigfig=2, rows=[2,10,14]) #arel, w, bact3
    .fmt_number(columns=["Value"], n_sigfig=4, rows=[4]) # fmax
    .fmt_scientific(columns=["Value"], n_sigfig=3, rows=[5,6,15]) #kpee,ksee,kca
)
gt_table

citation_map = {
    "van_soest_which_2000": "van Soest & Casius (2000)",
    "kistemaker_length-dependent_2005": "Kistemaker et al. (2005)",
    "hatze_myocybernetic_1981": "Hatze (1981)",
    "van_soest_contribution_1993": "van Soest & Bobbert (1993)"
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
with open("apptbl-parms.tex", "w", encoding="utf-8") as f:
    f.write(latex_str)
    
#%% Great table
def replace_citation(x):
    if x.startswith('@'):
        key = x[1:]
        display = citation_map.get(key, key)

        # extract the year = last underscore-separated part
        parts = key.split('_')
        year = parts[-1] if parts[-1].isdigit() else "year"

        return (
            f'<span class="citation" data-cites="{key}">'
            f'{display} (<a href="#ref-{key}" role="doc-biblioref" aria-expanded="false">{year}</a>)'
            f'</span>'
        )
    return x

df_gt = df.copy()
df_gt['Reference'] = df_gt['Reference'].apply(replace_citation)
df_gt = df_gt.reset_index()

gt_table = (GT(df_gt)
    .tab_stub(rowname_col="index", groupname_col="Partype")
    .cols_align(align='left', columns=["Description","Parameter"])
    .fmt_number(columns=["Value"], n_sigfig=3)
    .fmt_number(columns=["Value"], n_sigfig=2, rows=[2,10,14]) #arel, w, bact3
    .fmt_number(columns=["Value"], n_sigfig=4, rows=[4]) # fmax
    .fmt_scientific(columns=["Value"], n_sigfig=3, rows=[5,6,15]) #kpee,ksee,kca
    .tab_style(style = style.text(style = "italic"), locations = loc.row_groups())
)
gt_table