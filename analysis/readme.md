---
title: "Analysis"
---

On these pages, the complete analysis accompanying the manuscript can be found. The analysis is organised into several parts. First, model predictions are derived. Second, the AMPO measurement data are preprocessed. Third, analyses are performed, with almost every part corresponding to a subsection of the Results section in the manuscript.

All custom modules developed for the manuscript can be found [here](modules.qmd). 

## 1. Model predictions

The page [model predictions](model-predictions.py) contains scripts to derive all model predictions. This includes evaluating the influence of knee joint movement on the maximally attainable AMPO for various SSCs: both those using the dynamometer trajectory (see Figure 3 in the manuscript) and those with different constant angular velocities for flexion and extension (in order to simulate a broad range of knee joint movements).

## 2. Preprocess data

The page [preprocess data](do-preprocess.py) contains the preprocessing steps described in the Methods. This includes time-synchronising the data from the dynamometer and EMG measurements, as well as computing the muscular contribution to the measured net knee joint moment. 

## 3.1 Isometric measurements

*Section: 'AMPO-scaling factors were successfully obtained'*

The page [isometric measurements](isometric-measurements.py) contains the analysis of the isometric measurements. Using these measurements, model parameters were estimated individually for each participant, allowing scaling of measured AMPO for each participant.

## 3.2 Analyse imposed knee joint movements

*Section: 'Knee joint movements were succesfuly imposed'*

The page [Knee joint movements](imposed-motion.check.py) contains the analysis of the imposed knee joint movements. The measured knee joint movements were compared against the desired trajectories to verify whether the intended knee joint movements were accurately imposed.

## 3.3 EMG and mechanical work

*Section: 'Timing of m. quadriceps femoris activation was adequate' & 'Mechanical work was achieved almost exclusively during knee extension'*

The page [EMG and mechanical work](emg-and-mech-work.py) contains the analyses of measured EMG signals of *m. quadriceps femoris* and the hamstring muscle group during knee extension and flexion, as well as the computation of the mechanical work during knee extension and flexion.

## 3.4. Measurements against predictions

*Section: 'Measured AMPO closely matched predicted maximally attainable AMPO'*

The page [measurements against predictions](measured-vs-predicted-ampo.py) contains the comparison of measured AMPO values with model predictions. It also assesses the reliability of measured AMPO through repeated conditions and variations within individual trials.

## 3.5. Influence SSC parameters

*Section: 'Influence of SSC parameters on the maximally attainable AMPO'*
 
The page [influence SSC parameterers](influence-ssc-parameters.py) contains the analysis of how SSC parameters influence the maximally attainable AMPO. The effects of cycle frequency, knee joint excursion, and other SSC factors are evaluated.