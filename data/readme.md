---
title: "Data"
---

All data of this study can be found on the [Github repository](https://github.com/edwinreuvers/hq-ampo). The `data/` folder is divided into three main parts:

* **`dataExp/`** → Experimentally measured data
* **`simsExp/`** → Hill-type MTC model predictions with knee joint movements that could be imposed by the knee dynamometer 
* **`simsSSC/`** – Hill-type MTC model predictions with knee joint movements with different constant knee angular velocities during flexion and extension

## `dataExp/` – Experimental data

This folder contains all experimentally measured data from participants. 

### Level 2 - Participant folder

* **`pp01/`, `pp02/`, ...**
  Each folder corresponds to an individual participant.

#### Level 3- Measurement day

* **`day1/`** – Data of day 1 (training session)
* **`day2/`** – Data of day 2 (training session)
* **`day3/`** – Data of day 3 (isometric measurements + training session)
* **`day4/`** – Data of day 4 (AMPO measurements)
* **`AMPO measurements/`** – Preprocessed data AMPO measurements (see [here](analysis/do-preprocess.py]))

Files follow a structured naming scheme:

```
ppXX_dayY_<type>_tZ
```

Where:

* `ppXX` → participant ID
* `dayY` → measurement day
* `<type>` → condition or measurement type
* `tZ` → trial number

Example:

* `pp05_day2_cond02_t1` → participant 5, day 2, condition 2, trial 1

Note that, conditions `01–09` correspond to the following condition names described in the manuscript:

```
['0.40','0.35C','0.30','0.25','0.20','0.35A','0.35B','0.35D','0.35E']
```

##### Day 1 & 2 – Training

On these day participants trained for the AMPO measurements. For some participants (not all) data is saved.

##### Day 3 – Isometric measurements + Training

Includes both training trials and additional measurement types.

**Training trials:**

```
ppXX_day3_condZZ_tZ
```

**Moment–angle relation:**

```
ppXX_day3_mhXX_tZ
```

* `mhXX` → moment-angle measurement at a specific joint angle (e.g. `mh16` = 1.6 rad)

**Gravity measurements:**

```
ppXX_day3_gravity_tZ
```

* Passive moment measured across knee angles (used for gravity correction; see paper)

##### Day 4 – AMPO measurements

**Condition trials:**

```
ppXX_day4_condZZ_tZ
```

* `.mat` file → dynamometer data
* `.csv` file → EMG data

Example:

* `pp05_day4_cond06_t1`

**Maximum voluntary contractions (MVC):**

```
ppXX_day4_mvc_<muscle>_tZ
```

* `<muscle>` → e.g. `hams` (hamstrings)

Example:

* `pp05_day4_mvc_hams_t2`


**Gravity measurements:**

```
ppXX_day4_gravity_tZ
```


## `simsExp/` – Simulations using dynamometer trajectories

This folder contains predictions from a Hill-type MTC model for a wide range of knee joint movements that can be imposed by the dynamometer.

Knee joint movements are defined by:

* Knee joint excursion
* Cycle frequency
* Fraction of the cycle time spent shortening (FTS)

Due to limits on acceleration, not all combinations are feasible.
Simulations are only provided for feasible combinations.

In this folder, two type of files are present. The first is:

`cond_<condition_name>.csv` → prediction for condition with name `condition_name`

The second is for a large number of combinations of knee joint excursion, cycle frequency and FTS:

```plaintext
kje<excursion>_cf<cf>_fts<fts>.csv
```

Where:

* `kje` → knee joint excursion
* `cf` → cycle frequency
* `fts` → fraction of the cycle time spent shortening

### Level 3 - Sensitivity analysis

* **`sensitivity/`**

Contains predictions for the experimental conditions, but for a decrease in 10% of $L_{CE}^{opt}$.

## `simsSSC/` – Simulations using different constant velocities

This folder contains predictions from a Hill-type MTC model for a wide range of knee joint movements with different constant knee angular velocities during flexion and extension.

Knee joint movements are defined by:

* Knee joint excursion
* Cycle frequency
* Fraction of the cycle time spent shortening (FTS)

Due to limits on acceleration, not all combinations are feasible.
Simulations are only provided for feasible combinations.

Files are named as follows:

```plaintext
kje<excursion>_cf<cf>_fts<fts>.csv
```

Where:

* `kje` → knee joint excursion
* `cf` → cycle frequency
* `fts` → fraction of the cycle time spent shortening

## Folder tree
```plaintext
data/
├─ dataExp/                      # Experimentally measured data
│   ├─ pp01/                     # Participant 1
│   │   ├─ day1/                 # Training session
│   │   │   ├─ pp01_day1_cond01_t1.csv
│   │   │   ├─ pp01_day1_cond01_t2.csv
│   │   │   └─ ...
│   │   ├─ day2/                 # Training session
│   │   │   └─ pp01_day2_condZZ_tZ.csv
│   │   ├─ day3/                 # Isometric measurements + training
│   │   │   ├─ pp01_day3_condZZ_tZ.csv
│   │   │   ├─ pp01_day3_mhXX_tZ.csv
│   │   │   └─ pp01_day3_gravity_tZ.csv
│   │   ├─ day4/                 # AMPO measurements
│   │   │   ├─ pp01_day4_condZZ_tZ.mat
│   │   │   ├─ pp01_day4_condZZ_tZ.csv
│   │   │   ├─ pp01_day4_mvc_hams_tZ.csv
│   │   │   └─ pp01_day4_gravity_tZ.csv
│   │   └─ AMPO_preprocessed/    # Preprocessed AMPO data
│   │       └─ pp01_day4_condZZ_tZ_preprocessed.csv
│   ├─ pp02/
│   │   └─ ...
│   └─ ppNN/                      # Last participant
│
├─ simsExp/                       # Hill-type MTC model predictions for dynamometer trajectories
│   ├─ cond_0.40.csv
│   ├─ cond_0.35C.csv
│   ├─ ...
│   ├─ kje<excursion>_cf<cf>_fts<fts>.csv
│   └─ sensitivity/               # Sensitivity analysis (lce_opt*0.9)
│       ├─ cond_0.40.csv
│       ├─ cond_0.35C.csv
│       └─ ...
│
└─ simsSSC/                       # Hill-type MTC model predictions for different constant angular velocities
    ├─ kje<excursion>_cf<cf>_fts<fts>.csv
    └─ ...
```