##############################################################################
#
# BIOE3001: Quantitative Methods in Biomedical Engineering
# Tevyn Vergara | 46421201
#
# Code #2: Model Parameterization for Mark Data and Sensitivity Analysis
#
##############################################################################
# Disclaimer: Claude Opus 5.5 was used in this assignment for:
#
# 1. Layout tips
# 2. Syntax checking
# 3. Reorganisation of structure, parameters and functions
# 4. Troubleshooting and debugging
#
############################# Import Libraries ################################
import numpy as np
import matplotlib.pyplot as plt

############################# Table 1 Input Data ##############################
# [1] {Puttonen, 2008}. IV infusion, levosimendan and metabolite plasma
# concentrations vs time

## Dosing regimen: 0.1 ug/kg/min (100 ng/kg/min) constant IV infusion for 24
## hrs, no loading bolus. From [1].

T1_dose = 100                       # ng/kg/min, dosing infusion rate
T1_T_inf = 24                       # hrs, infusion duration
T1_load = 0                         # ug/kg, no loading bolus. From [1]

## Patient specific parameters
T1_BW = 80                          # kg, sample body weight
# (space for Table 1 cohort demographics)

## TABLE 1 DATA: Clinical data of parent drug and metabolites, from Table 1 data
T1_time = [2, 4, 8, 12, 24, 25, 26, 28, 30, 36, 37, 38, 40, 44, 48, 60, 72, 96, 144, 192, 240]
T1_Levo = [32.5, 28.5, 34, 40, 59, 22.4, 10.4, 4.1, 1.12, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
T1_m1   = [0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.8, 0.9, 1.05, 1.2, 1.55, 2.1, 1.9, 1.95, 1.5, 0.7, 0.35, 0.15]
T1_m2   = [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.6, 0.7, 0.85, 1.1, 1.75, 2.35, 2.4, 2.35, 2.05, 1.2, 0.55, 0.3]

############################# Table 3 Input Data ##############################
# [3] {Álvarez, 2006}. IV loading dose + 24 h infusion, HR, CI and MAP vs time

## Dosing regimen: 12 ug/kg loading dose at t = 0, followed by 200 ng/kg/min
## constant IV infusion for 24 hrs. From [3].

T3_load  = 12                       # ug/kg, loading dose at t = 0 (instant bolus)
T3_dose  = 200                      # ng/kg/min, maintenance infusion rate
T3_T_inf = 24                       # hrs, maintenance infusion duration

## Patient specific parameters
T3_HR0  = 83                        # bpm, baseline HR, t = 0 row of [3]
T3_MAP0 = 84                        # mmHg, baseline MAP, t = 0 row of [3]
T3_CI0  = 2.08                      # L/min/m^2, baseline cardiac index, t = 0 row of [3]
# T3_BW  =                          # kg, cohort body weight (TBA from [3] demographics)
# T3_BSA =                          # m^2, body surface area, for CI -> CO (TBA)

## Clinical data
## TABLE 3 Heart Rate, Cardiac Output (Derived) and MAP 
T3_time = [0, 6, 12, 18, 24, 36, 48]              # hrs, time column
T3_CI   = [2.08, 2.25, 2.83, 2.86, 2.9, 2.87, 2.84]   # L/min/m^2, cardiac index column
T3_HR   = [83, 95, 95.5, 96, 96, 94, 92]             # bpm, heart rate column
T3_MAP  = [84, 72, 75.5, 75, 76, 77, 78]             # mmHg, MAP column


############################### Time Course ###################################
## Time course: set in days (runs in hrs for better computation accuracy)
## Single dose only: Tables 1 and 3 each report one dosing course

T_days = 12                         # days, time course (covers the 240 hr Table 1 data)

dt = 0.1                            # hrs, time step
T = T_days * 24                     # hrs, time course

############################# Parameters: Patient #############################
# Patient-Specific Body parameters (Reference for dose, used for later Code 2 and 3)
BW = 80                             # kg, sample body weight from [1],[10]

# Conversion macros for patient PD
MMHG_TO_PA = 133.322                # Pa per mmHg
ML_TO_M3 = 1e-6                     # m^3 per mL

# Patient-specific PD Heart Parameters (for curing)
HR0 = 70.0 / 60.0                   # Hz, baseline heart rate. From [10]
SV0 = 77.0 * ML_TO_M3               # m^3, baseline SV. From [11]
MAP0 = 105.0 * MMHG_TO_PA           # Pa, Baseline MAP. From [11]

# Patient-specific PD Brain Parameters (for curing)
v_cereb0 = 0.40                     # m/s, baseline MCA velocity. From [11]
P_brain = 10.0 * MMHG_TO_PA         # Pa, intracranial pressure. From [12]
r_cereb = 0.00128                   # m, MCA M1 radius. From [15]

############################# Parameters: Dosing ##############################
# Each regimen = [loading dose (ng/kg), infusion rate (ng/kg/min), infusion duration (hrs)]
# Built from the Table 1 and Table 3 input data, so simulate() can run either one
# Usual range: 0.05 - 0.2ug/kg/min. From [16]

T1_regimen = [T1_load * 1000, T1_dose, T1_T_inf]  # Table 1 [1], ug/kg x 1000 -> ng/kg
T3_regimen = [T3_load * 1000, T3_dose, T3_T_inf]  # Table 3 [3], ug/kg x 1000 -> ng/kg

################### Parameters: Drug (Pharmacokinetics) #######################

# Parameters: Levosimendan (Active)
#
# Initial drug dosing based off Table 1 dosing regimen since Safer Care Victoria [16] 
# has similar dosing regimen to source
#
# Levosimendan enters at constant infusion rate but constant loss clears most
# of it on arrival. C_int is the t = 0 intercept of the line fitted to the 
# 2–24 hr data (no sample was taken at t = 0)

levo_half = 0.9                     # hrs, Levosimendan half-life. From [1]
k0 = (np.log(2) / levo_half) * dt   # 1/Hrs, parent half-life decay (DERIVED)
f0 = 0.03                           # Free Fraction Levosimendan. From [4]

C_slope = 1.3388                    # ng/mL/hr, net rise during infusion (FITTED FROM [1])
C0_levo = 25.00                     # ng/mL, intercept at t=0 (CALCULATED FROM [1])
V_kg = 132                          # mL/kg, parent volume of distribution (BEHAVIOUR FROM [1])

# Dose conversions (dose-independent). R_inf, L_inf, C0_inf and C_dose_load
# depend on the regimen, so they are calculated inside simulate()
dose_ref = T1_dose                  # ng/kg/min, Table 1 dose that C_slope and C0_levo were measured at
phi = C_slope / (dose_ref * 60 / V_kg) # fraction of infused drug retained in plasma, fixed drug property (DERIVED)
C0_per_dose = C0_levo / dose_ref    # ng/mL per (ng/kg/min), intercept per unit dose (DERIVED)

# Parameters: Gut Transit Compartment (Time Delay between parent and metabolites)
T_gut = 30                          # hrs, time when gut starts to release (FITTED FROM [1])
gut_rate = 53                       # hrs, gut release rate From [13]

# Parameters: OR-1855 Metabolite (Inactive)
m1 = 0.05                           # Formation Fraction into OR-1855 metabolite. From [13]
Vm_ratio = 0.5                      # parent-to-metabolite volume ratio (FITTED FROM [1])

# Parameters: OR-1896 Metabolite (Active)
m2_half = 7.7                       # Hrs, OR-1896 half-life (FITTED)
f2 = 0.61                           # Free Fraction OR-1896. From [1], [4]

# Parameters: Acetylation (OR-1855 -> OR-1896) through NAT2 enzyme
ac_rate = 9.6                       # Hrs, acetylation rate. From [13]

################### Parameters: Body (Pharmacodynamics) #######################
# PD is kept in SI for easy conversion
# Patient baselines are defined at top

# Hill Kinetics: drug effect on stroke volume
dSVmax = 0.28                       # %, maximum SV increase from Levosimendan. From [14]
EC50_SV = 0.497                     # ng/mL (Fitted)

# Hill Kinetics: drug effect on heart rate
Emax_HR = 0.11                      # %, maximum HR increase from Levosimendan. From [2][3]
EC50_HR = 0.57                      # ng/mL, Free Ce hill kinetic term. Derived from [2]

# Hill Kinetics: drug dilation of the systemic (body) vasculature
Emax_R = 0.4                        # fractional max R_sys drop. (Fitted)
EC50_sys = 0.498                    # ng/mL (Fitted)

# Derivation for baseline systemic resistance
# (how strong vessels oppose blood flow)
# Resistance = (pressure / flow), both baseline
CO0 = SV0 * HR0                     # m^3/s, baseline cardiac output (DERIVED)
R_sys0 = MAP0 / CO0                 # Pa.s/m^3, baseline systemic resistance (DERIVED)

### Parameters: Brain Block (MAP, Ce_eff --> V_cereb)

A_cereb = np.pi * (r_cereb) ** 2    # m^2, MCA cross-section

# Derivation for baseline cerebral resistance, similar to R_sys
# Resistance = pressure / flow, both baseline

CPP0 = MAP0 - P_brain               # Pa, baseline cerebral perfusion pressure (DERIVED)
Q_cereb0 = v_cereb0 * A_cereb       # m^3/s, baseline cerebral flow (DERIVED)
R_cereb0 = CPP0 / Q_cereb0          # Pa.s/m^3, baseline cerebral resistance (DERIVED)

# Hill kinetics for drug binding to cerebral vasculature resistance
Emax_cereb = 0.405                  # fractional max R_cereb drop (Fitted)
EC50_cereb = 0.5                    # ng/mL (Fitted)

# Delay between plasma Ce and drug effect at tissue
keo_thalf = 10                      # hrs, compartment-effect time delay (Fitted)

############################ Parameters to Fit ################################
# Initial values (starting guesses for the optimiser) are the Code 1 values above.
# Passed to the fit as one parameter list. From [17]. Order is fixed:
# PK set by Table 1 first, then PD set by Table 3

P_names = ["gut_rate", "Vm_ratio", "ac_rate", "m2_half",   # PK, Table 1
           "f0", "keo_thalf",                              # PD, Table 3
           "Emax_R", "EC50_sys", "dSVmax", "EC50_SV", "Emax_HR", "EC50_HR"]

params0 = [gut_rate, Vm_ratio, ac_rate, m2_half,
           f0, keo_thalf,
           Emax_R, EC50_sys, dSVmax, EC50_SV, Emax_HR, EC50_HR]

### Heart (Mass Transfer System) ###
#
# Takes effect-site drug Ce_eff and runs three Hill Kinetic responses:
# Systemic resistance, stroke volume and heart rate.
# P_HB = [Emax_R, EC50_sys, dSVmax, EC50_SV, Emax_HR, EC50_HR] (fitted values)
# Outputs MAP --> Brain
#
def heart_block(Ce_eff_v, P_HB):
    Emax_R, EC50_sys, dSVmax, EC50_SV, Emax_HR, EC50_HR = P_HB
    R_sys_v = R_sys0 * (1 - Emax_R * Ce_eff_v / (EC50_sys + Ce_eff_v))
    SV_v = SV0 * (1 + dSVmax * Ce_eff_v / (EC50_SV + Ce_eff_v))
    HR_v = HR0 * (1 + Emax_HR * Ce_eff_v / (EC50_HR + Ce_eff_v))
    CO_v = SV_v * HR_v
    MAP_v = CO_v * R_sys_v
    return R_sys_v, SV_v, HR_v, CO_v, MAP_v

### Cerebral Vasculature (Mass Transfer System) ###
#
# Takes lagged effective drug and MAP (both from the heart block)
# Combined lumped resistance and MAP to solve for blood velocity
# Equations from [5].
def cerebral_block(Ce_val, MAP_v):
    Cerebral_R = R_cereb0 * (1 - Emax_cereb * Ce_val / (EC50_cereb + Ce_val))
    Blood_V = (MAP_v - P_brain) / (Cerebral_R * A_cereb)
    return Cerebral_R, Blood_V

############################# Model Function ##################################
# Runs the Code 1 model for one parameter set and one dosing regimen, so the
# parameterisation can call it many times. Model loop inside a function of
# the parameters, from [17]
#
# params  = fitted parameters, same order as P_names
# regimen = [loading dose (ng/kg), infusion rate (ng/kg/min), infusion duration (hrs)]
#
def simulate(params, regimen):

    ### Unpack parameters and regimen ###
    gut_rate, Vm_ratio, ac_rate, m2_half, f0, keo_thalf = params[0:6]
    P_HB = params[6:12]                 # heart block PD parameters
    dose_load, dose_sup, dose_inf = regimen[0], regimen[1], regimen[2]

    ### Rate constants (pre-multiplied by dt) ###
    kg = (np.log(2) / gut_rate) * dt    # 1/hrs, gut release rate as half life
    kac = (np.log(2) / ac_rate) * dt    # 1/Hrs, acetylation as half life 
    k2 = (np.log(2) / m2_half) * dt     # 1/Hrs, OR-1896 half-life decay 
    k_eo = (np.log(2) / keo_thalf) * dt # 1/hrs, effect-compartment delay 

    ### Dose conversions for main loop ###
    R_inf = (dose_sup * 60 / V_kg) * dt # ng/mL per step, infusion input
    C0_inf = C0_per_dose * dose_sup     # ng/mL, initial jump for this dose 
    L_inf = (1 - phi) * R_inf           # ng/mL per step, loss during infusion
    C_dose_load = dose_load / V_kg      # ng/mL, loading dose jump at t = 0

    ### Initial Conditions ###
    CLevo      = [None] * int(T // dt)      # ng/mL, Levosimendan (parent)
    Cdose_load = [None] * int(T // dt)      # ng/mL, loading dose part of the parent
    CGut       = [None] * int(T // dt)      # ng/mL, gut transit compartment
    C1855      = [None] * int(T // dt)      # ng/mL, OR-1855
    C1896      = [None] * int(T // dt)      # ng/mL, OR-1896
    Ce         = [None] * int(T // dt)      # ng/mL, free active drug in plasma
    Ce_eff     = [None] * int(T // dt)      # ng/mL, effect-site concentration (lagged Ce)

    R_sys      = [None] * int(T // dt)      # Pa.s/m^3, systemic resistance
    SV         = [None] * int(T // dt)      # m^3, stroke volume
    HR         = [None] * int(T // dt)      # Hz, heart rate
    CO         = [None] * int(T // dt)      # m^3/s, cardiac output
    MAP        = [None] * int(T // dt)      # Pa, mean arterial pressure
    R_cereb    = [None] * int(T // dt)      # Pa.s/m^3, cerebral resistance
    V_cereb    = [None] * int(T // dt)      # m/s, cerebral blood velocity

    CLevo[0] = C0_inf + C_dose_load        # ng/mL, initial + loading dose
    Cdose_load[0] = C_dose_load             # ng/mL, just loading dose
    CGut[0]  = 0                            # ng/mL
    C1855[0] = 0                            # ng/mL
    C1896[0] = 0                            # ng/mL
    Ce[0] = f0 * CLevo[0] + f2 * C1896[0]   # ng/mL, quick implementation of effective drug
    Ce_eff[0] = 0                           # ng/mL

    # Implementing baseline patient parameters as initial conditions
    R_sys[0], SV[0], HR[0], CO[0], MAP[0] = heart_block(Ce_eff[0], P_HB)
    R_cereb[0], V_cereb[0] = cerebral_block(Ce_eff[0], MAP[0])

    ### Main Loop ###

    for t in range(0, int(T // dt - 1)):

        ### Levosimendan (ACTIVE) ###

        # dose_loading dose part: given at t = 0, cleared by first-order decay from the
        # start (the infusion loss L_inf only acts on the infused drug)
        dCdose_load = -k0 * Cdose_load[t]
        Cdose_load[t + 1] = Cdose_load[t] + dCdose_load
        
        # If within "dose_inf" hrs, drug is infusing
        if t * dt < dose_inf:
            dCLevo = R_inf - L_inf + dCdose_load
            Form = m1 * Vm_ratio * (L_inf + k0 * Cdose_load[t])
            
        # Else, drug is naturally expelled with decay
        else:
            dCLevo = -k0 * CLevo[t]
            Form = m1 * Vm_ratio * k0 * CLevo[t]
        CLevo[t + 1] = CLevo[t] + dCLevo

        ### Gut Transit Compartment (time capture) ###
        if t * dt < T_gut:
            Rel = 0
        else:
            Rel = kg * CGut[t]

        dCGut = Form - Rel
        CGut[t + 1] = CGut[t] + dCGut

        ### OR-1855 (INACTIVE Metabolite) ###
        # Formed by gut metabolism, lost from acetylation (1896) 
        dC1855 = Rel - kac * C1855[t]
        C1855[t + 1] = C1855[t] + dC1855

        ### OR-1896 (ACTIVE Metabolite) ###
        # Formed by acetylation and self decay
        dC1896 = kac * C1855[t] - k2 * C1896[t]
        C1896[t + 1] = C1896[t] + dC1896

        ### Active Drug Concentration (PK Output)
        Ce[t + 1] = f0 * CLevo[t + 1] + f2 * C1896[t + 1]

        ### Effect Compartment (delay between plasma and tissue) ###
        Ce_eff[t + 1] = Ce_eff[t] + k_eo * (Ce[t] - Ce_eff[t])

        ### Heart PD (Mass Transfer System) ###
        R_sys[t + 1], SV[t + 1], HR[t + 1], CO[t + 1], MAP[t + 1] = heart_block(Ce_eff[t + 1], P_HB)

        ### Cerebral Vasculature PD (Mass Transfer System) ###
        R_cereb[t + 1], V_cereb[t + 1] = cerebral_block(Ce_eff[t + 1], MAP[t + 1])

    return (CLevo, CGut, C1855, C1896, Ce, Ce_eff,
            R_sys, SV, HR, CO, MAP, R_cereb, V_cereb)

############################ Initial Model Runs ###############################
# Code 1 initial values, run with each table's dosing regimen

## Table 1 regimen [1]
(CLevo, CGut, C1855, C1896, Ce, Ce_eff,
 R_sys, SV, HR, CO, MAP, R_cereb, V_cereb) = simulate(params0, T1_regimen)

## Table 3 regimen [3]
(CLevo_T3, CGut_T3, C1855_T3, C1896_T3, Ce_T3, Ce_eff_T3,
 R_sys_T3, SV_T3, HR_T3, CO_T3, MAP_T3, R_cereb_T3, V_cereb_T3) = simulate(params0, T3_regimen)

############################# Helper Functions ################################
time_hrs = np.arange(0, int(T // dt)) * dt      # hrs, x-axis for fit plots (data is in hrs)

# % change from baseline (Table 3: each cohort's own baseline drops out)
def pct(y, y0):
    return (np.array(y, dtype=float) - y0) / y0 * 100

# % change of a model curve from its drug-naive first value
def pct_curve(y):
    return pct(y, y[0])

# Model value at each data time (index = t / dt). From [17]
def sample(curve, t_data):
    idx = [int(round(t / dt)) for t in t_data]
    return np.array(curve, dtype=float)[idx]

# R^2 and RMSE from [17]. NRMSE = RMSE / data range, to compare outputs with different units
def fit_metrics(y_data, y_model):
    y_data = np.array(y_data, dtype=float)
    residual = y_data - y_model
    R2 = 1 - np.sum(residual ** 2) / np.sum((y_data - y_data.mean()) ** 2)
    RMSE = np.sqrt(np.mean(residual ** 2))
    NRMSE = RMSE / (y_data.max() - y_data.min())
    return R2, RMSE, NRMSE

# Initial model, fitted model and data on one figure. From [17]
def plot_fit(y_init, y_fit, t_data, y_data, ylabel, title):
    plt.figure()
    plt.plot(time_hrs, y_init, "r", label="Initial model")
    if y_fit is not None:
        plt.plot(time_hrs, y_fit, "b--", label="Fitted model")
    plt.plot(t_data, y_data, "ko", label="Data")
    plt.ylabel(ylabel)
    plt.xlabel("Time (hrs)")
    plt.title(title)
    plt.legend()
    plt.show()

############################## Plot Figure ####################################

time = (np.arange(0, int(T // dt)) * dt) / 24 # days, x-axis in real time not step number

## Levosimendan Concentration 
plt.figure()
plt.plot(time, CLevo)
plt.ylabel("Levosimendan Concentration in Blood (ng/mL)")
plt.xlabel("Time (days)")
plt.title("Concentration of Levosimendan (Parent)")
plt.show()

## Gut Transition of OR1855
plt.figure()
plt.plot(time, CGut)
plt.ylabel("Gut Concentration (ng/mL)")
plt.xlabel("Time (days)")
plt.title("Gut Transit of OR-1855 metabolite")
plt.show()

## OR-1855 Concentration 
plt.figure()
plt.plot(time, C1855)
plt.ylabel("OR-1855 Concentration in Blood (ng/mL)")
plt.xlabel("Time (days)")
plt.title("OR-1855 (Inactive Metabolite)")
plt.show()

## OR-1896 Concentration 
plt.figure()
plt.plot(time, C1896)
plt.ylabel("OR-1896 in Blood (ng/mL)")
plt.xlabel("Time (days)")
plt.title("OR-1896 (Active Metabolite)")
plt.show()

## Pharmacodynamic outputs with SI conversions
MAP_mmHg = np.array(MAP) / MMHG_TO_PA
HR_bpm = np.array(HR) * 60
V_cereb_cms = np.array(V_cereb) * 100

## MAP 
plt.figure()
plt.plot(time, MAP_mmHg)
plt.ylabel("MAP (mmHg)")
plt.xlabel("Time (days)")
plt.title("Mean Arterial Pressure (MAP)")
plt.show()

## Heart Rate
plt.figure()
plt.plot(time, HR_bpm)
plt.ylabel("HR (bpm)")
plt.xlabel("Time (days)")
plt.title("Heart Rate (HR)")
plt.show()

## Blood Velocity through the Middle Cerebral Artery (MCA M1)
plt.figure()
plt.plot(time, V_cereb_cms)
plt.ylabel("Blood Velocity through MCA (cm/s)")
plt.xlabel("Time (days)")
plt.title("Cerebral Blood Flow Velocity")
plt.show()

############################## References #####################################
# Declaration of AI use
# [0] Anthropic, "Claude, Opus 5.5," Claude, 22 October 2026 [Online]. https://claude.ai/new
#
# Full references details are in reference manager
#
# Table 1.
# [1] {Puttonen, 2008}
#
# Table 2.
# [2] {Kivikko, 2015}
#
# Table 3.
# [3] {Álvarez, 2006}
#
# Others:
# [4] {Antila, 2004}
# [5] {Robinson, 2026}
# [6] {Kashou, 2022}
# [7] {Khalil, 2019}
# [8] {DeMers, 2023}
# [9] {Munakomi, 2026}
# [10] {Böhm, 2012}
# [11] {Miller, 2021}
# [12] {Hood, 2023}
# [13] {Bertin, 2026}
# [14] {Slawsky, 2000}
# [15] {Mirza, 2024}
# [16] {SafeCareVictoria, 2018}
# [17] {Mann, 2016}
