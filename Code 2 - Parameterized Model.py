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
from scipy.optimize import least_squares

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

levo_half = 0.9                          # hrs, Levosimendan half-life. From [1]
k0 = (np.log(2) / levo_half) * dt        # 1/Hrs, parent half-life decay (DERIVED)
f0 = 0.03                                # Free Fraction Levosimendan. From [4]

C_slope = 1.3388                         # ng/mL/hr, net rise during infusion (FITTED FROM [1])
C0_levo = 25.00                          # ng/mL, intercept at t=0 (CALCULATED FROM [1])
V_kg = 132                               # mL/kg, parent volume of distribution (BEHAVIOUR FROM [1])

# Dose conversions (dose-independent). R_inf, L_inf, C0_inf and C_dose_load
# depend on the regimen, so they are calculated inside simulate()
dose_ref = T1_dose                       # ng/kg/min, Table 1 dose that C_slope and C0_levo were measured at
phi = C_slope / (dose_ref * 60 / V_kg)   # fraction of infused drug retained in plasma, fixed drug property (DERIVED)
C0_per_dose = C0_levo / dose_ref         # ng/mL per (ng/kg/min), intercept per unit dose (DERIVED)

# Parameters: Gut Transit Compartment (Time Delay between parent and metabolites)
T_gut = 30                               # hrs, time when gut starts to release (FITTED FROM [1])
gut_rate = 53                            # hrs, gut release rate From [13]

# Parameters: OR-1855 Metabolite (Inactive)
m1 = 0.05                                # Formation Fraction into OR-1855 metabolite. From [13]
Vm_ratio = 0.5                           # parent-to-metabolite volume ratio (FITTED FROM [1])

# Parameters: OR-1896 Metabolite (Active)
m2_half = 7.7                            # Hrs, OR-1896 half-life (FITTED)
f2 = 0.61                                # Free Fraction OR-1896. From [1], [4]

# Parameters: Acetylation (OR-1855 -> OR-1896) through NAT2 enzyme
ac_rate = 9.6                            # Hrs, acetylation rate. From [13]

################### Parameters: Body (Pharmacodynamics) #######################
# PD is kept in SI for easy conversion
# Patient baselines are defined at top

# Hill Kinetics: drug effect on stroke volume
dSVmax = 0.28                            # %, maximum SV increase from Levosimendan. From [14]
EC50_SV = 0.497                          # ng/mL (Fitted)

# Hill Kinetics: drug effect on heart rate
Emax_HR = 0.11                           # %, maximum HR increase from Levosimendan. From [2][3]
EC50_HR = 0.57                           # ng/mL, Free Ce hill kinetic term. Derived from [2]

# Hill Kinetics: drug dilation of the systemic (body) vasculature
Emax_R = 0.4                             # fractional max R_sys drop. (Fitted)
EC50_sys = 0.498                         # ng/mL (Fitted)

# Derivation for baseline systemic resistance
# (how strong vessels oppose blood flow)
# Resistance = (pressure / flow), both baseline
CO0 = SV0 * HR0                          # m^3/s, baseline cardiac output (DERIVED)
R_sys0 = MAP0 / CO0                      # Pa.s/m^3, baseline systemic resistance (DERIVED)

### Parameters: Brain Block (MAP, Ce_eff --> V_cereb)

A_cereb = np.pi * (r_cereb) ** 2         # m^2, MCA cross-section

# Derivation for baseline cerebral resistance, similar to R_sys
# Resistance = pressure / flow, both baseline

CPP0 = MAP0 - P_brain                    # Pa, baseline cerebral perfusion pressure (DERIVED)
Q_cereb0 = v_cereb0 * A_cereb            # m^3/s, baseline cerebral flow (DERIVED)
R_cereb0 = CPP0 / Q_cereb0               # Pa.s/m^3, baseline cerebral resistance (DERIVED)

# Hill kinetics for drug binding to cerebral vasculature resistance
Emax_cereb = 0.405                       # fractional max R_cereb drop (Fitted)
EC50_cereb = 0.5                         # ng/mL (Fitted)

# Delay between plasma Ce and drug effect at tissue
keo_thalf = 10                           # hrs, compartment-effect time delay (Fitted)

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
    kg = (np.log(2) / gut_rate) * dt        # 1/hrs, gut release rate as half life
    kac = (np.log(2) / ac_rate) * dt        # 1/Hrs, acetylation as half life 
    k2 = (np.log(2) / m2_half) * dt         # 1/Hrs, OR-1896 half-life decay 
    k_eo = (np.log(2) / keo_thalf) * dt     # 1/hrs, effect-compartment delay 

    ### Dose conversions for main loop ###
    R_inf = (dose_sup * 60 / V_kg) * dt     # ng/mL per step, infusion input
    C0_inf = C0_per_dose * dose_sup         # ng/mL, initial jump for this dose 
    L_inf = (1 - phi) * R_inf               # ng/mL per step, loss during infusion
    C_dose_load = dose_load / V_kg          # ng/mL, loading dose jump at t = 0

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

    CLevo[0] = C0_inf + C_dose_load         # ng/mL, initial + loading dose
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

# R^2 and RMSE. From [17]
def fit_metrics(y_data, y_model):
    y_data = np.array(y_data, dtype=float)
    residual = y_data - y_model
    R2 = 1 - np.sum(residual ** 2) / np.sum((y_data - y_data.mean()) ** 2)
    RMSE = np.sqrt(np.mean(residual ** 2))
    return R2, RMSE

# Initial model, fitted model and data on one subplot. From [17]
def plot_fit(ax, y_init, y_fit, t_data, y_data, ylabel, title):
    ax.plot(time_hrs, y_init, "r", label="Initial model")
    ax.plot(time_hrs, y_fit, "b--", label="Fitted model")
    ax.plot(t_data, y_data, "ko", label="Data")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (hrs)")
    ax.set_title(title)
    ax.legend()

########################## Model Parameterisation #############################
# One parameter set fits both studies. Each part of the model is fitted to its
# own dataset, then the fitted parts are combined into one model. From [17]
#
# 1. PK parameters fitted to Table 1 (Table 1 regimen). The PD parameters do
#    not affect Table 1, so they stay at their initial values here
# 2. PD parameters fitted to Table 3 (Table 3 regimen), with the fitted PK fixed
# 3. Fitted PK + PD combined into one parameter set, params_fit
#
# least_squares with bounds keeps fractions within 0-1 and rates positive. From [17]

N_PK = 4                                # first 4 entries of P_names are PK (Table 1)

## Bounds, same order as P_names (physical limits only)
P_lower = [0.5, 0.01, 0.5, 0.5,         # hrs, -, hrs, hrs
           0.001, 0.1,                  # -, hrs
           0, 0.001, 0, 0.001, 0, 0.001]
P_upper = [200, 10, 200, 200,
           1, 72,
           1, 10, 1, 10, 1, 10]

## Table 1: each curve divided by its own data range, so all three carry equal weight
T1_range = [np.ptp(T1_Levo), np.ptp(T1_m1), np.ptp(T1_m2)]  # ng/mL, max - min

## Table 3: % change from baseline. t = 0 left out (0 % by construction)
T3_t = T3_time[1:]
T3_HR_pct = pct(T3_HR, T3_HR0)[1:]
T3_MAP_pct = pct(T3_MAP, T3_MAP0)[1:]
T3_CI_pct = pct(T3_CI, T3_CI0)[1:]      # CI = CO / BSA, so % CI = % CO (BSA cancels)

## Residual functions: data - model at the data times. From [17]
def funcPK(params_PK, params_PD):
    CLevo_m, CGut_m, C1855_m, C1896_m = simulate(list(params_PK) + list(params_PD), T1_regimen)[0:4]
    res_Levo = (np.array(T1_Levo) - sample(CLevo_m, T1_time)) / T1_range[0]
    res_1855 = (np.array(T1_m1) - sample(C1855_m, T1_time)) / T1_range[1]
    res_1896 = (np.array(T1_m2) - sample(C1896_m, T1_time)) / T1_range[2]
    return np.concatenate([res_Levo, res_1855, res_1896])

def funcPD(params_PD, params_PK):
    HR_m, CO_m, MAP_m = simulate(list(params_PK) + list(params_PD), T3_regimen)[8:11]
    res_HR = T3_HR_pct - sample(pct_curve(HR_m), T3_t)
    res_MAP = T3_MAP_pct - sample(pct_curve(MAP_m), T3_t)
    res_CI = T3_CI_pct - sample(pct_curve(CO_m), T3_t)
    return np.concatenate([res_HR, res_MAP, res_CI])

## 1. PK fitted to Table 1
fit_PK = least_squares(funcPK, params0[:N_PK], args=(params0[N_PK:],),
                       bounds=(P_lower[:N_PK], P_upper[:N_PK]))

## 2. PD fitted to Table 3, fitted PK held fixed
fit_PD = least_squares(funcPD, params0[N_PK:], args=(fit_PK.x,),
                       bounds=(P_lower[N_PK:], P_upper[N_PK:]))

## 3. One parameter set
params_fit = list(fit_PK.x) + list(fit_PD.x)

## Standard errors: covariance = (J^T J)^-1 x residual variance SSR / (N - p)
def std_err(fit):
    J = fit.jac
    cov = np.linalg.pinv(J.T @ J) * np.sum(fit.fun ** 2) / (len(fit.fun) - len(fit.x))
    return np.sqrt(np.diag(cov))

P_SE = list(std_err(fit_PK)) + list(std_err(fit_PD))

############################# Fitted Model Runs ###############################
# One fitted parameter set, run with each table's dosing regimen

## Table 1 regimen [1]
(CLevo_fit, CGut_fit, C1855_fit, C1896_fit, Ce_fit, Ce_eff_fit,
 R_sys_fit, SV_fit, HR_fit, CO_fit, MAP_fit, R_cereb_fit, V_cereb_fit) = simulate(params_fit, T1_regimen)

## Table 3 regimen [3]
(CLevo_T3_fit, CGut_T3_fit, C1855_T3_fit, C1896_T3_fit, Ce_T3_fit, Ce_eff_T3_fit,
 R_sys_T3_fit, SV_T3_fit, HR_T3_fit, CO_T3_fit, MAP_T3_fit, R_cereb_T3_fit, V_cereb_T3_fit) = simulate(params_fit, T3_regimen)

######################## Results of Fitted Parameters #########################

## Parameter table: initial vs fitted, standard error, and bound check
P_units = ["hrs", "-", "hrs", "hrs", "-", "hrs", "-", "ng/mL", "-", "ng/mL", "-", "ng/mL"]

print("\nFitted Model Parameters")
print(f"{'Parameter':<11}{'Initial':>9}{'Fitted':>10}{'SE':>10}{'Change':>9}  {'Units':<7}Set by")
print("-" * 70)
for i, name in enumerate(P_names):
    change = (params_fit[i] - params0[i]) / params0[i] * 100
    set_by = "Table 1" if i < N_PK else "Table 3"
    on_bound = "  (at bound)" if np.isclose(params_fit[i], [P_lower[i], P_upper[i]], rtol=1e-3).any() else ""
    print(f"{name:<11}{params0[i]:>9.4g}{params_fit[i]:>10.4g}{P_SE[i]:>10.3g}{change:>8.1f}%  {P_units[i]:<7}{set_by}{on_bound}")

## Fit quality: R^2 and RMSE before and after parameterisation. From [17]
# Table 3 in its original units: model % change applied to the Table 3 cohort baseline
def to_T3_units(y_model, y0_data):
    return y0_data * (1 + pct_curve(y_model) / 100)

comparisons = [
    ("T1 Levosimendan", T1_time, T1_Levo,    CLevo,             CLevo_fit,             "ng/mL"),
    ("T1 OR-1855",      T1_time, T1_m1,      C1855,             C1855_fit,             "ng/mL"),
    ("T1 OR-1896",      T1_time, T1_m2,      C1896,             C1896_fit,             "ng/mL"),
    ("T3 HR",  T3_t, T3_HR[1:],  to_T3_units(HR_T3, T3_HR0),   to_T3_units(HR_T3_fit, T3_HR0),   "bpm"),
    ("T3 MAP", T3_t, T3_MAP[1:], to_T3_units(MAP_T3, T3_MAP0), to_T3_units(MAP_T3_fit, T3_MAP0), "mmHg"),
    ("T3 CI",  T3_t, T3_CI[1:],  to_T3_units(CO_T3, T3_CI0),   to_T3_units(CO_T3_fit, T3_CI0),   "L/min/m^2"),
]

print("\nFit Quality (initial -> fitted)")
print(f"{'Output':<17}{'N':>3}{'R2 init':>9}{'R2 fit':>9}{'RMSE init':>11}{'RMSE fit':>10}  Units")
print("-" * 70)
for name, t_d, y_d, y_i, y_f, units in comparisons:
    R2_i, RMSE_i = fit_metrics(y_d, sample(y_i, t_d))
    R2_f, RMSE_f = fit_metrics(y_d, sample(y_f, t_d))
    print(f"{name:<17}{len(t_d):>3}{R2_i:>9.3f}{R2_f:>9.3f}{RMSE_i:>11.3f}{RMSE_f:>10.3f}  {units}")

############################## Plot Figure ####################################

## Figure 1: Table 1 data [1] vs initial and fitted model (Table 1 regimen)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
plot_fit(ax[0], CLevo, CLevo_fit, T1_time, T1_Levo,
         "Levosimendan Concentration (ng/mL)", "Levosimendan (Parent)")
plot_fit(ax[1], C1855, C1855_fit, T1_time, T1_m1,
         "OR-1855 Concentration (ng/mL)", "OR-1855 (Inactive Metabolite)")
plot_fit(ax[2], C1896, C1896_fit, T1_time, T1_m2,
         "OR-1896 Concentration (ng/mL)", "OR-1896 (Active Metabolite)")
fig.suptitle("Table 1: Levosimendan and Metabolites")
plt.tight_layout()
plt.show()

## Figure 2: Table 3 data [3] vs initial and fitted model (Table 3 regimen),
## model % change applied to the Table 3 cohort baseline
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
plot_fit(ax[0], to_T3_units(HR_T3, T3_HR0), to_T3_units(HR_T3_fit, T3_HR0), T3_time, T3_HR,
         "HR (bpm)", "Heart Rate")
plot_fit(ax[1], to_T3_units(MAP_T3, T3_MAP0), to_T3_units(MAP_T3_fit, T3_MAP0), T3_time, T3_MAP,
         "MAP (mmHg)", "Mean Arterial Pressure")
for a in ax:
    a.set_xlim(0, 2 * T3_time[-1])          # hrs, data window plus the same again
fig.suptitle("Table 3: Heart Rate and MAP")
plt.tight_layout()
plt.show()

## Code 1 plots: initial and fitted model (Table 1 regimen)
time = (np.arange(0, int(T // dt)) * dt) / 24 # days, x-axis in real time not step number

code1_plots = [
    (CLevo, CLevo_fit, "Levosimendan Concentration in Blood (ng/mL)", "Concentration of Levosimendan (Parent)"),
    (CGut,  CGut_fit,  "Gut Concentration (ng/mL)",                   "Gut Transit of OR-1855 metabolite"),
    (C1855, C1855_fit, "OR-1855 Concentration in Blood (ng/mL)",      "OR-1855 (Inactive Metabolite)"),
    (C1896, C1896_fit, "OR-1896 in Blood (ng/mL)",                    "OR-1896 (Active Metabolite)"),
    (np.array(MAP) / MMHG_TO_PA, np.array(MAP_fit) / MMHG_TO_PA,      "MAP (mmHg)", "Mean Arterial Pressure (MAP)"),
    (np.array(HR) * 60,          np.array(HR_fit) * 60,               "HR (bpm)",   "Heart Rate (HR)"),
    (np.array(V_cereb) * 100,    np.array(V_cereb_fit) * 100,         "Blood Velocity through MCA (cm/s)", "Cerebral Blood Flow Velocity"),
]
for y_i, y_f, ylabel, title in code1_plots:
    plt.figure()
    plt.plot(time, y_i, "r", label="Initial model")
    plt.plot(time, y_f, "b--", label="Fitted model")
    plt.ylabel(ylabel)
    plt.xlabel("Time (days)")
    plt.title(title)
    plt.legend()
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
