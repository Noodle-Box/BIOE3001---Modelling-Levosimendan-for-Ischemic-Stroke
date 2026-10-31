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
from SALib.sample import saltelli
from SALib.analyze import sobol

############################## RUN MODE #######################################
# Global toggle for run mode
# 
# FALSE: Just parameterisation of Code #1
# TRUE: Both paramterisation and sensitivity analysis observing Cerebral Blood Flow (V_cereb)

run_sensitivity = False             # BOOLEAN, CHANGE WHEN NEEDED

###############################################################################
###################### Section 1: Model Implementation #########################
###############################################################################

############################# Table 1 Input Data ##############################
# Levosimendan, OR-1855 and OR-1896

# Dosing regimen: 0.1 ug/kg/min (100 ng/kg/min) constant IV infusion for 24
# hrs, no loading bolus. From [1].

T1_dose = 100                       # ng/kg/min, dosing infusion rate
T1_T_inf = 24                       # hrs, infusion duration
T1_load = 0                         # ug/kg, no loading bolus. From [1]

# Patient specific parameters
T1_BW = 80                          # kg, sample body weight
# (space for Table 1 cohort demographics)

## TABLE 1 DATA: Clinical data of parent drug and metabolites, from Table 1 data
T1_time = [2, 4, 8, 12, 24, 25, 26, 28, 30, 36, 37, 38, 40, 44, 48, 60, 72, 96, 144, 192, 240]
T1_Levo = [32.5, 28.5, 34, 40, 59, 22.4, 10.4, 4.1, 1.12, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
T1_m1   = [0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.8, 0.9, 1.05, 1.2, 1.55, 2.1, 1.9, 1.95, 1.5, 0.7, 0.35, 0.15]
T1_m2   = [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.6, 0.7, 0.85, 1.1, 1.75, 2.35, 2.4, 2.35, 2.05, 1.2, 0.55, 0.3]

############################# Table 3 Input Data ##############################
# HR and MAP
# Dosing regimen: 12 ug/kg loading dose at t = 0, followed by 200 ng/kg/min
# constant IV infusion for 24 hrs. From [3].

T3_load  = 12                       # ug/kg, loading dose at t = 0 (instant bolus)
T3_dose  = 200                      # ng/kg/min, maintenance infusion rate
T3_T_inf = 24                       # hrs, maintenance infusion duration

# Patient specific parameters
T3_HR0  = 83                        # bpm, baseline HR, t = 0 row of [3]
T3_MAP0 = 84                        # mmHg, baseline MAP, t = 0 row of [3]
# T3_BW  =                          # kg, cohort body weight (TBA from [3] demographics)

# Clinical data
# TABLE 3 Heart Rate and MAP
T3_time = [0, 6, 12, 18, 24, 36, 48]              # hrs, time column
T3_HR   = [83, 95, 95.5, 96, 96, 94, 92]             # bpm, heart rate column
T3_MAP  = [84, 72, 75.5, 75, 76, 77, 78]             # mmHg, MAP column


############################### Time Course ###################################
# Time course: set in days, runs in hrs
# Single dose only: Tables 1 and 3 each report one dosing course

T_days = 12                         # days, time course (covers the 240 hr Table 1 data)

dt = 0.1                            # hrs, time step
T = T_days * 24                     # hrs, time course

############################# Parameters: Patient #############################
# Patient-Specific Body parameters
BW = 80                             # kg, sample body weight from [1], [7]

# Unit conversion macros
MMHG_TO_PA = 133.322                # Pa per mmHg
ML_TO_M3 = 1e-6                     # m^3 per mL

# Patient-specific heart baselines
HR0 = 70.0 / 60.0                   # Hz, baseline heart rate. From [7]
SV0 = 77.0 * ML_TO_M3               # m^3, baseline SV. From [8]
MAP0 = 105.0 * MMHG_TO_PA           # Pa, baseline MAP. From [8]

# Patient-specific brain baselines
v_cereb0 = 0.40                     # m/s, baseline MCA velocity. From [8]
P_brain = 10.0 * MMHG_TO_PA         # Pa, intracranial pressure. From [11]
r_cereb = 0.00128                   # m, MCA M1 radius. From [12]

############################# Parameters: Dosing ##############################
# Each regimen = [loading dose (ng/kg), infusion rate (ng/kg/min), infusion duration (hrs)]
# Built from the Table 1 and Table 3 input data, so simulate() can run either one
# Usual range: 0.05 - 0.2ug/kg/min. From [6]

T1_regimen = [T1_load * 1000, T1_dose, T1_T_inf]  # Table 1 [1], ug/kg x 1000 -> ng/kg
T3_regimen = [T3_load * 1000, T3_dose, T3_T_inf]  # Table 3 [3], ug/kg x 1000 -> ng/kg

################### Parameters: Drug (Pharmacokinetics) #######################

# Parameters: Levosimendan (Active)
#
# Initial drug dosing based off Table 1 dosing regimen since Safer Care Victoria [6] 
# has similar dosing regimen to source
#
# Levosimendan enters at constant infusion rate but constant loss clears most
# of it on arrival. C0_levo is the t = 0 intercept of the line fitted to the 
# 2–24 hr data (no sample was taken at t = 0)

levo_half = 0.9                          # hrs, Levosimendan half-life. From [1]
k0 = (np.log(2) / levo_half) * dt        # 1/hrs, parent half-life decay (DERIVED)
f0 = 0.03                                # Free Fraction Levosimendan. From [4]

C_slope = 1.3388                         # ng/mL/hr, net rise during infusion (FITTED FROM [1])
C0_levo = 25.00                          # ng/mL, intercept at t=0 (CALCULATED FROM [1])
V_kg = 132                               # mL/kg, parent volume of distribution (BEHAVIOUR FROM [1])

# Dose conversions (dose-independent). R_inf, L_inf, C0_inf and C_dose_load
# depend on the regimen, so they are calculated inside simulate()
dose_ref = T1_dose                       # ng/kg/min, Table 1 dose that C_slope and C0_levo were measured at
phi = C_slope / (dose_ref * 60 / V_kg)   # fraction of infused drug retained in plasma, fixed drug property (DERIVED)
C0_per_dose = C0_levo / dose_ref         # ng/mL per (ng/kg/min), intercept per unit dose (DERIVED)

# Parameters: Gut Transit (delay before metabolites)
T_gut = 30                               # hrs, time when gut starts to release (FITTED FROM [1])
gut_rate = 53                            # hrs, gut release rate From [9]

# Parameters: OR-1855 Metabolite (Inactive)
m1 = 0.05                                # Formation Fraction into OR-1855 metabolite. From [9]
Vm_ratio = 0.5                           # parent-to-metabolite volume ratio (FITTED FROM [1])

# Parameters: OR-1896 Metabolite (Active)
m2_half = 7.7                            # hrs, OR-1896 half-life (FITTED)
f2 = 0.61                                # Free Fraction OR-1896. From [1], [4]

# Parameters: Acetylation (OR-1855 -> OR-1896) through NAT2 enzyme
ac_rate = 9.6                            # hrs, acetylation rate. From [9]

################### Parameters: Body (Pharmacodynamics) #######################
# PD is kept in SI, patient baselines are defined at top

# Shared EC50 for SV, HR, R_sys and R_cereb
EC50 = 0.5                               # ng/mL, shared EC50 (Fitted)

# Hill Kinetics: drug effect on stroke volume
dSVmax = 0.28                            # fraction, maximum SV increase from Levosimendan. From [10]

# Hill Kinetics: drug effect on heart rate
Emax_HR = 0.11                           # fraction, maximum HR increase from Levosimendan. From [2],[3]

# Fast parent-weighted effect (Ce_HR) on HR and resistance (R_sys, R_cereb):
# Not explicit but needed from back calibration of Code #2 to better fit PD Table 3 data
# Values fitted but parameters supported by [3] and [14]
k_HR = 6                                 # parent potency on HR and vessels (Fitted)
keo_HR_thalf = 6                         # hrs, fast effect delay for HR and vessels (Fitted)
k_eo_HR = (np.log(2)/keo_HR_thalf)*dt       # 1/hrs, pre-multiplied by dt

# Hill Kinetics: dilation of systemic and cerebral vessels
Emax_R = 0.4                             # fractional max R_sys and R_cereb drop. (Fitted)

# Derivation for baseline systemic resistance
# (how strong vessels oppose blood flow)
# Baseline systemic resistance = pressure / flow
CO0 = SV0 * HR0                          # m^3/s, baseline cardiac output (DERIVED)
R_sys0 = MAP0 / CO0                      # Pa.s/m^3, baseline systemic resistance (DERIVED)

### Parameters: Brain Block (MAP, Ce_HR --> V_cereb)
A_cereb = np.pi * (r_cereb) ** 2         # m^2, MCA cross-section

# Derivation for baseline cerebral resistance
# Baseline cerebral resistance = pressure / flow

CPP0 = MAP0 - P_brain                    # Pa, baseline cerebral perfusion pressure (DERIVED)
Q_cereb0 = v_cereb0 * A_cereb            # m^3/s, baseline cerebral flow (DERIVED)
R_cereb0 = CPP0 / Q_cereb0               # Pa.s/m^3, baseline cerebral resistance (DERIVED)

# Effect delay for SV
keo_thalf = 10                           # hrs, compartment-effect time delay (Fitted)

############################ Parameters to Fit ################################
# Initial values (starting guesses for the optimiser) are the Code 1 values above.
# Passed to the fit as one parameter list. From [13]. Order is fixed:
# PK set by Table 1 first, then PD set by Table 3

P_names = ["gut_rate", "Vm_ratio", "ac_rate", "m2_half",   # PK, Table 1
           "f0", "keo_thalf",                              # PD, Table 3
           "Emax_R", "EC50", "dSVmax", "Emax_HR",
           "k_HR", "keo_HR_thalf"]                          # PD model structure (fixed)

params0 = [gut_rate, Vm_ratio, ac_rate, m2_half,
           f0, keo_thalf,
           Emax_R, EC50, dSVmax, Emax_HR,
           k_HR, keo_HR_thalf]

# Heart (Mass Transfer System) 
#
# Takes effect-site drugs and runs three Hill Kinetic responses:
# Systemic resistance, stroke volume and heart rate (R_sys and HR use Ce_HR, SV uses Ce_eff)
# P_HB = [Emax_R, EC50, dSVmax, Emax_HR] (fitted values, EC50 shared by every effect)
# Outputs MAP --> Brain
#
def heart_block(Ce_eff_v, Ce_HR_v, P_HB):
    Emax_R, EC50, dSVmax, Emax_HR = P_HB
    R_sys_v = R_sys0 * (1 - Emax_R * Ce_HR_v / (EC50 + Ce_HR_v))
    SV_v = SV0 * (1 + dSVmax * Ce_eff_v / (EC50 + Ce_eff_v))
    HR_v = HR0 * (1 + Emax_HR * Ce_HR_v / (EC50 + Ce_HR_v))
    CO_v = SV_v * HR_v
    MAP_v = CO_v * R_sys_v
    return R_sys_v, SV_v, HR_v, CO_v, MAP_v

# Cerebral Vasculature (Mass Transfer System)
#
# Takes the fast effect-site drug (Ce_HR) and MAP (from the heart block)
# Combined lumped resistance and MAP to solve for blood velocity
# Emax_v, EC50_v = shared Emax_R and EC50 (same as the heart block, so v_cereb
# does not dip below baseline)
# Equations from [5].
def cerebral_block(Ce_val, MAP_v, Emax_v, EC50_v):
    Cerebral_R = R_cereb0 * (1 - Emax_v * Ce_val / (EC50_v + Ce_val))
    Blood_V = (MAP_v - P_brain) / (Cerebral_R * A_cereb)
    return Cerebral_R, Blood_V

############################# Model Function ##################################
# Runs the Code 1 model for one parameter set and one dosing regimen, so the
# parameterisation can call it many times. Model loop inside a function of
# the parameters, from [13]
#
# params  = fitted parameters, same order as P_names
# regimen = [loading dose (ng/kg), infusion rate (ng/kg/min), infusion duration (hrs)]
#
def simulate(params, regimen):

    # Unpack parameters and regimen
    gut_rate, Vm_ratio, ac_rate, m2_half, f0, keo_thalf = params[0:6]
    P_HB = params[6:10]                 # heart block PD parameters
    k_HR, keo_HR_thalf = params[10:12]  # fast effect site (HR and vessels)
    dose_load, dose_sup, dose_inf = regimen[0], regimen[1], regimen[2]

    # Rate constants (pre-multiplied by dt)
    kg = (np.log(2) / gut_rate) * dt        # 1/hrs, gut release rate as half life (DERIVED)
    kac = (np.log(2) / ac_rate) * dt        # 1/hrs, acetylation as half life (DERIVED)
    k2 = (np.log(2) / m2_half) * dt         # 1/hrs, OR-1896 half-life decay (DERIVED)
    k_eo = (np.log(2) / keo_thalf) * dt     # 1/hrs, pre-multiplied by dt
    k_eo_HR = (np.log(2) / keo_HR_thalf) * dt  # 1/hrs, pre-multiplied by dt

    # Dose conversions for main loop
    R_inf = (dose_sup * 60 / V_kg) * dt     # ng/mL per step, infusion input
    C0_inf = C0_per_dose * dose_sup         # ng/mL, initial jump for this dose
    L_inf = (1 - phi) * R_inf               # ng/mL per step, loss during infusion
    C_dose_load = dose_load / V_kg          # ng/mL, loading dose jump at t = 0

    # Initial Conditions
    CLevo      = [None] * int(T // dt)      # ng/mL, Levosimendan (parent)
    Cdose_load = [None] * int(T // dt)      # ng/mL, loading dose part of the parent
    CGut       = [None] * int(T // dt)      # ng/mL, gut transit compartment
    C1855      = [None] * int(T // dt)      # ng/mL, OR-1855
    C1896      = [None] * int(T // dt)      # ng/mL, OR-1896
    Ce         = [None] * int(T // dt)      # ng/mL, free active drug in plasma
    Ce_eff     = [None] * int(T // dt)      # ng/mL, effect-site concentration (lagged Ce)
    Ce_HR      = [None] * int(T // dt)      # ng/mL, fast effect-site conc (HR, vessels)

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
    Ce[0] = f0 * CLevo[0] + f2 * C1896[0]   # ng/mL, free active drug
    Ce_eff[0] = 0                           # ng/mL
    Ce_HR[0] = 0                            # ng/mL

    # Patient baselines as initial conditions
    R_sys[0], SV[0], HR[0], CO[0], MAP[0] = heart_block(Ce_eff[0], Ce_HR[0], P_HB)
    R_cereb[0], V_cereb[0] = cerebral_block(Ce_HR[0], MAP[0], P_HB[0], P_HB[1])

    ### Main Loop ###

    for t in range(0, int(T // dt - 1)):

        # Levosimendan (ACTIVE)

        # Loading dose: first-order decay from t = 0
        dCdose_load = -k0 * Cdose_load[t]
        Cdose_load[t + 1] = Cdose_load[t] + dCdose_load
        
        # Infusing if within dose_inf hrs
        if t * dt < dose_inf:
            dCLevo = R_inf - L_inf + dCdose_load
            Form = m1 * Vm_ratio * (L_inf + k0 * Cdose_load[t])
            
        # Else, first order decay
        else:
            dCLevo = -k0 * CLevo[t]
            Form = m1 * Vm_ratio * k0 * CLevo[t]
        CLevo[t + 1] = CLevo[t] + dCLevo

        ### Gut Transit Compartment ###
        if t * dt < T_gut:
            Rel = 0
        else:
            Rel = kg * CGut[t]

        dCGut = Form - Rel
        CGut[t + 1] = CGut[t] + dCGut

        # OR-1855 (INACTIVE Metabolite)
        # Formed by gut metabolism, lost from acetylation (1896) 
        dC1855 = Rel - kac * C1855[t]
        C1855[t + 1] = C1855[t] + dC1855

        # OR-1896 (ACTIVE Metabolite)
        # Formed by acetylation, lost to self decay
        dC1896 = kac * C1855[t] - k2 * C1896[t]
        C1896[t + 1] = C1896[t] + dC1896

        # Active Drug Concentration (PK Output)
        Ce[t + 1] = f0 * CLevo[t + 1] + f2 * C1896[t + 1]

        # Effect Compartment (delay between plasma and tissue)
        Ce_eff[t + 1] = Ce_eff[t] + k_eo * (Ce[t] - Ce_eff[t])
        Ce_HR[t + 1] = Ce_HR[t] + k_eo_HR * (k_HR * f0 * CLevo[t] + f2 * C1896[t] - Ce_HR[t])

        # Heart PD (Mass Transfer System)
        R_sys[t + 1], SV[t + 1], HR[t + 1], CO[t + 1], MAP[t + 1] = heart_block(Ce_eff[t + 1], Ce_HR[t + 1], P_HB)

        # Cerebral Vasculature PD (Mass Transfer System)
        R_cereb[t + 1], V_cereb[t + 1] = cerebral_block(Ce_HR[t + 1], MAP[t + 1], P_HB[0], P_HB[1])

    return (CLevo, CGut, C1855, C1896, Ce, Ce_eff,
            R_sys, SV, HR, CO, MAP, R_cereb, V_cereb)

############################ Initial Model Runs ###############################
# Code 1 initial values, run with each table's dosing regimen

# Table 1 regimen [1]
(CLevo, CGut, C1855, C1896, Ce, Ce_eff,
 R_sys, SV, HR, CO, MAP, R_cereb, V_cereb) = simulate(params0, T1_regimen)

# Table 3 regimen [3]
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

# Model value at each data time (index = t / dt). From [13]
def sample(curve, t_data):
    idx = [int(round(t / dt)) for t in t_data]
    return np.array(curve, dtype=float)[idx]

# R^2 and RMSE. From [13]
def fit_metrics(y_data, y_model):
    y_data = np.array(y_data, dtype=float)
    residual = y_data - y_model
    R2 = 1 - np.sum(residual ** 2) / np.sum((y_data - y_data.mean()) ** 2)
    RMSE = np.sqrt(np.mean(residual ** 2))
    return R2, RMSE

# Initial model, fitted model and data on one subplot. From [13]
def plot_fit(ax, y_init, y_fit, t_data, y_data, ylabel, title):
    ax.plot(time_hrs, y_init, "r", label="Initial model")
    ax.plot(time_hrs, y_fit, "b--", label="Fitted model")
    ax.plot(t_data, y_data, "ko", label="Data")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (hrs)")
    ax.set_title(title)
    ax.legend()
    
    
###############################################################################
###################### Section 2: Model Parameterisation ######################
###############################################################################

# One parameter set to fit both studies. This will make a robust-general PKPD model
#   1. PK fitted to Table 1 (PD has no effect on Table 1)
#   2. PD fitted to Table 3 (fitted PK held fixed)
#
# least_squares with bounds keeps parameters physical
# 
# Implementation from [13]

N_PK = 4                               # first 4 P_names are PK

# Bounds (same order as P_names)
P_lower = [0.5, 0.01, 0.5, 0.5, 0.001, 0.1, 0, 0.001, 0, 0, 1, 0.1]
P_upper = [200, 10, 200, 200, 1, 72, 1, 10, 1, 1, 50, 72]

# Table 1: each curve divided by its own range
T1_range = [np.ptp(T1_Levo), np.ptp(T1_m1), np.ptp(T1_m2)]  # ng/mL, max - min

# PD parameters held fixed 
P_fixed = ["f0", "EC50", "dSVmax", "k_HR", "keo_HR_thalf"]
PD_names = P_names[N_PK:]
PD_free = [n for n in PD_names if n not in P_fixed]
i_free = [P_names.index(n) for n in PD_free]

# Full PD listL fitted free values + fixed Code #1 values
def build_PD(x):
    P = dict(zip(P_names, params0))
    P.update(zip(PD_free, x))
    return [P[n] for n in PD_names]

# Table 3: % change from baseline (t=0 left out)
T3_t = T3_time[1:]
T3_HR_pct = pct(T3_HR, T3_HR0)[1:]
T3_MAP_pct = pct(T3_MAP, T3_MAP0)[1:]

# Table 3 weights: each output divided by its max % change, then split 50/50
# No Cardiac Index (CI) needed
W_HR, W_MAP = 0.5, 0.5
T3_scale = [np.max(np.abs(T3_HR_pct)), np.max(np.abs(T3_MAP_pct))]

# Residual functions: data - model at the data times. From [13]
def funcPK(params_PK, params_PD):
    CLevo_m, CGut_m, C1855_m, C1896_m = simulate(list(params_PK) + list(params_PD), T1_regimen)[0:4]
    res_Levo = (np.array(T1_Levo) - sample(CLevo_m, T1_time)) / T1_range[0]
    res_1855 = (np.array(T1_m1) - sample(C1855_m, T1_time)) / T1_range[1]
    res_1896 = (np.array(T1_m2) - sample(C1896_m, T1_time)) / T1_range[2]
    return np.concatenate([res_Levo, res_1855, res_1896])

def funcPD(params_PD_free, params_PK):
    HR_m, CO_m, MAP_m = simulate(list(params_PK) + build_PD(params_PD_free), T3_regimen)[8:11]
    res_HR = W_HR * (T3_HR_pct - sample(pct_curve(HR_m), T3_t)) / T3_scale[0]
    res_MAP = W_MAP * (T3_MAP_pct - sample(pct_curve(MAP_m), T3_t)) / T3_scale[1]
    return np.concatenate([res_HR, res_MAP])

# PK fitted to Table 1
fit_PK = least_squares(funcPK, params0[:N_PK], args=(params0[N_PK:],),
                       bounds=(P_lower[:N_PK], P_upper[:N_PK]))

# PD fitted to Table 3
fit_PD = least_squares(funcPD, [params0[i] for i in i_free], args=(fit_PK.x,),
                       bounds=([P_lower[i] for i in i_free], [P_upper[i] for i in i_free]))

# One parameter set
params_fit = list(fit_PK.x) + build_PD(fit_PD.x)

# Standard errors from the Jacobian: cov = (J^T J)^-1 x SSR / (N - p)
def std_err(fit):
    J = fit.jac
    cov = np.linalg.pinv(J.T @ J) * np.sum(fit.fun ** 2) / (len(fit.fun) - len(fit.x))
    return np.sqrt(np.diag(cov))

SE_PD = dict(zip(PD_free, std_err(fit_PD)))
P_SE = list(std_err(fit_PK)) + [SE_PD.get(n, np.nan) for n in PD_names]   # nan = fixed

############################# Fitted Model Runs ###############################
# Fitted parameters run with each table's regimen

# Table 1 regimen [1]
(CLevo_fit, CGut_fit, C1855_fit, C1896_fit, Ce_fit, Ce_eff_fit,
 R_sys_fit, SV_fit, HR_fit, CO_fit, MAP_fit, R_cereb_fit, V_cereb_fit) = simulate(params_fit, T1_regimen)

# Table 3 regimen [3]
(CLevo_T3_fit, CGut_T3_fit, C1855_T3_fit, C1896_T3_fit, Ce_T3_fit, Ce_eff_T3_fit,
 R_sys_T3_fit, SV_T3_fit, HR_T3_fit, CO_T3_fit, MAP_T3_fit, R_cereb_T3_fit, V_cereb_T3_fit) = simulate(params_fit, T3_regimen)

######################## Results of Fitted Parameters #########################

# Parameter table
P_units = ["hrs", "-", "hrs", "hrs", "-", "hrs", "-", "ng/mL", "-", "-", "-", "hrs"]

print("\nFitted Model Parameters")
print(f"{'Parameter':<13}{'Initial':>9}{'Fitted':>10}{'SE':>10}{'Change':>9}  {'Units':<7}Set by")
print("-" * 72)
for i, name in enumerate(P_names):
    change = (params_fit[i] - params0[i]) / params0[i] * 100
    set_by = "Table 1" if i < N_PK else ("Fixed" if name in P_fixed else "Table 3")
    SE_str = "-" if name in P_fixed else f"{P_SE[i]:.3g}"
    on_bound = "  (at bound)" if np.isclose(params_fit[i], [P_lower[i], P_upper[i]], rtol=1e-3).any() else ""
    print(f"{name:<13}{params0[i]:>9.4g}{params_fit[i]:>10.4g}{SE_str:>10}{change:>8.1f}%  {P_units[i]:<7}{set_by}{on_bound}")

# Fit quality: R^2 and RMSE, initial vs fitted. From [13]
# Model % change converted to Table 3 units (bpm, mmHg)
def to_T3_units(y_model, y0_data):
    return y0_data * (1 + pct_curve(y_model) / 100)

comparisons = [
    ("T1 Levosimendan", T1_time, T1_Levo,    CLevo,             CLevo_fit,             "ng/mL"),
    ("T1 OR-1855",      T1_time, T1_m1,      C1855,             C1855_fit,             "ng/mL"),
    ("T1 OR-1896",      T1_time, T1_m2,      C1896,             C1896_fit,             "ng/mL"),
    ("T3 HR",  T3_t, T3_HR[1:],  to_T3_units(HR_T3, T3_HR0),   to_T3_units(HR_T3_fit, T3_HR0),   "bpm"),
    ("T3 MAP", T3_t, T3_MAP[1:], to_T3_units(MAP_T3, T3_MAP0), to_T3_units(MAP_T3_fit, T3_MAP0), "mmHg"),
]

print("\nFit Quality (initial -> fitted)")
print(f"{'Output':<17}{'N':>3}{'R2 init':>9}{'R2 fit':>9}{'RMSE init':>11}{'RMSE fit':>10}  Units")
print("-" * 70)
for name, t_d, y_d, y_i, y_f, units in comparisons:
    R2_i, RMSE_i = fit_metrics(y_d, sample(y_i, t_d))
    R2_f, RMSE_f = fit_metrics(y_d, sample(y_f, t_d))
    print(f"{name:<17}{len(t_d):>3}{R2_i:>9.3f}{R2_f:>9.3f}{RMSE_i:>11.3f}{RMSE_f:>10.3f}  {units}")

############################## Plot Figure ####################################

# Figure 1: Table 1 data [1] vs model
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
plot_fit(ax[0], CLevo, CLevo_fit, T1_time, T1_Levo,
         "Levosimendan Concentration (ng/mL)", "Levosimendan (Parent)")
plot_fit(ax[1], C1855, C1855_fit, T1_time, T1_m1,
         "OR-1855 Concentration (ng/mL)", "OR-1855 (Inactive Metabolite)")
plot_fit(ax[2], C1896, C1896_fit, T1_time, T1_m2,
         "OR-1896 Concentration (ng/mL)", "OR-1896 (Active Metabolite)")
fig.suptitle("PK: Levosimendan and Metabolites (Dosing regimen from Table 1")
plt.tight_layout()
plt.show()

# Figure 2: Table 3 data [3] vs model
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
plot_fit(ax[0], to_T3_units(HR_T3, T3_HR0), to_T3_units(HR_T3_fit, T3_HR0), T3_time, T3_HR,
         "HR (bpm)", "Heart Rate")
plot_fit(ax[1], to_T3_units(MAP_T3, T3_MAP0), to_T3_units(MAP_T3_fit, T3_MAP0), T3_time, T3_MAP,
         "MAP (mmHg)", "Mean Arterial Pressure")
for a in ax:
    a.set_xlim(0, 2 * T3_time[-1])          
fig.suptitle("PD: Heart Rate and MAP (Dosing regimen from Table 3)")
plt.tight_layout()
plt.show()

# Code 1 plots: initial vs fitted
time = (np.arange(0, int(T // dt)) * dt) / 24 # days

code1_plots = [
    (CLevo, CLevo_fit, "Levosimendan Concentration in Blood (ng/mL)", "Parameterised Levosimendan Concentration"),
    (CGut,  CGut_fit,  "Gut Concentration (ng/mL)",                   "Parameterised Gut Transition"),
    (C1855, C1855_fit, "OR-1855 Concentration in Blood (ng/mL)",      "Parameterised OR-1855 Concentration"),
    (C1896, C1896_fit, "OR-1896 in Blood (ng/mL)",                    "Parameterised OR-1896 Concentration"),
    (np.array(MAP) / MMHG_TO_PA, np.array(MAP_fit) / MMHG_TO_PA,      "MAP (mmHg)", "Parameterised MAP"),
    (np.array(HR) * 60,          np.array(HR_fit) * 60,               "HR (bpm)",   "Parameterised HR"),
    (np.array(V_cereb) * 100,    np.array(V_cereb_fit) * 100,         "Blood Velocity through MCA (cm/s)", "Parameterised CBF"),
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

###############################################################################
###################### Section 3: Sensitivity Analysis ########################
###############################################################################

# Sobol sensitivity analysis to see how much each parameter contributes to Cerebral Blood Flow

# Two observation states:
#       1. Peak (max) v_cereb
#       2. v_cereb at day 7 (when drops <50cm/s) 
#
# All parameters varied together by ±20 %
# S1 = effect of parameter alone
# ST = effect along with interactions with others
#
# N_sobol * 26 = 26624 runs 
#
# Implementation from [13]

N_sobol = 1024      # Saltelli base samples
t_obs = 7           # days 

# Model outputs for one parameter set
def funcS(params):
    out = simulate(list(params), T1_regimen)
    CBF = np.array(out[12]) * 100               # cm/s, cerebral blood flow
    return [CBF.max(), CBF[int(round(t_obs * 24 / dt))]]

S_outputs = ["v_cereb peak (cm/s)", f"v_cereb day {t_obs} (cm/s)"]

if run_sensitivity:

    # Parameters and and ±20 % bounds
    problem = {
        'num_vars': len(P_names),
        'names': P_names,
        'bounds': [[p * 0.8, p * 1.2] for p in params_fit]}

    param_values = saltelli.sample(problem, N_sobol)    # Saltelli sampler
    Y = np.zeros([param_values.shape[0], len(S_outputs)])

    # Runs each sample through the model
    for i, X in enumerate(param_values):
        Y[i] = funcS(X)
        if i % (len(param_values) // 10) == 0:
            print(f"Sobol runs: {i}/{len(param_values)}")

    # Sobol indices for each output
    Si_all = [sobol.analyze(problem, Y[:, k]) for k in range(len(S_outputs))]

    print("\nSobol Sensitivity Analysis")
    for name, Si in zip(S_outputs, Si_all):
        print(f"\n{name}")
        print(f"{'Parameter':<13}{'S1':>8}{'S1 conf':>9}{'ST':>8}{'ST conf':>9}")
        print("-" * 47)
        for j, p in enumerate(P_names):
            print(f"{p:<13}{Si['S1'][j]:>8.3f}{Si['S1_conf'][j]:>9.3f}{Si['ST'][j]:>8.3f}{Si['ST_conf'][j]:>9.3f}")

    # S1 and ST bar chart for each output
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(len(P_names))
    for a, name, Si in zip(ax.flat, S_outputs, Si_all):
        a.bar(x - 0.2, Si['S1'], 0.4, label="S1 (parameter alone)")
        a.bar(x + 0.2, Si['ST'], 0.4, label="ST (with interactions)")
        a.set_xticks(x, P_names, rotation=45, ha="right")
        a.set_ylabel("Sobol index")
        a.set_title(name)
        a.legend()
    fig.suptitle("Sobol Sensitivity Analysis (±20 %)")
    plt.tight_layout()
    plt.show()

############################## References #####################################
# Declaration of AI use
# Anthropic, "Claude, Opus 5.5," Claude, 22 October 2026 [Online]. https://claude.ai/new
#
# Full references details are in report
# [1] (Puttonen et al., 2008)
# [2] (Kivikko et al., 2015)
# [3] (Álvarez et al., 2006)
# [4] (Antila et al., 2004)
# [5] (Robinson & Allenby, 2026)
# [6] (Safer Care Victoria, 2018)
# [7] (Böhm et al., 2012)
# [8] (Miller et al., 2021)
# [9] (Bertin et al., 2026)
# [10] (Slawsky et al., 2000)
# [11] (Hood et al., 2023)
# [12] (Mirza et al., 2024)
# [13] (Mann & Allenby, 2026b)
# [14] (Lilleberg et al., 2007)
