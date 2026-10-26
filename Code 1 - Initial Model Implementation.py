##############################################################################
#
# BIOE3001: Quantitative Methods in Biomedical Engineering
# Tevyn Vergara | 46421201
#
# Code #1: Initial Model Implementation 
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

####################### Dose Scheduling and Time Steps ########################
## Time course: set in days, runs in hrs

### Run Mode Toggle ###
# TRUE = scheduling on, dosing schedule to maintain homeostasis levels
# FALSE = scheduling off, 1 dose for shape and quantity analysis

repeat_dosing = True                # BOOLEAN, CHANGE WHEN NEEDED

if repeat_dosing:
    T_days = 30                     # days, time course to show repeat dosing
else:
    T_days = 12                     # days, time course

dt = 0.1                            # hrs, time step
T = T_days * 24                     # hrs, time course

############################# Parameters: Patient #############################
# Patient-Specific Body parameters 
BW = 80                             # kg, sample body weight from [1], [10]

# Unit conversion macros
MMHG_TO_PA = 133.322                # Pa per mmHg
ML_TO_M3 = 1e-6                     # m^3 per mL

# Patient-specific heart baselines 
HR0 = 70.0 / 60.0                   # Hz, baseline heart rate. From [10]
SV0 = 77.0 * ML_TO_M3               # m^3, baseline SV. From [11]
MAP0 = 105.0 * MMHG_TO_PA           # Pa, baseline MAP. From [11]

# Patient-specific brain baselines 
v_cereb0 = 0.40                     # m/s, baseline MCA velocity. From [11]
P_brain = 10.0 * MMHG_TO_PA         # Pa, intracranial pressure. From [12]
r_cereb = 0.00128                   # m, MCA M1 radius. From [15]

############################# Parameters: Dosing ##############################
# Dosing: 100 ng/kg/min IV for 24 hrs
# no loading dose (usual 50–200), From [16]

dose_load = 0                       # ng/kg, loading dose at t = 0
dose_sup = 100                      # ng/kg/min, supplementary dosing infusion rate
dose_inf = 24                       # hrs, infusion duration
dose_sch = 7                        # days, dose scheduling for repeated dosing

# Repeat dosing for scheduling
if repeat_dosing:
    T_rep = dose_sch * 24           # hrs, interval between infusion starts
    N_inf = int(np.ceil(T / T_rep)) # number of infusions 
else:
    T_rep = T                       # hrs, single dose
    N_inf = 1                       # one infusion only

# Time steps where repeat infusion starts 
start_steps = [int(round(k * T_rep / dt)) for k in range(1, N_inf)]

################### Parameters: Drug (Pharmacokinetics) #######################

# Parameters: Levosimendan (Active)
#
# Initial drug dosing based off Table 1 dosing regimen since Safer Care Victoria [16] 
# has similar dosing regimen to source
#
# Levosimendan enters at constant infusion rate but constant loss clears most
# of it on arrival. C0_levo is the t = 0 intercept of the line fitted to the 
# 2–24 hr data (no sample was taken at t = 0)

levo_half = 0.9                     # hrs, Levosimendan half-life. From [1]
k0 = (np.log(2) / levo_half) * dt   # 1/hrs, parent half-life decay (DERIVED)
f0 = 0.03                           # Free Fraction Levosimendan. From [4]

C_slope = 1.3388                    # ng/mL/hr, net rise during infusion (FITTED FROM [1])
C0_levo = 25.00                     # ng/mL, intercept at t=0 (CALCULATED FROM [1])
V_kg = 132                          # mL/kg, parent volume of distribution (BEHAVIOUR FROM [1])

# Dose conversions for main loop
R_inf = (dose_sup * 60 / V_kg) * dt # ng/mL per step, infusion input
phi = C_slope / (R_inf / dt)        # fraction of infused drug retained in plasma (DERIVED)
L_inf = (1 - phi) * R_inf           # ng/mL per step, loss during infusion
C_dose_load = dose_load / V_kg      # ng/mL, loading dose jump at t = 0

# Parameters: Gut Transit (delay before metabolites)
T_gut = 30                          # hrs, time when gut starts to release (FITTED FROM [1])
gut_rate = 53                       # hrs, gut release rate From [13]
kg = (np.log(2) / gut_rate) * dt    # 1/hrs, gut release rate as half life (DERIVED)

# Parameters: OR-1855 Metabolite (Inactive)
m1 = 0.05                           # Formation Fraction into OR-1855 metabolite. From [13]
Vm_ratio = 0.5                      # parent-to-metabolite volume ratio (FITTED FROM [1])

# Parameters: OR-1896 Metabolite (Active)
m2_half = 7.7                       # hrs, OR-1896 half-life (FITTED)
k2 = (np.log(2) / m2_half) * dt     # 1/hrs, half-life decay (DERIVED)
f2 = 0.61                           # Free Fraction OR-1896. From [1], [4]

# Parameters: Acetylation (OR-1855 -> OR-1896) through NAT2 enzyme
ac_rate = 9.6                       # hrs, acetylation rate. From [13]
kac = (np.log(2)/ac_rate)*dt        # 1/hrs, acetylation as half life(DERIVED)

################### Parameters: Body (Pharmacodynamics) #######################
# PD is kept in SI, patient baselines are defined at top

# Shared EC50 for SV, HR, R_sys and R_cereb
EC50 = 0.5                          # ng/mL, shared EC50 (Fitted)

# Hill Kinetics: drug effect on stroke volume
dSVmax = 0.28                       # fraction, maximum SV increase from Levosimendan. From [14]

# Hill Kinetics: drug effect on heart rate
Emax_HR = 0.11                      # fraction, maximum HR increase from Levosimendan. From [2],[3]

# Hill Kinetics: dilation of systemic and cerebral vessels
Emax_R = 0.4                        # fractional max R_sys and R_cereb drop. (Fitted)

# Derivation for baseline systemic resistance
# (how strong vessels oppose blood flow)
# Baseline systemic resistance = pressure / flow
CO0 = SV0 * HR0                     # m^3/s, baseline cardiac output (DERIVED)
R_sys0 = MAP0 / CO0                 # Pa.s/m^3, baseline systemic resistance (DERIVED)

### Parameters: Brain Block (MAP, Ce_HR --> V_cereb)

A_cereb = np.pi * (r_cereb) ** 2    # m^2, MCA cross-section

# Derivation for baseline cerebral resistance
# Baseline cerebral resistance = pressure / flow

CPP0 = MAP0 - P_brain               # Pa, baseline cerebral perfusion pressure (DERIVED)
Q_cereb0 = v_cereb0 * A_cereb       # m^3/s, baseline cerebral flow (DERIVED)
R_cereb0 = CPP0 / Q_cereb0          # Pa.s/m^3, baseline cerebral resistance (DERIVED)

# Effect delay for SV
keo_thalf = 10                      # hrs, compartment-effect time delay (Fitted)
k_eo = (np.log(2) / keo_thalf) * dt # 1/hrs, pre-multiplied by dt

# Fast parent-weighted effect (Ce_HR) on HR and resistance (R_sys, R_cereb):
# Not explicit but needed from back calibration of Code #2 to better fit PD Table 3 data
# Values fitted but parameters supported by [3] and [18]
p_HR = 6                            # parent potency on HR and vessels (Fitted)
keo_HR_thalf = 6                    # hrs, fast effect delay for HR and vessels (Fitted)
k_HR = (np.log(2)/keo_HR_thalf)*dt  # 1/hrs, pre-multiplied by dt

### Heart (Mass Transfer System) ###
#
# Takes effect-site drugs and runs three Hill Kinetic responses:
# Systemic resistance, stroke volume and heart rate (R_sys and HR use Ce_HR, SV uses Ce_eff)
# Outputs MAP --> Brain
#
def heart_block(Ce_eff_v, Ce_HR_v):
    R_sys_v = R_sys0 * (1 - Emax_R * Ce_HR_v / (EC50 + Ce_HR_v))
    SV_v = SV0 * (1 + dSVmax * Ce_eff_v / (EC50 + Ce_eff_v))
    HR_v = HR0 * (1 + Emax_HR * Ce_HR_v / (EC50 + Ce_HR_v))
    CO_v = SV_v * HR_v
    MAP_v = CO_v * R_sys_v
    return R_sys_v, SV_v, HR_v, CO_v, MAP_v

### Cerebral Vasculature (Mass Transfer System) ###
#
# Takes the fast effect-site drug (Ce_HR) and MAP (from the heart block)
# Combined lumped resistance and MAP to solve for blood velocity
# Equations from [5].
def cerebral_block(Ce_val, MAP_v):
    Cerebral_R = R_cereb0 * (1 - Emax_R * Ce_val / (EC50 + Ce_val))
    Blood_V = (MAP_v - P_brain) / (Cerebral_R * A_cereb)
    return Cerebral_R, Blood_V

######################### Initial Conditions ##################################
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

CLevo[0] = C0_levo + C_dose_load        # ng/mL, initial + loading dose
Cdose_load[0] = C_dose_load             # ng/mL, just loading dose
CGut[0]  = 0                            # ng/mL
C1855[0] = 0                            # ng/mL
C1896[0] = 0                            # ng/mL
Ce[0] = f0 * CLevo[0] + f2 * C1896[0]   # ng/mL, free active drug
Ce_eff[0] = 0                           # ng/mL
Ce_HR[0] = 0                            # ng/mL

# Patient baselines as initial conditions
R_sys[0], SV[0], HR[0], CO[0], MAP[0] = heart_block(Ce_eff[0], Ce_HR[0])
R_cereb[0], V_cereb[0] = cerebral_block(Ce_HR[0], MAP[0])

############################# Main Loop #######################################

for t in range(0, int(T // dt - 1)):

    ### Levosimendan (ACTIVE) ###

    # Loading dose: first-order decay from t = 0
    dCdose_load = -k0 * Cdose_load[t]
    Cdose_load[t + 1] = Cdose_load[t] + dCdose_load
    
    # Infusing if within dose_inf hrs of the current infusion start
    t_cycle = (t * dt) % T_rep       
    if t_cycle < dose_inf and t * dt < N_inf * T_rep:
        dCLevo = R_inf - L_inf + dCdose_load
        Form = m1 * Vm_ratio * (L_inf + k0 * Cdose_load[t])
        
    # Else, first order decay
    else:
        dCLevo = -k0 * CLevo[t]
        Form = m1 * Vm_ratio * k0 * CLevo[t]
    CLevo[t + 1] = CLevo[t] + dCLevo

    # Each repeat infusion starts with the same jump as t = 0
    if (t + 1) in start_steps:
        CLevo[t + 1] += C0_levo

    ### Gut Transit Compartment ###
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
    # Formed by acetylation, lost to self decay
    dC1896 = kac * C1855[t] - k2 * C1896[t]
    C1896[t + 1] = C1896[t] + dC1896

    ### Active Drug Concentration (PK Output)
    Ce[t + 1] = f0 * CLevo[t + 1] + f2 * C1896[t + 1]

    ### Effect Compartment (delay between plasma and tissue) ###
    Ce_eff[t + 1] = Ce_eff[t] + k_eo * (Ce[t] - Ce_eff[t])
    Ce_HR[t + 1] = Ce_HR[t] + k_HR * (p_HR * f0 * CLevo[t] + f2 * C1896[t] - Ce_HR[t])

    ### Heart PD (Mass Transfer System) ###
    R_sys[t + 1], SV[t + 1], HR[t + 1], CO[t + 1], MAP[t + 1] = heart_block(Ce_eff[t + 1], Ce_HR[t + 1])

    ### Cerebral Vasculature PD (Mass Transfer System) ###
    R_cereb[t + 1], V_cereb[t + 1] = cerebral_block(Ce_HR[t + 1], MAP[t + 1])

############################## Plot Figure ####################################

time = (np.arange(0, int(T // dt)) * dt) / 24 # days

## Levosimendan Concentration 
plt.figure()
plt.plot(time, CLevo)
plt.ylabel("Levosimendan Concentration in Blood (ng/mL)")
plt.xlabel("Time (days)")
plt.title("Concentration of Levosimendan (Parent)")
plt.show()

## Gut Transition of OR-1855
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
# [18] {Lilleberg, 2007}
