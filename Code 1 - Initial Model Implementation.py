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

############################### Time Course ###################################
## Time course: set in days (runs in hrs for better computation accuracy)
T_days = 12                         # days, time course (effective drug time, calibrated to Table 1)
dt = 0.1                            # hrs, time step
T = T_days * 24                     # hrs, time course

############################# Parameters: Patient #############################
# Patient-Specific Body parameters
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
# Dosing regimen: 0.1 ug/kg/min (100 ng/kg/min) constant IV infusion for 24 hrs
# No loading dose
# Usual range: 0.05 - 0.2ug/kg/min
# From [16]

dose_load = 0                       # ng/kg, loading dose at t = 0
dose_sup = 100                      # ng/kg/min, supplementary dosing infusion rate
T_inf = 24                          # hrs, infusion duration

################### Parameters: Drug (Pharmacokinetics) #######################

# Parameters: Levosimendan (Active)
#
# Initial drug dosing based off Table 1 dosing regimen since Safer Care Victora [16] 
# has similar dosing regimen to source
#
# Levosimendan enters at constant infusion rate but constant loss clears most
# of it on arrival. C_int is the t = 0 intercept of the line fitted to the 
# 2–24 hr data (no sample was taken at t = 0)

levo_half = 0.9                     # hrs, Levosimendan half-life. From [1]
k0 = (np.log(2) / levo_half) * dt   # 1/Hrs, half-life decay for infusion and loading dose
f0 = 0.03                           # Free Fraction Levosimendan. From [4]

C_slope = 1.3388                    # ng/mL/hr, net rise during infusion (FITTED FROM [1])
C_int = 25.00                       # ng/mL, intercept at t=0 (CALCULATED FROM [1])
V_kg = 132                          # mL/kg, parent volume of distribution (BEHAVIOUR FROM [1])

# Dose conversions for main loop
R = (dose_sup * 60 / V_kg) * dt     # ng/mL per step, infusion input
phi = C_slope / (R / dt)            # fraction of infused drug retained in plasma (DERIVED)
L = (1 - phi) * R                   # ng/mL per step, loss during infusion
C0_levo = C_int                     # ng/mL, initial condition (fast distribution phase)
C_dose_load = dose_load / V_kg      # ng/mL, loading dose jump at t = 0

# Parameters: Gut Transit Compartment (Time Delay between parent and metabolites)
T_gut = 30                          # hrs, time when gut starts to release (FITTED FROM [1])
gut_rate = 53                       # hrs, gut release rate. From [13]
kg = (np.log(2) / gut_rate) * dt    # 1/hrs, gut release rate (DERIVED)

# Parameters: OR-1855 Metabolite (Inactive)
m1 = 0.05                           # Formation Fraction into OR-1855 metabolite. From [13]
Vm_ratio = 0.5                      # parent-to-metabolite volume ratio (FITTED FROM [1]

# Parameters: OR-1896 Metabolite (Active)
m2_half = 7.7                       # Hrs, OR-1896 half-life (FITTED)
k2 = (np.log(2) / m2_half) * dt     # 1/Hrs, half-life decay (DERIVED)
f2 = 0.61                           # Free Fraction OR-1896. From [1], [4]

# Parameters: Acetylation (OR-1855 -> OR-1896) through NAT2 enzyme
ac_rate = 9.6                       # Hrs, Acetylation rate. From [13]
kac = (np.log(2)/ac_rate)*dt        # 1/Hrs, acetylation (DERIVED)

################### Parameters: Body (Pharmacodynamics) #######################
# PD is kept in SI for easy conversion
# Patient baselines are defined at top

# Hill Kinetics: drug effect on stroke volume
dSVmax = 0.28                       # %, maximum SV increase from Levosimendan. From [14]
EC50_SV = 0.497                     # ng/mL (Fitted)

# Hill Kinetics drug effect on heart rate
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
k_eo = (np.log(2) / keo_thalf) * dt # 1/hrs, pre-multiplied by dt

### Heart (Mass Transfer System) ###
#
# Takes effective drug from dose and lags it to Ce_eff
# Then, runs three Hill Kinetic responses:
# Systemic resistance, stroke volume and heart rate.
# Outputs MAP and Ce_eff --> Brain
#
def heart_block(Ce_val, Ce_eff_prev):
    Ce_eff_v = Ce_eff_prev + k_eo * (Ce_val - Ce_eff_prev)
    R_sys_v = R_sys0 * (1 - Emax_R * Ce_eff_v / (EC50_sys + Ce_eff_v))
    SV_v = SV0 * (1 + dSVmax * Ce_eff_v / (EC50_SV + Ce_eff_v))
    HR_v = HR0 * (1 + Emax_HR * Ce_eff_v / (EC50_HR + Ce_eff_v))
    CO_v = SV_v * HR_v
    MAP_v = CO_v * R_sys_v
    return Ce_eff_v, R_sys_v, SV_v, HR_v, CO_v, MAP_v

### Cerebral Vasculature (Mass Transfer System) ###
#
# Takes lagged effective drug and MAP (both from the heart block)
# Combined lumped resistance and MAP to solve for blood velocity
# Equations from [5].
def cerebral_block(Ce_val, MAP_v):
    Cerebral_R = R_cereb0 * (1 - Emax_cereb * Ce_val / (EC50_cereb + Ce_val))
    Blood_V = (MAP_v - P_brain) / (Cerebral_R * A_cereb)
    return Cerebral_R, Blood_V

######################### Initial Conditions ##################################

CLevo = [None] * int(T // dt)       # ng/mL, Levosimendan (total parent)
Cdose_load = [None] * int(T // dt)  # ng/mL, dose_loading dose part of the parent
CGut  = [None] * int(T // dt)       # ng/mL, Gut transit compartment
C1855 = [None] * int(T // dt)       # ng/mL, OR-1855
C1896 = [None] * int(T // dt)       # ng/mL, OR-1896

CLevo[0] = C0_levo + C_dose_load  # ng/mL
CLevo[1] = C0_levo + C_dose_load  # ng/mL
Cdose_load[0] = C_dose_load       # ng/mL
Cdose_load[1] = C_dose_load       # ng/mL
CGut[0]  = 0  # ng/mL
CGut[1]  = 0  # ng/mL
C1855[0] = 0  # ng/mL
C1855[1] = 0  # ng/mL
C1896[0] = 0  # ng/mL
C1896[1] = 0  # ng/mL

Ce      = [None] * int(T // dt)      # ng/mL, active (plasma) drug concentration
Ce_eff  = [None] * int(T // dt)      # ng/mL, lagged effect-site concentration
R_sys   = [None] * int(T // dt)      # Pa.s/m^3, systemic resistance
SV      = [None] * int(T // dt)      # m^3, stroke volume
HR      = [None] * int(T // dt)      # Hz, heart rate
CO      = [None] * int(T // dt)      # m^3/s, cardiac output
MAP     = [None] * int(T // dt)      # Pa, mean arterial pressure
R_cereb = [None] * int(T // dt)      # Pa.s/m^3, cerebral resistance
V_cereb = [None] * int(T // dt)      # m/s, cerebral blood velocity

MAP[0]     = 0  
MAP[1]     = 0
HR[0]      = 0
HR[1]      = 0
V_cereb[0] = 0
V_cereb[1] = 0

############################# Main Loop #######################################

# t = 0 and t = 1, matching CLevo[0:2] -- Ce and Ce_eff are drug-naive, so
# both PD blocks sit at their nominal baseline
Ce[0] = 0  # ng/mL
Ce[1] = 0  # ng/mL
Ce_eff[0] = 0  # ng/mL
Ce_eff[1] = 0  # ng/mL
for idx in (0, 1):
    _, R_sys[idx], SV[idx], HR[idx], CO[idx], MAP[idx] = heart_block(Ce[idx], Ce_eff[idx])
    R_cereb[idx], V_cereb[idx] = cerebral_block(Ce_eff[idx], MAP[idx])

for t in range(1, int(T // dt - 1)):

    ### Levosimendan (ACTIVE) ###

    # dose_loading dose part: given at t = 0, cleared by first-order decay from the
    # start (the infusion loss L only acts on the infused drug)
    dCdose_load = -k0 * Cdose_load[t]
    Cdose_load[t + 1] = Cdose_load[t] + dCdose_load

    # During the 24 hour infusion, a constant loss L clears drug as it arrives,
    # so levosimendan rises slowly in a straight line. After 24 hours, the
    # infusion stops and the body clears it by first-order decay.
    # Form = cleared drug sent to the gut (OR-1855 route), scaled into the
    # metabolite volume.
    if t * dt < T_inf:
        dCLevo = R - L + dCdose_load
        Form = m1 * Vm_ratio * (L + k0 * Cdose_load[t])
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
    # Formed by gut metabolism, lost from acetylation (1896) and self decay
    dC1855 = Rel - kac * C1855[t]
    C1855[t + 1] = C1855[t] + dC1855

    ### OR-1896 (ACTIVE Metabolite) ###
    # Formed by acetylation and self decay
    dC1896 = kac * C1855[t] - k2 * C1896[t]
    C1896[t + 1] = C1896[t] + dC1896

    ### Active Drug Concentration (PK Output)
    Ce[t + 1] = f0 * CLevo[t + 1] + f2 * C1896[t + 1]

    ### Heart PD (Mass Transfer System) ###
    (Ce_eff[t + 1], R_sys[t + 1], SV[t + 1], HR[t + 1], CO[t + 1], MAP[t + 1]) = heart_block(Ce[t], Ce_eff[t])

    ### Cerebral Vasculature PD (Mass Transfer System) ###
    R_cereb[t + 1], V_cereb[t + 1] = cerebral_block(Ce_eff[t + 1], MAP[t + 1])


############################## Plot Figure ####################################

time = (np.arange(0, int(T // dt)) * dt) / 24 # days, x-axis in real time not step number

## Levosimendan Concentration 
plt.figure()
plt.plot(time, CLevo)
plt.ylabel("Levosimendan Concentration (ng/mL)")
plt.xlabel("Time (days)")
plt.title("Concentration of Levosimendan (Parent)")
plt.show()

## GUT Transition of OR1855
plt.figure()
plt.plot(time, CGut)
plt.ylabel("Gut Concentration (ng/mL)")
plt.xlabel("Time (days)")
plt.title("Gut Transit of OR-1855 metabolite")
plt.show()

## OR-1855 Concentration 
plt.figure()
plt.plot(time, C1855)
plt.ylabel("OR-1855 vs DATA (ng/mL)")
plt.xlabel("Time (days)")
plt.title("OR-1855 (Inactive Metabolite)")
plt.show()

## OR-1896 Concentration 
plt.figure()
plt.plot(time, C1896)
plt.ylabel("OR-1896 vs DATA (ng/mL)")
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
plt.ylabel("Blood Velocity (cm/s)")
plt.xlabel("Time (days)")
plt.title("Cerebral Blood Flow Velocity")
plt.show()

############################## References #####################################
# Declaration of AI use
# [0] Anthropic, "Claude, Opus 5.5," Claude, 22 October 2026 [Online]. https://claude.ai/new
#
# Provided Table sources from Mark project context
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
#
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
