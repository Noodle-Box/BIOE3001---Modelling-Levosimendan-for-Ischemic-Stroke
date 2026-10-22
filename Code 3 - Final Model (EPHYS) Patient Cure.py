##############################################################################
#
# BIOE3001: Quantitative Methods in Biomedical Engineering
# Tevyn Vergara | 46421201
#
# Main Code for Levosimendan Model for Ischemic Stroke
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
## Time course
dt = 0.1  # hrs
T = 300  # hrs, time course

###################### Parameters: Patient & Dosing ###########################

# Dosing regimen: 0.1 ug/kg/min (100 ng/kg/min) constant IV infusion for 24
# hrs, no loading bolus. From [1].

BW = 80                             # kg, sample body weight
dose = 100                          # ng/kg/min, dosing infusion rate
T_inf = 24                          # hrs, infusion duration
C_peak = 59                         # ng/mL, max plasma concentration of infusion

# Infusion conversions for main loop
Rate_ref = dose * BW * 60           # ng/hr infusion rate whole body rate

# Infusion Rate going in during 24-hour infusion regimen
levo_half = 0.9                                    # hrs, Levosimendan half-life, from [1]
V = Rate_ref / ((np.log(2) / levo_half) * C_peak)  # mL, volume distribution (DERIVED)
R = (Rate_ref / V) * dt                            # ng/mL, added per step

################### Parameters: Drug (Pharmacokinetics) #######################

# Parameters: Levosimendan (Active)
k0 = (np.log(2) / levo_half) * dt   # 1/Hrs, half-life decay
f0 = 0.05                           # Free Fraction Levosimendan (FITTED)

# Parameters: Gut Transit Compartment (Time Delay between parent and metabolites)
T_gut = 30                          # hrs, time when gut starts to release (FITTED FROM [1])
gut_rate = 66                       # hrs, gut release rate (FITTED)
kg = (np.log(2) / gut_rate) * dt    # 1/hrs, gut release rate (DERIVED)

# Parameters: OR-1855 Metabolite (Inactive)
m1 = 0.0268                         # Formation Fraction from Levo (FITTED)

# Parameters: OR-1896 Metabolite (Active)
m2_half = 7.7                       # Hrs, OR-1896 half-life (FITTED)
k2 = (np.log(2) / m2_half) * dt     # 1/Hrs, half-life decay (DERIVED)
f2 = 0.61                           # Free Fraction OR-1896. From [4]

# Parameters: Acetylation (OR-1855 -> OR-1896) through NAT2 enzyme
ac_rate = 5                         # Hrs, Acetylation rate (FITTED)
kac = (np.log(2)/ac_rate)*dt        # 1/Hrs, acetylation (DERIVED)

################### Parameters: Body (Pharmacodynamics) #######################

# PD is kept in SI (Pa, m^3, Pa.s/m^3, m/s, Hz)

# Conversion macros
MMHG_TO_PA = 133.322                # Pa per mmHg
ML_TO_M3 = 1e-6                     # m^3 per mL

# Hill Kinetics for drug binding to affect stroke volume
SV0 = 70.0 * ML_TO_M3               # m^3, baseline SV. From [7]
dSVmax = 0.167                      # fractional max SV rise. From [3]
EC50_SV = 0.497                     # ng/mL (FITTED)

# Hill Kinetics for drug binding to affect heart rate [6]
HR0 = 70.0 / 60.0                   # Hz, baseline heart rate
Emax_HR = 0.171                     # fractional max HR rise. From [2][3]
EC50_HR = 0.78                      # ng/mL, direct Hill term (FITTED)

# Hill Kinetics for drug binding to dilating system "body" vasculature
Emax_R = 0.358                      # fractional max R_sys drop (FITTED)
EC50_sys = 0.498                    # ng/mL (FITTED)

# Derivation for baseline systemic resistance
# (how hard blood is pushed through vessels)
# Resistance = (pressure / flow), both baseline
MAP0 = 93.0 * MMHG_TO_PA            # Pa, Baseline MAP. From [8]
CO0 = SV0 * HR0                     # m^3/s, baseline cardiac output (DERIVED)
R_sys0 = MAP0 / CO0                 # Pa.s/m^3, baseline systemic resistance (DERIVED)

### Parameters: Brain Block (MAP, Ce_eff --> V_cereb)

P_brain = 10.0 * MMHG_TO_PA         # Pa, intracranial pressure. From [9]
r_cereb = 0.0015                    # m, MCA radius. From [5]
A_cereb = np.pi * (r_cereb) ** 2    # m^2, MCA cross-section

# Derivation for baseline cerebral resistance, similar to R_sys
# Resistance = pressure / flow, both baseline

v_cereb0 = 0.48                     # m/s, baseline MCA velocity. From [2]
CPP0 = MAP0 - P_brain               # Pa, baseline cerebral perfusion (DERIVED)
Q_cereb0 = v_cereb0 * A_cereb       # m^3/s, baseline cerebral flow (DERIVED)
R_cereb0 = CPP0 / Q_cereb0          # Pa.s/m^3, baseline cerebral resistance (DERIVED)

# Hill kinetics for drug binding to cerebral vasculature resistance
Emax_cereb = 0.405                  # fractional max R_cereb drop (FITTED)
EC50_cereb = 0.5                    # ng/mL (FITTED)

# Delay between plasma Ce and drug effect at tissue
keo_thalf = 6.0                     # hrs, compartment-effect time delay (FITTED)
k_eo = (np.log(2) / keo_thalf) * dt # 1/hrs, pre-multiplied by dt


### Heart (Mass Transfer System) ###
#
# Takes effective drug from dose and lags it to Ce_eff
# Then, runs three Hill Kinetic responses:
# System resistance, stroke volume and heart rate.
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

CLevo = [None] * int(T // dt)       # ng/mL, Levosimendan
CGut  = [None] * int(T // dt)       # ng/mL, Gut transit compartment
C1855 = [None] * int(T // dt)       # ng/mL, OR-1855
C1896 = [None] * int(T // dt)       # ng/mL, OR-1896

CLevo[0] = 0  # ng/mL
CLevo[1] = 0  # ng/mL
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

    # Constant infusion for the first 24 hours, so infusion and elimination run
    # together and levosimendan plateaus. After 24 hours, decay only.
    if t * dt < T_inf:
        dCLevo = R - k0 * CLevo[t]
    else:
        dCLevo = -k0 * CLevo[t]
    CLevo[t + 1] = CLevo[t] + dCLevo

    ### Gut Transit Compartment (time capture) ###

    if t * dt < T_gut:
        Rel = 0
    else:
        Rel = kg * CGut[t]

    dCGut = m1 * k0 * CLevo[t] - Rel
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

    ### Heart (Mass Transfer System) ###
    (
        Ce_eff[t + 1],
        R_sys[t + 1],
        SV[t + 1],
        HR[t + 1],
        CO[t + 1],
        MAP[t + 1],
    ) = heart_block(Ce[t], Ce_eff[t])

    ### Cerebral Vasculature (Mass Transfer System) ###
    R_cereb[t + 1], V_cereb[t + 1] = cerebral_block(Ce_eff[t + 1], MAP[t + 1])


############################## Plot Figure ####################################

time = (np.arange(0, int(T // dt)) * dt) # hrs, x-axis in real time not step number

## Clinical data of parent drug and metabolites, from Table 1 data
t_data = [24, 25, 26, 28, 30, 36, 37, 38, 40, 44, 48, 60, 72, 96, 144, 192, 240]
levo_data = [59, 22.4, 10.4, 4.1, 1.12, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
m1_data = [0,0,0,0,0.05,0.8, 0.9, 1.05, 1.2, 1.55, 2.1, 1.9, 1.95, 1.5, 0.7, 0.35, 0.15]
m2_data = [0,0,0,0,0.1,0.6,0.7,0.85,1.1,1.75,2.35,2.4,2.35,2.05,1.2,0.55,0.3]

## Levosimendan against clinical data [1]
plt.figure()
plt.plot(time, CLevo)
plt.plot(t_data, levo_data, "o")
plt.ylabel("Levosimendan Concentration (ng/mL)")
plt.xlabel("Time (hrs)")
plt.title("Levosimendan (Parent) vs Clinical Data")
plt.show()

## GUT Transition of OR1855
plt.figure()
plt.plot(time, CGut)
plt.ylabel("Gut Concentration (ng/mL)")
plt.xlabel("Time (hrs)")
plt.title("Gut Transit of OR-1855 metabolite (Time Compartment)")
plt.show()

## OR-1855 against the clinical data [1]
plt.figure()
plt.plot(time, C1855)
plt.plot(t_data, m1_data, "o")
plt.ylabel("OR-1855 vs DATA (ng/mL)")
plt.xlabel("Time (hrs)")
plt.title("OR-1855 (Inactive Metabolite) vs Clinical Data")
plt.show()

## OR-1896 against the clinical data [1]
plt.figure()
plt.plot(time, C1896)
plt.plot(t_data, m2_data, "o")
plt.ylabel("OR-1896 vs DATA (ng/mL)")
plt.xlabel("Time (hrs)")
plt.title("OR-1896 (Active Metabolite) vs Clinical Data")
plt.show()

## Pharmacodynamic outputs -- converted from SI to clinically readable units
## just for these plots (the model itself stays in SI throughout)
MAP_mmHg = np.array(MAP) / MMHG_TO_PA
HR_bpm = np.array(HR) * 60
V_cereb_cms = np.array(V_cereb) * 100

plt.figure()
plt.plot(time, MAP_mmHg)
plt.ylabel("MAP (mmHg)")
plt.xlabel("Time (hrs)")
plt.title("Mean Arterial Pressure (MAP)")
plt.show()

plt.figure()
plt.plot(time, HR_bpm)
plt.ylabel("HR (bpm)")
plt.xlabel("Time (hrs)")
plt.title("Heart Rate (HR)")
plt.show()

plt.figure()
plt.plot(time, V_cereb_cms)
plt.ylabel("Blood Velocity (cm/s)")
plt.xlabel("Time (hrs)")
plt.title("Cerebral Blood Flow Velocity")
plt.show()

########################## Loading Patient Data ###############################

# TBA

############################## References #####################################
# Declaration of AI use
# [0] Anthropic, "Claude, Sonnet 5," Claude, 18 September 2026 [Online]. https://claude.ai/new
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
