##############################################################################
#
# BIOE3001: Quantitative Methods in Biomedical Engineering
# Tevyn Vergara | 46421201
#
# Code #3: Final Model (EPHYS) Patient Cure
#
# Code #1 model structure with the parameterised values from Code #2.
# Values changed by the parameterisation are marked (Parameterized)
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
import scipy.signal as sig_sci          # peak detection
import wfdb                         # wavefile database package (pip install wfdb)
from pathlib import Path            # consistent paths between mac/linux/windows computers

############################# EPHYS Data Path #################################

# Change path for marking/usage here please:
data_dir = Path(r"C:\BIOE3001 - FINAL\Project Resources\Patient Data (16_10)")

###############################################################################
###################### Section 1: Model Implementation #########################
###############################################################################

####################### Dose Scheduling and Time Steps ########################
## Time course: set in days, runs in hrs

# Run Mode Toggle
# TRUE = scheduling on, dosing schedule to maintain homeostasis levels
# FALSE = scheduling off, 1 dose for shape and quantity analysis

repeat_dosing = True                # BOOLEAN, CHANGE WHEN NEEDED

if repeat_dosing:
    T_days = 30                     # days, time course to show repeat dosing
else:
    T_days = 12                     # days, time course

# Plot Toggle
# TRUE = also show the raw and processed EPHYS data plots (Section 2)
# FALSE = treatment plots only (Section 3)

show_EPHYS_plots = True             # BOOLEAN, CHANGE WHEN NEEDED

dt = 0.1                            # hrs, time step
T = T_days * 24                     # hrs, time course

############################# Parameters: Patient #############################
# Patient-Specific Body parameters
BW = 80                             # kg, sample body weight from [1], [10]

# Unit conversion macros
MMHG_TO_PA = 133.322                # Pa per mmHg
ML_TO_M3 = 1e-6                     # m^3 per mL

# Patient-specific heart baselines
# HR0, MAP0 and v_cereb0 are set per patient in Section 3 from the EPHYS data
HR0 = None                          # Hz, baseline heart rate (Section 2)
SV0 = 77.0 * ML_TO_M3               # m^3, baseline SV. From [11] (not measured in EPHYS)
MAP0 = None                         # Pa, baseline MAP (Section 2)

# Patient-specific brain baselines
v_cereb0 = None                     # m/s, baseline MCA velocity (Section 2)
P_brain = 10.0 * MMHG_TO_PA         # Pa, intracranial pressure. From [12]
r_cereb = 0.00128                   # m, MCA M1 radius. From [15]

############################# Parameters: Dosing ##############################
# Dosing: 100 ng/kg/min IV for 24 hrs
# no loading dose (usual 50–200), From [16]

dose_load = 0                       # ng/kg, loading dose at t = 0
dose_sup = 100                      # ng/kg/min, supplementary dosing infusion rate
dose_inf = 24                       # hrs, infusion duration
dose_sch = 5                        # days, dose scheduling for repeated dosing

# Default regimen, passed to simulate() (Section 3 builds the patient regimens)
regimen_default = dict(dose_load=dose_load, dose_sup=dose_sup,
                       dose_inf=dose_inf, dose_sch=dose_sch)

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

# Dose scaling (as in Code 2), so any infusion rate can be used for the patient
# regimen. Per-regimen dose conversions are done in simulate()

dose_ref = 100                      # ng/kg/min, Table 1 dose that C_slope and C0_levo were measured at [1]
phi = C_slope / (dose_ref * 60 / V_kg) # fraction of infused drug retained in plasma, fixed drug property (DERIVED)

# Parameters: Gut Transit (delay before metabolites)
T_gut = 30                          # hrs, time when gut starts to release (FITTED FROM [1])
gut_rate = 65.9                     # hrs, gut release rate (Parameterized)
kg = (np.log(2) / gut_rate) * dt    # 1/hrs, gut release rate as half life (DERIVED)

# Parameters: OR-1855 Metabolite (Inactive)
m1 = 0.05                           # Formation Fraction into OR-1855 metabolite. From [13]
Vm_ratio = 0.5275                   # parent-to-metabolite volume ratio (Parameterized)

# Parameters: OR-1896 Metabolite (Active)
m2_half = 7.642                     # hrs, OR-1896 half-life (Parameterized)
k2 = (np.log(2) / m2_half) * dt     # 1/hrs, half-life decay (DERIVED)
f2 = 0.61                           # Free Fraction OR-1896. From [1], [4]

# Parameters: Acetylation (OR-1855 -> OR-1896) through NAT2 enzyme
ac_rate = 4.951                     # hrs, acetylation rate (Parameterized)
kac = (np.log(2)/ac_rate)*dt        # 1/hrs, acetylation as half life(DERIVED)

################### Parameters: Body (Pharmacodynamics) #######################
# PD is kept in SI, patient baselines are defined at top

# Shared EC50 for SV, HR, R_sys and R_cereb
EC50 = 0.5                          # ng/mL, shared EC50 (Fitted)

# Hill Kinetics: drug effect on stroke volume
dSVmax = 0.28                       # fraction, maximum SV increase from Levosimendan. From [14]

# Hill Kinetics: drug effect on heart rate
Emax_HR = 0.153                     # fraction, maximum HR increase from Levosimendan (Parameterized)

# Hill Kinetics: dilation of systemic and cerebral vessels
Emax_R = 0.3675                     # fractional max R_sys and R_cereb drop (Parameterized)

### Parameters: Brain Block (MAP, Ce_HR --> V_cereb)

A_cereb = np.pi * (r_cereb) ** 2    # m^2, MCA cross-section

# Effect delay for SV
keo_thalf = 13.85                   # hrs, compartment-effect time delay (Parameterized)
k_eo = (np.log(2) / keo_thalf) * dt # 1/hrs, pre-multiplied by dt

# Fast parent-weighted effect (Ce_HR) on HR and resistance (R_sys, R_cereb):
# Not explicit but needed from back calibration of Code #2 to better fit PD Table 3 data
# Values fitted but parameters supported by [3] and [18]
k_HR = 6                            # parent potency on HR and vessels (Fitted)
keo_HR_thalf = 6                    # hrs, fast effect delay for HR and vessels (Fitted)
k_eo_HR = (np.log(2)/keo_HR_thalf)*dt  # 1/hrs, pre-multiplied by dt

# Heart (Mass Transfer System)
#
# Takes effect-site drugs and runs three Hill Kinetic responses:
# Systemic resistance, stroke volume and heart rate (R_sys and HR use Ce_HR, SV uses Ce_eff)
# Outputs MAP --> Brain
#
def heart_block(Ce_eff_v, Ce_HR_v, HR0, R_sys0):
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
# Equations from [5].
def cerebral_block(Ce_val, MAP_v, R_cereb0):
    Cerebral_R = R_cereb0 * (1 - Emax_R * Ce_val / (EC50 + Ce_val))
    Blood_V = (MAP_v - P_brain) / (Cerebral_R * A_cereb)
    return Cerebral_R, Blood_V

# This function runs the model for one patient and one dosing regimen
#
# patient: HR0 (Hz), MAP0 (Pa), v_cereb0 (m/s)
# regimen: dose_load (ng/kg), dose_sup (ng/kg/min), dose_inf (hrs), dose_sch (days)

def simulate(patient, regimen):
    HR0, MAP0, v_cereb0 = patient["HR0"], patient["MAP0"], patient["v_cereb0"]
    dose_load, dose_sup = regimen["dose_load"], regimen["dose_sup"]
    dose_inf, dose_sch = regimen["dose_inf"], regimen["dose_sch"]

    # Patient baselines 
    CO0 = SV0 * HR0                     # m^3/s, baseline cardiac output
    R_sys0 = MAP0 / CO0                 # Pa.s/m^3, baseline systemic resistance
    CPP0 = MAP0 - P_brain               # Pa, baseline cerebral perfusion pressure
    Q_cereb0 = v_cereb0 * A_cereb       # m^3/s, baseline cerebral flow
    R_cereb0 = CPP0 / Q_cereb0          # Pa.s/m^3, baseline cerebral resistance

    # Repeat dosing for scheduling 
    if repeat_dosing:
        T_rep = dose_sch * 24           # hrs, interval between infusion starts
        N_inf = int(np.ceil(T / T_rep)) # number of infusions
    else:
        T_rep = T                       # hrs, single dose
        N_inf = 1                       # one infusion only

    # Time steps where repeat infusion starts
    start_steps = [int(round(k * T_rep / dt)) for k in range(1, N_inf)]

    # Dose conversions for main loop
    R_inf = (dose_sup * 60 / V_kg) * dt    # ng/mL per step, infusion input
    C0_inf = C0_levo / dose_ref * dose_sup # ng/mL, initial jump for this dose
    L_inf = (1 - phi) * R_inf              # ng/mL per step, loss during infusion
    C_dose_load = dose_load / V_kg         # ng/mL, loading dose jump at t = 0

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

    CLevo[0] = C0_inf + C_dose_load         # ng/mL, initial + loading dose
    Cdose_load[0] = C_dose_load             # ng/mL, just loading dose
    CGut[0]  = 0                            # ng/mL
    C1855[0] = 0                            # ng/mL
    C1896[0] = 0                            # ng/mL
    Ce[0] = f0 * CLevo[0] + f2 * C1896[0]   # ng/mL, free active drug
    Ce_eff[0] = 0                           # ng/mL
    Ce_HR[0] = 0                            # ng/mL

    # Patient baselines as initial conditions
    R_sys[0], SV[0], HR[0], CO[0], MAP[0] = heart_block(Ce_eff[0], Ce_HR[0], HR0, R_sys0)
    R_cereb[0], V_cereb[0] = cerebral_block(Ce_HR[0], MAP[0], R_cereb0)

    ############################# Main Loop #######################################

    for t in range(0, int(T // dt - 1)):

        # Levosimendan (ACTIVE)

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
            CLevo[t + 1] += C0_inf

        # Gut Transit Compartment
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

        # Effect Compartment
        Ce_eff[t + 1] = Ce_eff[t] + k_eo * (Ce[t] - Ce_eff[t])
        Ce_HR[t + 1] = Ce_HR[t] + k_eo_HR * (k_HR * f0 * CLevo[t] + f2 * C1896[t] - Ce_HR[t])

        # Heart PD 
        R_sys[t + 1], SV[t + 1], HR[t + 1], CO[t + 1], MAP[t + 1] = heart_block(Ce_eff[t + 1], Ce_HR[t + 1], HR0, R_sys0)

        # Brain PD
        R_cereb[t + 1], V_cereb[t + 1] = cerebral_block(Ce_HR[t + 1], MAP[t + 1], R_cereb0)
        
    # Builds time axis and returns parameter outputs for Section 3
    time = (np.arange(0, int(T // dt)) * dt) / 24   # days
    outputs = dict(CLevo=CLevo, C1896=C1896, HR=HR, MAP=MAP, V_cereb=V_cereb)
    return {"time": time, **{key: np.array(val) for key, val in outputs.items()}}

###############################################################################
###################### Section 2: Unpacking EPHYS Data ########################
###############################################################################
# Copied directly from Harriet and Dwayne's Implementation (Thank you)

############################# Load Patient Data ###############################
# Stroke patient = unhealthy baseline (initial values)
# No stroke patient = target

# HYPERTENSION: Stroke vs No Stroke
S0175 = wfdb.rdrecord(data_dir / "s0175-tcd-0")  # S0175 - 64yo, Stroke, Hypertension (Baseline)
S0157 = wfdb.rdrecord(data_dir / "s0157-tcd-0")  # S0157 - 67yo, No Stroke, Hypertension (Target)

# NORMAL TENSION: Stroke vs No Stroke
S0199 = wfdb.rdrecord(data_dir / "s0199-tcd-0")  # S0199 - 77yo, Stroke, Normal Tension (Baseline)
S0200 = wfdb.rdrecord(data_dir / "s0200-tcd-0")  # S0200 - 70yo, No Stroke, Normal Tension (Target)


patients = {"S0175 (Hypertension, Stroke)": S0175,
            "S0157 (Hypertension, No Stroke)": S0157,
            "S0199 (Normal Tension, Stroke)": S0199,
            "S0200 (Normal Tension, No Stroke)": S0200}

# Patient body weights, directly from dataset [20]
BW_pt = {"S0175 (Hypertension, Stroke)": 77,          # kg
         "S0157 (Hypertension, No Stroke)": 74,       # kg
         "S0199 (Normal Tension, Stroke)": 87,        # kg
         "S0200 (Normal Tension, No Stroke)": 76}     # kg

# Patient ages, directly from dataset [20]
Age_pt = {"S0175 (Hypertension, Stroke)": 64,         # years
          "S0157 (Hypertension, No Stroke)": 67,      # years
          "S0199 (Normal Tension, Stroke)": 77,       # years
          "S0200 (Normal Tension, No Stroke)": 70}    # years

########################## Extract Signals of Interest ########################
# Works with the wfdb data directly (from [19])
# Signals picked by name, as channel order differs between patients

channels = ["cerebral_blood_velocity_right",    # cm/s, RMCA velocity
            "cerebral_blood_velocity_left",     # cm/s, LMCA velocity
            "abp"]                              # mmHg, blood pressure
labels = ["RMCA_Velocity (cm/s)", "LMCA_Velocity (cm/s)",
          "Blood_Pressure (mmHg)"]

def unpack(Pt):
    Fs = Pt.fs                              # Hz, sampling rate
    dT = 1 / Fs                             # s, time between samples
    time_vector = np.arange(Pt.sig_len) * dT   # s, recording time
    signals = {ch: Pt.p_signal[:, Pt.sig_name.index(ch)] for ch in channels + ["ecg"]}
    return Fs, dT, time_vector, signals

EPHYS = {name: unpack(Pt) for name, Pt in patients.items()}

############################## Plot Raw Data ##################################
timeseq = [30, 35]                          # s, arbitrary 5 s window (from [19])

if show_EPHYS_plots:
    for name, (Fs, dT, time_vector, signals) in EPHYS.items():
        i0, i1 = timeseq[0] * Fs, timeseq[1] * Fs
        plt.figure()
        for ch, lab in zip(channels, labels):
            plt.plot(time_vector[i0:i1], signals[ch][i0:i1], label=lab)
        plt.xlabel("Time (s)")
        plt.ylabel("Raw Signal")
        plt.title(f"Raw EPHYS Data: {name}")
        plt.legend()
        plt.show()

########################### Processing EPHYS Data #############################
#
# 1. Cut each recording to its resting window using the marker files
# 2. Median filter MCA velocity (and BP for beat detection) to remove single-sample spikes
# 3. Find each heart beat from blood pressure peaks (systole) and troughs (diastole)
# 4. HR from band-passed ECG R-peaks, SBP/DBP from peaks/troughs, MAP over whole beats
# 5. MCA velocity per beat, dropout beats removed, L and R averaged
#
# Dropouts: MCA velocity <= 0 is lost Doppler signal (probe movement, poor
# temporal window), not real flow. In a living brain MCA flow stays forward
# all beat, so any beat with a dropout is masked out instead of averaged in.
# Whole beats are removed, as removing single samples would bias the mean up
#
# Implementation based on [19]

prom_BP = 10                        # mmHg, min prominence of BP peaks/troughs (from [20])
beat_min = 0.33                     # s, min time between detected peaks (180 bpm)
HR_min = 40                         # bpm, lowest realistic resting HR
HR_max = 120                        # bpm, highest realistic resting HR

# Median filter: replaces each sample with the median of its neighbours, so a
# single-sample spike is removed while the beat shape (peaks, upstroke) is kept
med_window = 0.1                    # s, filter window (much shorter than a beat)

# ECG band-pass 
fc_HP = 3                            # Hz, HP cut off (baseline wander)
fc_LP = 20                           # Hz, LP cut off (muscle/movement noise and data sampled at 50Hz)
order = 4                           # Butterworth filter order
R_prom = 0.5                        # R-peak prominence, fraction of 99th percentile
                                    # of filtered ECG (ECG amplitude varies per patient)
LMCA = "cerebral_blood_velocity_left"
RMCA = "cerebral_blood_velocity_right"

# S0199 uses LMCA only: 99% of RMCA beats are dropouts
LMCA_only = ["S0199 (Normal Tension, Stroke)"]

# Butterworth filters for bandpass
def butter_highpass_filter(data, cutoff, fs, f_nyq, order):
    b, a = sig_sci.butter(order, cutoff / f_nyq, btype='high', analog=False)
    return sig_sci.filtfilt(b, a, data)

def butter_lowpass_filter(data, cutoff, fs, f_nyq, order):
    b, a = sig_sci.butter(order, cutoff / f_nyq, btype='low', analog=False)
    return sig_sci.filtfilt(b, a, data)

# Resting window: Keep data from baseline toto the first marker that is NOT baseline
# Remove respiratory noise
def rest_window(Pt):
    markers = wfdb.rdann(str(data_dir / Pt.record_name), "marker")
    k0 = markers.aux_note.index("BASELINE")                   
    k1 = next(k for k in range(k0, len(markers.aux_note))
              if markers.aux_note[k] != "BASELINE")           
    return markers.sample[k0], markers.sample[k1]

# Function to take patient's loaded record and returns the baseline values for
# patient curing
def process(name, Pt):
    # Unpack patient signal from EPHYS data loading
    Fs, dT, time_vector, signals = EPHYS[name]
    
    # Get their resting window from marker file
    i0, i1 = rest_window(Pt)
    t = time_vector[i0:i1]
    k_med = int(med_window * Fs) // 2 * 2 + 1      # samples, odd window (5 at 50 Hz)
    BP = signals["abp"][i0:i1]                     # raw BP, used for all BP values
    BP_f = sig_sci.medfilt(BP, k_med)              # filtered BP, used to find all beats
    # (median filter clips the narrow systolic peaks at 50 Hz, SBP is read from raw BP)

    # Systolic peaks and diastolic troughs
    # Need to be at least 10 mmHg and 0.33s apart to be legit
    systole_ind, _ = sig_sci.find_peaks(BP_f, prominence=(prom_BP, None), distance=beat_min * Fs)
    diastole_ind, _ = sig_sci.find_peaks(-BP_f, prominence=(prom_BP, None), distance=beat_min * Fs)

    # Heart rate from band-passed ECG R-peaks
    # Need to be at least 0.33s apart to be legit
    f_nyq = Fs / 2                                 # Hz, Nyquist frequency
    ECG_HPF = butter_highpass_filter(signals["ecg"][i0:i1], fc_HP, Fs, f_nyq, order)
    ECG_HPfc_LPF = butter_lowpass_filter(ECG_HPF, fc_LP, Fs, f_nyq, order)
    r_peaks, _ = sig_sci.find_peaks(ECG_HPfc_LPF, distance=beat_min * Fs, prominence=(R_prom * np.percentile(ECG_HPfc_LPF, 99), None))
    # Time between consecutive R-peaks. Intervals outside [40,120] BPM filtered out
    rr = np.diff(t[r_peaks])        # s, R-wave intervals
    rr = rr[(rr >= 60 / HR_max) & (rr <= 60 / HR_min)]   # realistic resting HR only
    HR = 60 / np.mean(rr)           # bpm

    # Blood pressure (raw BP at detected indices)
    SBP = np.mean(BP[systole_ind])  # mmHg
    DBP = np.mean(BP[diastole_ind]) # mmHg
    MAP = np.mean(BP[systole_ind[0]:systole_ind[-1]])  # mmHg
    PP = SBP - DBP                  # mmHg, pulse pressure

    # MCA velocity per beat, beats with dropouts masked
    # Only S0199 has LMCA only
    sides = [LMCA] if name in LMCA_only else [LMCA, RMCA]
    v_side, dropped, v_masked = [], [], {}
    beats = list(zip(systole_ind[:-1], systole_ind[1:]))
    
    # Dropout Detection and Masking
    # Check each beat, only true beats if lowest velocity > 0
    # Physiologically, anything below that is False 
    # Only append true beats to MCA velocity value
    for ch in sides:
        v = sig_sci.medfilt(signals[ch][i0:i1], k_med)    # spikes removed
        clean = [v[a:b].min() > 0 for a, b in beats]
        v_side.append(np.mean([np.mean(v[a:b]) for (a, b), ok in zip(beats, clean) if ok]))
        dropped.append(100 * (1 - np.mean(clean)))
        for (a, b), ok in zip(beats, clean):
            # If false beat detected, mask them for the plotting
            if not ok:
                v[a:b] = np.nan     # masked beat (shown as gap in plot)
        v_masked[ch] = v
    v_MCA = np.mean(v_side)         # cm/s, L and R average (or LMCA only)

    return dict(Age=Age_pt[name], BW=BW_pt[name], HR=HR, SBP=SBP, DBP=DBP, MAP=MAP, PP=PP, v_MCA=v_MCA,
                v_side=dict(zip(sides, v_side)), dropped=dict(zip(sides, dropped)),
                n_beats=len(beats), t=t, BP=BP, systole_ind=systole_ind,
                diastole_ind=diastole_ind, v_masked=v_masked)

Pt_values = {name: process(name, Pt) for name, Pt in patients.items()}

########################## Baseline and Target Values #########################
# Stroke patient = baseline (initial values), no stroke patient = target

pairs = {"Hypertension": ("S0175 (Hypertension, Stroke)", "S0157 (Hypertension, No Stroke)"),
         "Normal Tension": ("S0199 (Normal Tension, Stroke)", "S0200 (Normal Tension, No Stroke)")}

# Final filtered values per patient. Format to 1 DP
def fmt(x):                         # "-" = side not used (S0199 RMCA)
    return "-" if np.isnan(x) else f"{x:.1f}"

# Create the table for terminal output. Run through amount of patients loaded
cols = ["Age (y)", "BW (kg)", "HR (bpm)", "MAP (mmHg)", "LMCA (cm/s)", "RMCA (cm/s)",
        "Averaged MCA (cm/s)", "Dropped Beats (L/R %)"]
w_pt = max(len(n) for n in Pt_values) + 1  
w_col = [len(c) + 2 for c in cols]  

# lines match the table width
header = f"{'Patient':{w_pt}}|" + "".join(f"{c:^{n}}" for c, n in zip(cols, w_col))
width = len(header)                 

# Table borders and sectioning
print("=" * width)
print(f"{'EPHYS Data Processing':^{width}}")
print("=" * width)
print(header)
print("-" * width)

# Table formatting for each side's values
for name, P in Pt_values.items():
    L = P["v_side"].get(LMCA, np.nan)
    R = P["v_side"].get(RMCA, np.nan)
    dL = P["dropped"].get(LMCA, np.nan)
    dR = P["dropped"].get(RMCA, np.nan)
    drop_txt = f"{dL:.0f} / " + ("-" if np.isnan(dR) else f"{dR:.0f}")
    row = [f"{P['Age']}", f"{P['BW']}", f"{P['HR']:.1f}", f"{P['MAP']:.1f}",
           fmt(L), fmt(R), f"{P['v_MCA']:.1f}", drop_txt]
    print(f"{name:{w_pt}}|" + "".join(f"{v:^{n}}" for v, n in zip(row, w_col)))

# Set up stroke values as baseline and non-stroke values as target for personal treatment
baseline = {group: Pt_values[stroke] for group, (stroke, healthy) in pairs.items()}
target = {group: Pt_values[healthy] for group, (stroke, healthy) in pairs.items()}

########################### Plot Processed Data ###############################
# 10 s of resting data: detected beats on BP and MCA velocity with masked beats

t_show = 10                         # s 

if show_EPHYS_plots:
    for name, P in Pt_values.items():
        
        # n. of samples in 10s window
        n = int(t_show * EPHYS[name][0])
        
        # Only sistolic peak and diastolic troughs
        s = P["systole_ind"][P["systole_ind"] < n]
        d = P["diastole_ind"][P["diastole_ind"] < n]
        
        # time axis shifted to start at 0
        t = P["t"][:n] - P["t"][0]
        
        fig, ax = plt.subplots(2, 1, sharex=True)
        
        # Blood Pressure and MAP
        ax[0].plot(t, P["BP"][:n], label="Blood Pressure")
        ax[0].plot(t[s], P["BP"][s], "x", label="Systole (peaks)")
        ax[0].plot(t[d], P["BP"][d], "rx", label="Diastole (troughs)")
        ax[0].axhline(P["MAP"], color="grey", linestyle="--", label=f"MAP = {P['MAP']:.1f} mmHg")
        ax[0].set_ylabel("Pressure (mmHg)")
        ax[0].legend(loc="upper right", fontsize=8)
        
        # LMCA, RMCA and VMCA
        for ch, v in P["v_masked"].items():
            ax[1].plot(t, v[:n], label=("LMCA" if ch == LMCA else "RMCA") + " (filtered, dropouts masked)")
        ax[1].axhline(P["v_MCA"], color="grey", linestyle="--", label=f"Mean = {P['v_MCA']:.1f} cm/s")
        ax[1].set_ylabel("MCA Velocity (cm/s)")
        ax[1].set_xlabel("Time from start of rest (s)")
        ax[1].legend(loc="upper right", fontsize=8)
        fig.suptitle(f"Processed EPHYS Data: {name}")
        plt.show()

###############################################################################
###################### Section 3: Curing the Patients #########################
###############################################################################

# Stroke patient baselines from Section 2, converted to model (SI) units

def patient_inputs(P):
    return dict(HR0=P["HR"] / 60,                   # Hz, from bpm
                MAP0=P["MAP"] * MMHG_TO_PA,         # Pa, from mmHg
                v_cereb0=P["v_MCA"] / 100)          # m/s, from cm/s

patient_in = {group: patient_inputs(P) for group, P in baseline.items()}

########################### Treatment Regimens ###############################
# Same per-kg regimen for every patient (similar age and weight), so only the
# patient baselines differ. Each infusion repeats every dose_sch days
#
# Table 1: 100 ng/kg/min for 24 hrs, no loading dose [1]
# Table 3: 12 ug/kg loading dose, then 200 ng/kg/min for 24 hrs [3]

# Regimen Toggles
# TRUE = run and plot this regimen for both patient groups (2 figures each)
# FALSE = skip this regimen

regimen_T1 = True                   # BOOLEAN, Table 1 regimen [1]
regimen_T3 = True                   # BOOLEAN, Table 3 regimen [3]

# Pre-stored regimens: (toggle, regimen)
all_regimens = {"Table 1": (regimen_T1, dict(dose_load=0, dose_sup=100, dose_inf=24, dose_sch=5)),         
                "Table 3": (regimen_T3, dict(dose_load=12 * 1000, dose_sup=200, dose_inf=24, dose_sch=5))}         

# Only the regimens toggled on are run and plotted
regimens = {name: reg for name, (on, reg) in all_regimens.items() if on}

def regimen_text(reg):
    # Short regimen description for plot titles (ug/kg/min, ug/kg)
    load = f"{reg['dose_load'] / 1000:g} µg/kg load + " if reg["dose_load"] else "no load, "
    return (f"{load}{reg['dose_sup'] / 1000:g} µg/kg/min x {reg['dose_inf']:g} hrs, "
            f"every {reg['dose_sch']:g} days")

# Model run for each regimen and stroke patient: results[regimen][group]
results = {name: {group: simulate(patient_in[group], reg) for group in patient_in}
           for name, reg in regimens.items()}

# Reference runs with no drug: stroke patient (untreated) and healthy patient (target)
regimen_none = dict(dose_load=0, dose_sup=0, dose_inf=dose_inf, dose_sch=dose_sch)
untreated = {group: simulate(patient_in[group], regimen_none) for group in patient_in}
healthy_in = {group: patient_inputs(P) for group, P in target.items()}
healthy_run = {group: simulate(healthy_in[group], regimen_none) for group in healthy_in}

############################### Safety Limits #################################
# Same limits for every patient (closest match to our stroke population)
#
# HR: rise limit over the first 24 hrs, stopping rule in prior stroke/TIA patients [2]
# MAP: floor from the DBP >= 60 mmHg warning, using MAP = DBP + PP/3 [21]
# Loading dose: only if baseline SBP >= 90 mmHg [22]
# v_MCA: absolute peak (vasospasm range) [23] and rate of rise [24]
# Dose: infusion rate range [16], [21]

dHR_lim = 15                        # bpm, max HR rise from baseline in the first 24 hrs [2]
DBP_lim = 60                        # mmHg, DBP floor, MAP floor = DBP_lim + PP0/3 [21]
SBP_load = 90                       # mmHg, min baseline SBP for a loading dose [22]
v_lim = 120                         # cm/s, absolute MCA velocity peak [23]
dv_lim = 50                         # cm/s, max MCA velocity rise within 24 hrs [24]
dose_min, dose_max = 50, 200        # ng/kg/min, infusion rate range (0.05-0.2 ug/kg/min) [16], [21]

def rise_24h(x):
    # Largest rise within any 24 hr window (each hour vs the max over the next 24 hrs)
    n24 = int(24 / dt)
    return max(x[i:i + n24].max() - x[i] for i in range(0, len(x) - n24, int(1 / dt)))

def safety(R, P, regimen):
    # Model outputs in clinical units, checked against each limit (True = pass)
    HR = R["HR"] * 60                   # bpm
    MAP = R["MAP"] / MMHG_TO_PA         # mmHg
    v = R["V_cereb"] * 100              # cm/s
    MAP_floor = DBP_lim + P["PP"] / 3   # mmHg, patient-specific floor
    n24 = int(24 / dt)                  # steps in the first 24 hrs
    dHR = HR[:n24].max() - HR[0]        # bpm, HR rise in the first 24 hrs
    checks = {"HR rise": dHR <= dHR_lim,
              "MAP floor": MAP.min() >= MAP_floor,
              "Loading dose": regimen["dose_load"] == 0 or P["SBP"] >= SBP_load,
              "v peak": v.max() <= v_lim,
              "v rise": rise_24h(v) <= dv_lim,
              "Dose rate": dose_min <= regimen["dose_sup"] <= dose_max}
    values = dict(HR_max=HR.max(), dHR=dHR, MAP_min=MAP.min(),
                  MAP_floor=MAP_floor, v_peak=v.max(), dv24=rise_24h(v))
    return checks, values

safe = {name: {group: safety(results[name][group], baseline[group], regimens[name])
               for group in patient_in} for name in regimens}

########################### Plot Treatment Results ############################
# One figure per regimen and patient group: effective drug, HR, MAP, MCA velocity
# Treated stroke patient (green), untreated stroke patient (purple),
# healthy patient (blue), limits (red dotted)

for name, reg in regimens.items():
    for group, R in results[name].items():
        P = baseline[group]
        stroke, healthy = pairs[group]
        stroke_id, healthy_id = stroke.split()[0], healthy.split()[0]
        t_d = R["time"]                     # days
        HR = R["HR"] * 60                   # bpm
        MAP = R["MAP"] / MMHG_TO_PA         # mmHg
        v = R["V_cereb"] * 100              # cm/s

        # Treated run first, then the reference runs (no drug), same units
        runs = [(R, "green", f"{stroke_id} treated"),
                (untreated[group], "purple", f"{stroke_id} untreated (stroke)"),
                (healthy_run[group], "blue", f"{healthy_id} (healthy)")]

        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(12, 8))

        # Effective drug: Levosimendan (parent) and OR-1896 (active metabolite)
        ax[0, 0].plot(t_d, R["CLevo"], label="Levosimendan")
        ax[0, 0].plot(t_d, R["C1896"], label="OR-1896")
        ax[0, 0].set_ylabel("Plasma Concentration (ng/mL)")
        ax[0, 0].set_title("Effective Drug")
        ax[0, 0].legend(loc="upper right", fontsize=8)

        # Heart rate with rise limit [2]
        for X, c, lab in runs:
            ax[0, 1].plot(t_d, X["HR"] * 60, color=c, label=lab)
        ax[0, 1].axhline(HR[0] + dHR_lim, color="red", linestyle=":",
                         label=f"Rise limit, first 24 hrs (baseline + {dHR_lim} bpm)")
        ax[0, 1].set_ylabel("Heart Rate (bpm)")
        ax[0, 1].set_title("Heart Rate")
        ax[0, 1].legend(loc="best", fontsize=8)

        # MAP with patient-specific floor [21]
        MAP_floor = DBP_lim + P["PP"] / 3
        for X, c, lab in runs:
            ax[1, 0].plot(t_d, X["MAP"] / MMHG_TO_PA, color=c, label=lab)
        ax[1, 0].axhline(MAP_floor, color="red", linestyle=":",
                         label=f"MAP floor ({MAP_floor:.1f} mmHg)")
        ax[1, 0].set_ylabel("MAP (mmHg)")
        ax[1, 0].set_xlabel("Time (days)")
        ax[1, 0].set_title("Mean Arterial Pressure")
        ax[1, 0].legend(loc="best", fontsize=8)

        # MCA velocity with absolute and rate of rise limits [23], [24]
        v_rate = np.minimum(v[0] + dv_lim * t_d, v_lim)   # max rise of 50 cm/s per day from baseline
        for X, c, lab in runs:
            ax[1, 1].plot(t_d, X["V_cereb"] * 100, color=c, label=lab)
        ax[1, 1].axhline(v_lim, color="red", linestyle=":", label=f"Absolute limit ({v_lim} cm/s)")
        ax[1, 1].plot(t_d, v_rate, color="red", linestyle=":", alpha=0.5,
                      label=f"Rate of rise limit ({dv_lim} cm/s per 24 hrs)")
        ax[1, 1].set_ylabel("MCA Velocity (cm/s)")
        ax[1, 1].set_xlabel("Time (days)")
        ax[1, 1].set_title("Cerebral Blood Velocity")
        ax[1, 1].legend(loc="center right", fontsize=8)

        fig.suptitle(f"{name} Regimen ({regimen_text(reg)}): {group}, "
                     f"{stroke_id} (Stroke) vs {healthy_id} (Target)")
        fig.tight_layout()
        plt.show()

# NOTES: At least 2 dosing regimens
# Cure at least 4 patients
#
# Hypertension: Stroke vs No Stroke
# Normal tension: Stroke vs No Stroke

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
# [18] {Lilleberg, 2007}
# [19] {Mann, 2026}
# [20] {Novak, 2010}
# [21] {Orion Pharma, 2025} Simdax data sheet
# [22] {SESLHD, 2024} Levosimendan medicine guideline
# [23] {Lindegaard, 1989}
# [24] {Grosset, 1992}
