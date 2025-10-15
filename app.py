
# streamlit_synergy_app.py
# -----------------------------------------------------------
# Age-Equalized Hormone–Lifestyle Synergy Score (v3.1, "honest" build)
# Designed for Streamlit. Includes placeholder reference tables
# you should replace with lab- and device-specific data.
#
# Usage:
#   pip install streamlit
#   streamlit run streamlit_synergy_app.py
#
# NOTE: All reference tables below are *illustrative placeholders*.
# Replace with your lab's percentiles and device-specific references
# for real-world deployment.
# -----------------------------------------------------------

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Optional, Literal, Dict, Callable

import streamlit as st

# -----------------------------
# Utils
# -----------------------------
def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def nz(x: float, eps: float = 1e-9) -> float:
    return x if abs(x) > eps else eps

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def months_since(d: date) -> float:
    if d is None:
        return 999.0
    today = date.today()
    return (today.year - d.year) * 12 + (today.month - d.month) + (today.day - d.day) / 30.0

def interp_by_age(age: int, table: Dict[int, float]) -> float:
    """Piecewise-linear interpolation across an age->value dict.
    Keys should be ascending (e.g., 20,30,40,...). Clamps outside range."""
    keys = sorted(table.keys())
    if age <= keys[0]:
        return table[keys[0]]
    if age >= keys[-1]:
        return table[keys[-1]]
    for i in range(len(keys)-1):
        a0, a1 = keys[i], keys[i+1]
        if a0 <= age <= a1:
            v0, v1 = table[a0], table[a1]
            t = (age - a0) / (a1 - a0)
            return v0 + t * (v1 - v0)
    return table[keys[-1]]

# -----------------------------
# Placeholder reference functions
# Replace these with your real tables!
# -----------------------------

# Total testosterone (ng/dL): median (T50) and ~95th percentile (T95)
T50_M = {20: 650, 30: 600, 40: 550, 50: 500, 60: 450, 70: 400}
T95_M = {20: 900, 30: 850, 40: 800, 50: 750, 60: 700, 70: 650}

# For females, total T is much lower (illustrative only)
T50_F = {20: 35, 30: 32, 40: 30, 50: 28, 60: 26, 70: 24}
T95_F = {20: 70, 30: 65, 40: 60, 50: 55, 60: 50, 70: 45}

def T50_fn(age: int, sex: str) -> float:
    return interp_by_age(age, T50_M if sex == "male" else T50_F)

def T95_fn(age: int, sex: str) -> float:
    return interp_by_age(age, T95_M if sex == "male" else T95_F)

# HRV (RMSSD, ms) P10/P90 (very rough placeholders)
HRV_P10_M = {20: 35, 30: 32, 40: 28, 50: 25, 60: 22, 70: 20}
HRV_P90_M = {20: 90, 30: 85, 40: 80, 50: 75, 60: 70, 70: 65}

HRV_P10_F = {20: 38, 30: 35, 40: 31, 50: 28, 60: 25, 70: 22}
HRV_P90_F = {20: 95, 30: 90, 40: 85, 50: 80, 60: 75, 70: 70}

def HRV_P10_fn(age: int, sex: str) -> float:
    return interp_by_age(age, HRV_P10_M if sex == "male" else HRV_P10_F)

def HRV_P90_fn(age: int, sex: str) -> float:
    return interp_by_age(age, HRV_P90_M if sex == "male" else HRV_P90_F)

# Resting Heart Rate (bpm) P10/P90 (lower is better; placeholders)
RHR_P10_M = {20: 50, 30: 51, 40: 52, 50: 53, 60: 54, 70: 55}
RHR_P90_M = {20: 80, 30: 80, 40: 81, 50: 82, 60: 83, 70: 84}

RHR_P10_F = {20: 53, 30: 54, 40: 55, 50: 56, 60: 57, 70: 58}
RHR_P90_F = {20: 83, 30: 83, 40: 84, 50: 85, 60: 86, 70: 87}

def RHR_P10_fn(age: int, sex: str) -> float:
    return interp_by_age(age, RHR_P10_M if sex == "male" else RHR_P10_F)

def RHR_P90_fn(age: int, sex: str) -> float:
    return interp_by_age(age, RHR_P90_M if sex == "male" else RHR_P90_F)

# VO2max (mL/kg/min) P10/P90 (illustrative; varies a lot by dataset)
VO2_P10_M = {20: 35, 30: 33, 40: 31, 50: 28, 60: 25, 70: 22}
VO2_P90_M = {20: 60, 30: 58, 40: 55, 50: 52, 60: 48, 70: 44}

VO2_P10_F = {20: 27, 30: 26, 40: 24, 50: 22, 60: 20, 70: 18}
VO2_P90_F = {20: 50, 30: 48, 40: 45, 50: 42, 60: 38, 70: 34}

def VO2_P10_fn(age: int, sex: str) -> float:
    return interp_by_age(age, VO2_P10_M if sex == "male" else VO2_P10_F)

def VO2_P90_fn(age: int, sex: str) -> float:
    return interp_by_age(age, VO2_P90_M if sex == "male" else VO2_P90_F)

# Body-fat optimal center and half-width (tolerance) by age/sex (placeholders)
BF_C_M = {20: 14, 30: 15, 40: 16, 50: 17, 60: 18, 70: 19}
BF_W_M = {20: 10, 30: 10, 40: 11, 50: 12, 60: 12, 70: 12}
BF_C_F = {20: 24, 30: 25, 40: 26, 50: 27, 60: 28, 70: 29}
BF_W_F = {20: 10, 30: 10, 40: 11, 50: 12, 60: 12, 70: 12}

def BF_C_fn(age: int, sex: str) -> float:
    return interp_by_age(age, BF_C_M if sex == "male" else BF_C_F)

def BF_W_fn(age: int, sex: str) -> float:
    return interp_by_age(age, BF_W_M if sex == "male" else BF_W_F)

# -----------------------------
# Normalization helpers
# -----------------------------
def norm_higher_is_better(x: float, P10: float, P90: float, alpha: float = 0.7) -> float:
    # Map P10->0, P90->1 then apply concave utility (alpha<1) for diminishing returns
    p = clip((x - P10) / nz(P90 - P10), 0.0, 1.0)
    return p ** alpha

def norm_lower_is_better(x: float, P10: float, P90: float, alpha: float = 0.7) -> float:
    # Lower is better, so flip inside normalization
    p = clip((P90 - x) / nz(P90 - P10), 0.0, 1.0)
    return p ** alpha

def norm_u_shape(x: float, C: float, W: float) -> float:
    # Soft-U around optimum C with half-width W
    return 1.0 - clip(abs(x - C) / nz(W), 0.0, 1.0)

# -----------------------------
# Components
# -----------------------------
def hormone_component(total_T_ngdl: float, age: int, sex: str) -> float:
    """Age-fair hormone component with diminishing returns (sigmoid near median, gentle tail above ~95th)."""
    T50 = T50_fn(age, sex)
    T95 = T95_fn(age, sex)
    s = max((T95 - T50) / 2.0, 1e-6)
    H_base = sigmoid((total_T_ngdl - T50) / s)
    H_tail = clip((total_T_ngdl - T95) / max(T95, 1e-6), 0.0, 1.0)
    H = 0.95 * H_base + 0.05 * H_tail
    return clip(H, 0.0, 1.0)

def training_factor(days_per_week: Optional[float] = None,
                    minutes_per_week: Optional[int] = None,
                    device_strain_0_21: Optional[float] = None) -> float:
    if device_strain_0_21 is not None:
        return math.sqrt(clip(device_strain_0_21, 0.0, 21.0) / 16.0)
    if minutes_per_week is not None:
        return math.sqrt(clip(minutes_per_week, 0, 300) / 300.0)
    if days_per_week is not None:
        return math.sqrt(clip(days_per_week, 0.0, 6.0) / 6.0)
    return 0.5

def sleep_factor(hours_avg: Optional[float], quality_0_1: Optional[float] = None) -> float:
    if hours_avg is None and quality_0_1 is None:
        return 0.5
    base = clip((hours_avg or 7.5) / 8.5, 0.0, 1.0)
    if hours_avg is not None and hours_avg < 6.0:
        base = clip(base - clip((6.0 - hours_avg) / 2.0, 0.0, 0.4), 0.0, 1.0)
    if quality_0_1 is not None:
        base = 0.7 * base + 0.3 * clip(quality_0_1, 0.0, 1.0)
    return base

def body_comp_factor(age: int, sex: str,
                     bodyfat_pct: Optional[float] = None,
                     waist_cm: Optional[float] = None,
                     height_cm: Optional[float] = None,
                     bmi: Optional[float] = None,
                     fat_mass_kg: Optional[float] = None,
                     lean_mass_kg: Optional[float] = None,
                     measured: bool = False) -> float:
    # If fat+lean given, derive bodyfat% from them
    if fat_mass_kg is not None and lean_mass_kg is not None and (fat_mass_kg + lean_mass_kg) > 0:
        bodyfat_pct = 100.0 * fat_mass_kg / (fat_mass_kg + lean_mass_kg)
        measured = True

    # Priority: body-fat% -> WHtR -> BMI
    if bodyfat_pct is not None:
        C = BF_C_fn(age, sex)
        W = BF_W_fn(age, sex)
        bf = norm_u_shape(bodyfat_pct, C, W)
        kappa = 1.00 if measured else 0.95  # reserve perfect 1.0 for measured
        return clip(kappa * bf, 0.0, 1.0)

    if waist_cm is not None and height_cm is not None and height_cm > 0:
        whtr = waist_cm / height_cm
        bf = norm_u_shape(whtr, 0.45, 0.10)
        return clip(bf, 0.0, 1.0)

    if bmi is not None:
        bf = norm_u_shape(bmi, 23.0, 7.0)
        return clip(0.95 * bf, 0.0, 1.0)  # BMI-only → honesty cap

    return 0.5

def hrv_norm(hrv_ms: float, age: int, sex: str, alpha: float = 0.7) -> float:
    return norm_higher_is_better(hrv_ms, HRV_P10_fn(age, sex), HRV_P90_fn(age, sex), alpha=alpha)

def rhr_norm(rhr_bpm: float, age: int, sex: str, alpha: float = 0.7) -> float:
    return norm_lower_is_better(rhr_bpm, RHR_P10_fn(age, sex), RHR_P90_fn(age, sex), alpha=alpha)

def vo2_norm(vo2: float, age: int, sex: str, alpha: float = 0.7) -> float:
    return norm_higher_is_better(vo2, VO2_P10_fn(age, sex), VO2_P90_fn(age, sex), alpha=alpha)

def recovery_anchor(hrv_f: Optional[float], rhr_f: Optional[float], vo2_f: Optional[float]) -> Optional[float]:
    if hrv_f is not None and rhr_f is not None and vo2_f is not None:
        return (hrv_f * rhr_f * vo2_f) ** (1.0 / 3.0)
    if hrv_f is not None and rhr_f is not None:
        return (hrv_f * rhr_f) ** 0.5
    return hrv_f if hrv_f is not None else rhr_f if rhr_f is not None else vo2_f

def lifestyle_component(age: int, sex: str,
                        # training
                        days_per_week: Optional[float],
                        minutes_per_week: Optional[int],
                        device_strain_0_21: Optional[float],
                        # sleep
                        sleep_hours_avg: Optional[float],
                        sleep_quality_0_1: Optional[float],
                        # body comp
                        bodyfat_pct: Optional[float],
                        waist_cm: Optional[float],
                        height_cm: Optional[float],
                        bmi: Optional[float],
                        fat_mass_kg: Optional[float],
                        lean_mass_kg: Optional[float],
                        measured_comp: bool,
                        # recovery
                        rhr_bpm: Optional[float],
                        hrv_ms: Optional[float],
                        vo2: Optional[float],
                        # confidence
                        confidence_c: float = 1.0) -> Dict[str, float]:

    Tf = training_factor(days_per_week, minutes_per_week, device_strain_0_21)
    Sf = sleep_factor(sleep_hours_avg, sleep_quality_0_1)
    Bf = body_comp_factor(age, sex, bodyfat_pct, waist_cm, height_cm, bmi, fat_mass_kg, lean_mass_kg, measured_comp)

    HRVf = hrv_norm(hrv_ms, age, sex) if hrv_ms is not None else None
    RHRf = rhr_norm(rhr_bpm, age, sex) if rhr_bpm is not None else None
    VO2f = vo2_norm(vo2, age, sex) if vo2 is not None else None
    Rf = recovery_anchor(HRVf, RHRf, VO2f)

    if Rf is None:
        L_raw = 0.35 * Tf + 0.35 * Sf + 0.30 * Bf
    else:
        L_raw = 0.25 * Tf + 0.25 * Sf + 0.25 * Bf + 0.25 * Rf

    # data confidence: c=1.0 device-verified, 0.9 mixed, 0.8 self-report
    L = L_raw * (1.0 - 0.10 * (1.0 - clip(confidence_c, 0.0, 1.0)))
    L = clip(L, 0.0, 1.0)

    return dict(Tf=Tf, Sf=Sf, Bf=Bf, HRVf=HRVf, RHRf=RHRf, VO2f=VO2f, Rf=Rf, L_raw=L_raw, L=L)

def synergy_score(total_T_ngdl: float,
                  age: int, sex: str,
                  # lifestyle args passthrough
                  **lifestyle_kwargs) -> Dict[str, float]:

    H = hormone_component(total_T_ngdl, age, sex)

    lout = lifestyle_component(age, sex, **lifestyle_kwargs)
    L = lout["L"]

    # interaction-heavy hybrid
    a = 0.20; b = 0.20
    mult = max(1.0 - a - b, 0.0)
    S_star = a * H + b * L + mult * (H * L)
    score_raw = 100.0 * clip(S_star, 0.0, 1.0)

    lout.update({"H": H, "Score_raw": score_raw})
    return lout

# -----------------------------
# Guardrails and labeling
# -----------------------------
def compute_cap(age: int, sex: str,
                # lab recency
                t_test_date: Optional[date],
                # absolutes
                sleep_hours_avg: Optional[float],
                rhr_bpm: Optional[float],
                hrv_ms: Optional[float],
                # composition method
                comp_method: Literal["measured_bf", "whtr", "bmi", "unknown"],
                # confidence
                confidence_c: float) -> int:
    cap = 100

    # recency cap
    if t_test_date is None or months_since(t_test_date) > 6.0:
        cap = min(cap, 95)

    # sleep absolute floor
    if sleep_hours_avg is not None and sleep_hours_avg < 7.0:
        cap = min(cap, 95)

    # HRV/RHR: require at least age-median and minimal absolutes (illustrative floors)
    # You should replace these absolute floors with literature-based values.
    if hrv_ms is not None:
        # age-median ~ P50 approx midpoint of P10/P90
        hrv_p10 = HRV_P10_fn(age, sex)
        hrv_p90 = HRV_P90_fn(age, sex)
        hrv_p = clip((hrv_ms - hrv_p10) / nz(hrv_p90 - hrv_p10), 0.0, 1.0)
        if hrv_p < 0.5 or hrv_ms < 20:
            cap = min(cap, 98)
    if rhr_bpm is not None:
        rhr_p10 = RHR_P10_fn(age, sex)
        rhr_p90 = RHR_P90_fn(age, sex)
        rhr_p = clip((rhr_p90 - rhr_bpm) / nz(rhr_p90 - rhr_p10), 0.0, 1.0)
        if rhr_p < 0.5 or rhr_bpm > 78:
            cap = min(cap, 98)

    # composition method cap
    if comp_method == "bmi":
        cap = min(cap, 98)
    elif comp_method == "unknown":
        cap = min(cap, 95)

    # confidence (self-report only doesn't *cap*, but it reduces L; you may also cap if desired)
    # Example optional cap:
    # if confidence_c < 1.0:
    #     cap = min(cap, 99)

    return cap

def label_from_score(score: float) -> str:
    if score < 40: return "Low"
    if score < 60: return "Below Avg"
    if score < 75: return "Good"
    if score < 90: return "Strong"
    return "Peak"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hormone–Lifestyle Synergy (v3.1)", page_icon="🧪", layout="wide")

st.title("🧪 Age-Equalized Hormone–Lifestyle Synergy (v3.1, honest)")

st.write("""
This app computes an **age-fair synergy score** that blends **hormones** and **lifestyle** with an interaction term (so weak areas pull the score down).
**100/100** is *possible but rare* and requires **recent labs**, **verified data**, and meeting **absolute floors**.
**All reference tables here are placeholders** — replace them with your lab/device references before production.
""")

with st.sidebar:
    st.header("Inputs")
    colA, colB = st.columns(2)
    with colA:
        age = st.number_input("Age (years)", 18, 90, 35)
    with colB:
        sex = st.selectbox("Sex", ["male", "female"])

    st.subheader("Hormone (Total Testosterone)")
    unit = st.selectbox("T units", ["ng/dL", "nmol/L"])
    T_value = st.number_input(f"Total T ({unit})", min_value=0.0, value=600.0 if sex=="male" else 35.0, step=1.0)
    t_date = st.date_input("Lab test date", value=date.today())

    # convert T to ng/dL
    T_ngdl = T_value if unit == "ng/dL" else T_value / 0.0347  # 1 ng/dL = 0.0347 nmol/L

    st.subheader("Lifestyle")
    sleep_hours = st.number_input("Average sleep (hours)", min_value=0.0, max_value=12.0, value=7.5, step=0.1)
    sleep_quality = st.slider("Sleep quality (0–1, optional)", 0.0, 1.0, 0.8)

    st.caption("Training (use any one or more; device strain preferred if available)")
    days_per_week = st.slider("Training days per week", 0.0, 7.0, 4.0, 0.5)
    minutes_per_week = st.number_input("Training minutes per week", 0, 2000, 180)
    device_strain = st.slider("Device strain (0–21)", 0.0, 21.0, 12.0, 0.1)

    st.subheader("Body Composition (choose any)")
    comp_method = "unknown"
    bodyfat_pct = st.number_input("Body-fat % (if known)", 0.0, 70.0, 18.0 if sex=="male" else 26.0, step=0.1)
    bf_measured = st.checkbox("Body-fat measured (DEXA/impedance + tape)?", value=False)

    waist_cm = st.number_input("Waist (cm, optional)", 0.0, 200.0, 0.0, step=0.1)
    height_cm = st.number_input("Height (cm)", 120.0, 220.0, 178.0, step=0.1)

    weight_kg = st.number_input("Weight (kg, optional)", 0.0, 250.0, 0.0, step=0.1)
    bmi = None
    if weight_kg > 0 and height_cm > 0:
        bmi = weight_kg / ((height_cm/100.0)**2)

    fat_mass_kg = st.number_input("Fat mass (kg, optional)", 0.0, 150.0, 0.0, step=0.1)
    lean_mass_kg = st.number_input("Lean mass (kg, optional)", 0.0, 150.0, 0.0, step=0.1)

    # Decide composition method used (for capping logic)
    if fat_mass_kg > 0 and lean_mass_kg > 0:
        comp_method = "measured_bf"
    elif bodyfat_pct > 0:
        comp_method = "measured_bf" if bf_measured else "whtr"  # treat as higher confidence if measured
    elif waist_cm > 0 and height_cm > 0:
        comp_method = "whtr"
    elif bmi is not None and bmi > 0:
        comp_method = "bmi"

    st.subheader("Recovery Anchors (optional)")
    rhr_bpm = st.number_input("Resting Heart Rate (bpm)", 0.0, 120.0, 60.0, step=1.0)
    hrv_ms = st.number_input("HRV (RMSSD, ms)", 0.0, 200.0, 55.0, step=1.0)
    vo2 = st.number_input("VO₂max (mL/kg/min, optional)", 0.0, 90.0, 0.0, step=0.5)

    st.subheader("Data Confidence")
    confidence_choice = st.selectbox("Data source", ["Device-verified (both sleep & training)", "Mixed", "Self-reported only"])
    confidence_c = {"Device-verified (both sleep & training)": 1.0,
                    "Mixed": 0.9,
                    "Self-reported only": 0.8}[confidence_choice]

st.markdown("---")

# Compute lifestyle and score
lifestyle_kwargs = dict(
    days_per_week=days_per_week if days_per_week > 0 else None,
    minutes_per_week=int(minutes_per_week) if minutes_per_week > 0 else None,
    device_strain_0_21=device_strain if device_strain > 0 else None,
    sleep_hours_avg=sleep_hours if sleep_hours > 0 else None,
    sleep_quality_0_1=sleep_quality,
    bodyfat_pct=bodyfat_pct if bodyfat_pct > 0 else None,
    waist_cm=waist_cm if waist_cm > 0 else None,
    height_cm=height_cm if height_cm > 0 else None,
    bmi=bmi if bmi is not None and bmi > 0 else None,
    fat_mass_kg=fat_mass_kg if fat_mass_kg > 0 else None,
    lean_mass_kg=lean_mass_kg if lean_mass_kg > 0 else None,
    measured_comp=bf_measured or (fat_mass_kg > 0 and lean_mass_kg > 0),
    rhr_bpm=rhr_bpm if rhr_bpm > 0 else None,
    hrv_ms=hrv_ms if hrv_ms > 0 else None,
    vo2=vo2 if vo2 > 0 else None,
    confidence_c=confidence_c
)

results = synergy_score(T_ngdl, age, sex, **lifestyle_kwargs)

cap = compute_cap(age, sex, t_date, lifestyle_kwargs["sleep_hours_avg"],
                  lifestyle_kwargs["rhr_bpm"], lifestyle_kwargs["hrv_ms"],
                  comp_method, confidence_c)

final_score = min(results["Score_raw"], cap)
label = label_from_score(final_score)

# Display
col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("Result")
    st.metric("Synergy Score", f"{final_score:.0f} / 100", help="Capped by guardrails for realism")
    st.write(f"**Band:** {label}")
    st.caption(f"Cap applied: {cap}. Raw score before cap: {results['Score_raw']:.1f}")

    st.markdown("**Components**")
    st.write(f"Hormone H: `{results['H']:.2f}`  |  Lifestyle L (after confidence): `{results['L']:.2f}`")
    st.write(f"• Training Tf: `{results['Tf']:.2f}`  • Sleep Sf: `{results['Sf']:.2f}`  • Body comp Bf: `{results['Bf']:.2f}`")
    if results['Rf'] is not None:
        st.write(f"• Recovery anchor Rf: `{results['Rf']:.2f}`  (HRVf: `{results['HRVf']}`  RHRf: `{results['RHRf']}`  VO2f: `{results['VO2f']}`)")

with col2:
    st.subheader("Notes")
    st.write("""
- **100/100** requires: recent labs (≤6 months), verified data, and hitting absolute floors (sleep/HRV/RHR/VO₂).
- Reference tables here are **placeholders** — replace them with your own lab/device percentiles.
- Body-fat% measured gets full credit; BMI-only is capped.
- Confidence scaling reduces L for self-report.
""")

st.markdown("---")
st.caption("Educational use only. Not medical advice. Replace reference tables with your own validated data before using with clients.")
