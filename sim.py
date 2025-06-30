import numpy as np
import pandas as pd
import random
import datetime
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

"""
Real-World Wearable Vital-Sign Simulator
=======================================
• **Sampling cadence:** **1-minute** intervals – typical of consumer smartwatches and clinical wearable patches.
• **Scenarios with realistic durations (in minutes):**
  ─ `asthma_attack` …… 10–30 min  
  ─ `panic_attack`  …… 10–20 min  
  ─ `febrile_sepsis` … 240–480 min (4-8 h)  
  ─ `post_op_infection` … 360–720 min (6-12 h)
• **3 % random NaNs** as drop-out artifacts.
• Keeps all prior physiology (BMI, pregnancy, comorbidities, medication effects, circadian rhythm).
"""

# ------------------------- Helper utilities ------------------------- #

def trunc_norm(mu: float, sigma: float, low: float, high: float) -> float:
    a, b = (low - mu) / sigma, (high - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)


def circadian_temp(hour: int, base: float = 36.8, amp: float = 0.3, peak_hour: int = 17) -> float:
    phase = 2 * np.pi * ((hour - peak_hour) % 24) / 24
    return base + amp * np.sin(phase)

# ----------------------- Profile generation ------------------------ #

def create_user_profile():
    age_weights = [0.15] * 10 + [0.25] * 13 + [0.25] * 20 + [0.15] * 10 + [0.05] * 10
    age = random.choices(range(18, 81), weights=age_weights, k=1)[0]
    gender = random.choice(["male", "female"])

    high_impact = [
        ("chronic_pain", 0.08), ("thyroid_disorder", 0.07), ("cancer", 0.05),
        ("COPD", 0.04), ("anemia", 0.06), ("smoking", 0.25)
    ]
    common = [
        ("stress", 0.10), ("hypertension", 0.15), ("asthma", 0.07),
        ("sleep_apnea", 0.08), ("diabetes", 0.10), ("cardiovascular_disease", 0.10),
        ("obesity", 0.20), ("chronic_kidney_disease", 0.03), ("arthritis", 0.05),
        ("beta_blockers", 0.07), ("bronchodilators", 0.05), ("ace_inhibitors", 0.04)
    ]
    problems = [c for c, p in (common + high_impact) if random.random() < p] or ["none"]

    pregnant = False
    if gender == "female" and 18 <= age <= 45:
        pregnant = random.random() < 0.15

    bmi = max(15, min(45, round(random.gauss(25, 4), 1)))
    altitude = random.choices([0, 1, 2], weights=[0.85, 0.10, 0.05])[0]
    fitness = random.choices(["low", "moderate", "high"], weights=[0.4, 0.5, 0.1])[0]

    return age, gender, problems, altitude, fitness, bmi, pregnant


# -------------------- Vital‑sign simulators ------------------------ #

def simulate_heart_rate(hour, activity, age, problems, fit, bmi, pregnant):
    hr = trunc_norm(72, 8, 45, 110)
    if age > 50:
        hr += 5
    if fit == "high":
        hr -= 10
    elif fit == "low":
        hr += 8

    # Activity load
    if activity == "high":
        hr += random.uniform(40, 70)
    elif activity == "moderate":
        hr += random.uniform(20, 40)

    # Condition modifiers
    if bmi >= 30 or "obesity" in problems:
        hr += 5
    if "stress" in problems or "chronic_pain" in problems:
        hr += 5
    if "smoking" in problems:
        hr += 5
    if "thyroid_disorder" in problems and random.random() < 0.5:  # hyperthyroid half the time
        hr += 10
    if "thyroid_disorder" in problems and random.random() >= 0.5:
        hr -= 10
    if "anemia" in problems:
        hr += 8
    if "beta_blockers" in problems:
        hr -= random.uniform(10, 15)
    if "bronchodilators" in problems:
        hr += 5
    if pregnant:
        hr += 10

    return round(max(hr, 40), 1)


def simulate_temperature(hour, problems, age, altitude, activity, bmi, pregnant):
    temp = circadian_temp(hour)

    if age > 60:
        temp -= 0.1
    if altitude == 1:
        temp -= 0.1
    elif altitude == 2:
        temp -= 0.2
    if activity == "high":
        temp += 0.3

    # Conditions
    if bmi >= 30 or "obesity" in problems:
        temp += 0.2
    if "stress" in problems or "chronic_pain" in problems:
        temp += 0.1
    if "thyroid_disorder" in problems and random.random() < 0.5:  # hyper
        temp += 0.3
    if "thyroid_disorder" in problems and random.random() >= 0.5:  # hypo
        temp -= 0.3
    if pregnant:
        temp += 0.3

    return round(temp + random.uniform(-0.2, 0.2), 1)


def simulate_blood_pressure(hour, activity, problems, age, bmi, pregnant):
    sys, dia = 115, 75

    if age > 50:
        sys += 5; dia += 3
    if bmi >= 30 or "obesity" in problems:
        sys += 5; dia += 3
    if pregnant:
        sys += 5; dia += 2

    if "hypertension" in problems:
        sys += 20; dia += 15
    if "beta_blockers" in problems:
        sys -= 10; dia -= 5
    if "ace_inhibitors" in problems:
        sys -= 5; dia -= 3
    if "smoking" in problems:
        sys += 5; dia += 3
    if "stress" in problems or "chronic_pain" in problems:
        sys += 5; dia += 3
    if "thyroid_disorder" in problems and random.random() < 0.5:
        sys += 5
    if "thyroid_disorder" in problems and random.random() >= 0.5:
        sys -= 5

    # Activity influence
    if activity == "high":
        sys += 10; dia += 5
    elif activity == "low":
        sys -= 5; dia -= 3
    if hour in range(12, 18):
        sys += 5

    return round(sys, 1), round(dia, 1)


def simulate_spo2(altitude, problems, bmi, pregnant):
    spo2 = 98 - altitude * 3  # ~‑3% per 1000‑1500m bucket

    if "COPD" in problems:
        spo2 -= 5
    if "asthma" in problems:
        spo2 -= 3
    if "sleep_apnea" in problems:
        spo2 -= 5
    if "smoking" in problems:
        spo2 -= 3
    if "anemia" in problems:
        spo2 -= 1

    return round(max(spo2 + random.uniform(-1.5, 1.5), 80), 1)


def simulate_rr(hour, activity, problems, age, bmi, pregnant):
    rr = trunc_norm(16, 2, 12, 28)

    if age > 60:
        rr += 1
    if activity == "high":
        rr += random.uniform(5, 10)
    if activity == "moderate":
        rr += random.uniform(2, 5)

    # Conditions
    if bmi >= 30:
        rr += 2
    if pregnant:
        rr += 2
    if "COPD" in problems:
        rr += 3
    if "asthma" in problems:
        rr += 2
    if "anemia" in problems:
        rr += 2
    if "thyroid_disorder" in problems and random.random() < 0.5:  # hyper
        rr += 2
    if "thyroid_disorder" in problems and random.random() >= 0.5:  # hypo
        rr -= 2

    if "stress" in problems or "chronic_pain" in problems:
        rr += 1

    return round(rr, 1)

# -------------------- Scenario injection helpers ------------------- #

DURATION_MAP = {
    "asthma_attack": lambda: random.randint(10, 30),        # minutes
    "panic_attack": lambda: random.randint(10, 20),
    "febrile_sepsis": lambda: random.randint(240, 480),
    "post_op_infection": lambda: random.randint(360, 720),
}


def apply_scenario(row: dict, scenario: str):
    if scenario == "febrile_sepsis":
        row["Body_Temperature"] += 1.5
        row["Heart_Rate"] += 30
        row["Systolic_BP"] -= 20
        row["Diastolic_BP"] -= 10
    elif scenario == "asthma_attack":
        row["SpO2"] -= 7
        row["Respiratory_Rate"] += 6
        row["Heart_Rate"] += 20
    elif scenario == "panic_attack":
        row["Heart_Rate"] += 25
        row["Respiratory_Rate"] += 8
        row["SpO2"] -= 2
        row["Body_Temperature"] += 0.4
    elif scenario == "post_op_infection":
        row["Body_Temperature"] += 1.0
        row["Heart_Rate"] += 20
        row["Systolic_BP"] -= 10
    # Clamp
    row["SpO2"] = max(row["SpO2"], 70)
    row["Heart_Rate"] = min(row["Heart_Rate"], 220)

# -------------------- Row generator --------------------- #

def generate_row(user_id: int, ts: datetime.datetime, profile: tuple) -> dict:
    age, gender, probs, altitude, fit, bmi, pregnant = profile
    activity = random.choices(["low", "moderate", "high"], [0.4, 0.4, 0.2])[0]

    hr = simulate_heart_rate(ts.hour, activity, age, probs, fit, bmi, pregnant)
    temp = simulate_temperature(ts.hour, probs, age, altitude, activity, bmi, pregnant)
    sys, dia = simulate_blood_pressure(ts.hour, activity, probs, age, bmi, pregnant)
    spo2 = simulate_spo2(altitude, probs, bmi, pregnant)
    rr = simulate_rr(ts.hour, activity, probs, age, bmi, pregnant)

    return {"User_ID": user_id, "Timestamp": ts,
            "Heart_Rate": hr, "Body_Temperature": temp,
            "Systolic_BP": sys, "Diastolic_BP": dia,
            "SpO2": spo2, "Respiratory_Rate": rr,
            "Scenario": "normal"}


def user_data(n=10):
    now = datetime.datetime.now()
    rows = [generate_row(uid, now, create_user_profile()) for uid in range(1, n + 1)]
    return pd.DataFrame(rows)


def hours_data(hours: int = 24, scenario: str | None = None):
    start = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    profile = create_user_profile()

    timestamps = pd.date_range(start, periods=hours, freq="H")
    rows = [generate_row(1, ts.to_pydatetime(), profile) for ts in timestamps]

    if scenario in DURATION_MAP:
        dur_min = DURATION_MAP[scenario]()
        dur_hr  = max(1, dur_min // 60)          # convert to hours
        start_hr = random.randint(0, hours - dur_hr)
        for i in range(start_hr, start_hr + dur_hr):
            apply_scenario(rows[i], scenario)
            rows[i]["Scenario"] = scenario

    for row in rows:
        for key in ("Heart_Rate", "Body_Temperature", "Systolic_BP", "SpO2", "Respiratory_Rate"):
            if random.random() < 0.03:
                row[key] = np.nan

    return pd.DataFrame(rows)

# -------------------- Minute‑level series generator --------------------- #

def minutes_data(minutes: int = 1440, scenario: str | None = None):
    """Return a 1‑minute cadence dataframe for a single user."""
    start = datetime.datetime.now().replace(second=0, microsecond=0)
    profile = create_user_profile()
    times = [start + datetime.timedelta(minutes=i) for i in range(minutes)]

    rows = [generate_row(1, ts, profile) for ts in times]

    # Inject scenario with realistic duration
    if scenario in DURATION_MAP:
        dur = DURATION_MAP[scenario]()
        if dur >= minutes:  # too long for series
            dur = minutes // 2
        start_idx = random.randint(0, minutes - dur)
        for i in range(start_idx, start_idx + dur):
            apply_scenario(rows[i], scenario)
            rows[i]["Scenario"] = scenario

    # 3 % random NaN artifacts
    for row in rows:
        for key in ("Heart_Rate", "Body_Temperature", "Systolic_BP", "SpO2", "Respiratory_Rate"):
            if random.random() < 0.03:
                row[key] = np.nan

    return pd.DataFrame(rows)

# ----------------------------- Demo ----------------------------- #

if __name__ == "__main__":

    df1 = user_data(5)
    df2 = hours_data(48, scenario="panic_attack")
    df3 = minutes_data(60, scenario="panic_attack")

    print(df1)
    print(df2[df2["Scenario"] != "normal"])

    plt.plot(df3["Timestamp"], df3["Heart_Rate"])
    plt.title("HR with Panic Attack")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
