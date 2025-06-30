import numpy as np
import pandas as pd
import random
import datetime
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

"""
Improved Wearable Vital-Sign Simulator
======================================
• Realistic temporal smoothing (limits per-minute changes).
• Sticky activity states to avoid erratic changes.
• 3% NaN artifacts (dropouts).
• Preserves comorbidities, medications, circadian rhythm.
"""

# ------------------------- Helper utilities ------------------------- #

def trunc_norm(mu, sigma, low, high):
    a, b = (low - mu) / sigma, (high - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)

def circadian_temp(hour, base=36.8, amp=0.3, peak_hour=17):
    phase = 2 * np.pi * ((hour - peak_hour) % 24) / 24
    return base + amp * np.sin(phase)

# ------------------------- Profile & Conditions ------------------------- #

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

    pregnant = gender == "female" and 18 <= age <= 45 and random.random() < 0.15
    bmi = max(15, min(45, round(random.gauss(25, 4), 1)))
    altitude = random.choices([0, 1, 2], [0.85, 0.10, 0.05])[0]
    fitness = random.choices(["low", "moderate", "high"], [0.4, 0.5, 0.1])[0]

    return age, gender, problems, altitude, fitness, bmi, pregnant

# ------------------------- Scenario Durations ------------------------- #

DURATION_MAP = {
    "asthma_attack": lambda: random.randint(10, 30),
    "panic_attack": lambda: random.randint(10, 20),
    "febrile_sepsis": lambda: random.randint(240, 480),
    "post_op_infection": lambda: random.randint(360, 720),
}

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


# ------------------------- User Class ------------------------- #

class User:
    def __init__(self, user_id):
        self.id = user_id
        self.age, self.gender, self.problems, self.altitude, self.fitness, self.bmi, self.pregnant = create_user_profile()
        self.last_vals = {}
        self.activity = random.choices(["low", "moderate", "high"], [0.4, 0.4, 0.2])[0]

    def smooth(self, key, val, max_delta):
        prev = self.last_vals.get(key)
        if prev is None:
            self.last_vals[key] = val
            return val
        new_val = max(min(val, prev + max_delta), prev - max_delta)
        self.last_vals[key] = new_val + random.uniform(-1, 1)
        return round(new_val, 1)

    def generate_row(self, ts):
        if random.random() < 0.1:
            self.activity = random.choices(["low", "moderate", "high"], [0.4, 0.4, 0.2])[0]

        hr = simulate_heart_rate(ts.hour, self.activity, self.age, self.problems, self.fitness, self.bmi, self.pregnant)
        temp = simulate_temperature(ts.hour, self.problems, self.age, self.altitude, self.activity, self.bmi, self.pregnant)
        sys, dia = simulate_blood_pressure(ts.hour, self.activity, self.problems, self.age, self.bmi, self.pregnant)
        spo2 = simulate_spo2(self.altitude, self.problems, self.bmi, self.pregnant)
        rr = simulate_rr(ts.hour, self.activity, self.problems, self.age, self.bmi, self.pregnant)

        return {
            "User_ID": self.id, "Timestamp": ts,
            "Heart_Rate": self.smooth("HR", hr, 6),
            "Body_Temperature": self.smooth("TEMP", temp, 0.2),
            "Systolic_BP": self.smooth("SYS", sys, 5),
            "Diastolic_BP": self.smooth("DIA", dia, 4),
            "SpO2": self.smooth("SpO2", spo2, 2),
            "Respiratory_Rate": self.smooth("RR", rr, 2),
            "Scenario": "normal"
        }

# ------------------------- Scenario Logic ------------------------- #

def apply_scenario(row, scenario):
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

    row["SpO2"] = max(row["SpO2"], 70)
    row["Heart_Rate"] = min(row["Heart_Rate"], 220)

# ------------------------- Simulator Class ------------------------- #

class Simulator:
    def __init__(self, n_users=1):
        self.users = [User(uid) for uid in range(1, n_users + 1)]

    def run_minutes(self, minutes=1440, scenario=None, na_rate=0.03):
        start = datetime.datetime.now().replace(second=0, microsecond=0)
        all_rows = []
        for i in range(minutes):
            ts = start + datetime.timedelta(minutes=i)
            for user in self.users:
                row = user.generate_row(ts)
                all_rows.append(row)

        if scenario:
            dur = DURATION_MAP.get(scenario, lambda: 0)()
            start_idx = random.randint(0, max(0, len(all_rows) - dur))
            for i in range(start_idx, start_idx + dur):
                if i < len(all_rows):
                    apply_scenario(all_rows[i], scenario)
                    all_rows[i]["Scenario"] = scenario

        for row in all_rows:
            for key in ("Heart_Rate", "Body_Temperature", "Systolic_BP", "SpO2", "Respiratory_Rate"):
                if random.random() < na_rate:
                    row[key] = np.nan

        return pd.DataFrame(all_rows)

# ------------------------- Demo ------------------------- #

if __name__ == "__main__":
    sim = Simulator(n_users=1)
    df_normal = sim.run_minutes(144)
    sim2 = Simulator(n_users=1)
    df_asthma = sim2.run_minutes(144, scenario="asthma_attack")

    plt.plot(df_normal["Timestamp"], df_normal["Heart_Rate"], label="Normal", solid_joinstyle="round", solid_capstyle="round")
    plt.plot(df_asthma["Timestamp"], df_asthma["Heart_Rate"], label="Asthma", linestyle="--", solid_joinstyle="round", solid_capstyle="round")
    plt.xticks(rotation=45)
    plt.title("HR with and without Asthma Attack")
    plt.legend()
    plt.tight_layout()
    plt.show()
