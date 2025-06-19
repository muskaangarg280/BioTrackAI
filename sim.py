import numpy as np
import pandas as pd
import random
import datetime
import matplotlib.pyplot as plt

# Function to simulate heart rate based on time of day and activity level
def simulate_heart_rate(time_of_day, activity_level):
    base_hr = 70  # base heart rate at rest
    
    # Fluctuations based on time of day
    if time_of_day in range(6, 9):  # morning (waking up)
        hr = base_hr + random.uniform(5, 15)  # small increase
    elif time_of_day in range(9, 12):  # daytime (work)
        hr = base_hr + random.uniform(10, 30)  # moderate activity
    elif time_of_day in range(12, 18):  # exercise time
        hr = base_hr + random.uniform(30, 50)  # exercise intensity
    elif time_of_day in range(18, 22):  # evening (resting)
        hr = base_hr + random.uniform(5, 10)  # decrease after work
    else:  # night (sleeping)
        hr = base_hr - random.uniform(5, 10)  # lower during sleep
    
    # Add random noise
    hr += random.uniform(-5, 5)
    
    # Activity Level Impact
    if activity_level == "high":
        hr += random.uniform(20, 40)  # high exercise increases heart rate
    elif activity_level == "low":
        hr -= random.uniform(5, 10)  # low activity can lower heart rate
    
    return round(hr, 1)

# Function to simulate body temperature
def simulate_temperature(time_of_day):
    base_temp = 36.5  # normal body temperature
    
    if time_of_day in range(6, 9):  # morning (waking up)
        temp = base_temp - random.uniform(0.1, 0.3)  # slightly cooler in morning
    elif time_of_day in range(9, 18):  # daytime
        temp = base_temp + random.uniform(0.1, 0.2)  # slight increase
    else:  # night (sleeping)
        temp = base_temp - random.uniform(0.2, 0.3)  # cool down at night
    
    # Add random noise for natural variation
    temp += random.uniform(-0.1, 0.1)
    
    return round(temp, 1)

# Function to simulate blood pressure
def simulate_blood_pressure(time_of_day, activity_level):
    systolic = 120  # base systolic pressure
    diastolic = 80  # base diastolic pressure
    
    if time_of_day in range(6, 9):  # morning
        systolic += random.uniform(5, 10)
        diastolic += random.uniform(3, 5)
    elif time_of_day in range(9, 12):  # workday
        systolic += random.uniform(5, 10)
        diastolic += random.uniform(2, 5)
    elif time_of_day in range(12, 18):  # exercise time
        systolic += random.uniform(20, 40)  # large increase
        diastolic += random.uniform(10, 15)  # large increase
    elif time_of_day in range(18, 22):  # evening relaxation
        systolic -= random.uniform(5, 10)
        diastolic -= random.uniform(3, 5)
    else:  # night sleep
        systolic -= random.uniform(5, 10)
        diastolic -= random.uniform(3, 5)
    
    # Activity Level Impact
    if activity_level == "high":
        systolic += random.uniform(15, 30)
        diastolic += random.uniform(10, 20)
    elif activity_level == "low":
        systolic -= random.uniform(5, 10)
        diastolic -= random.uniform(3, 5)
    
    return round(systolic, 1), round(diastolic, 1)

# Function to simulate blood oxygen saturation (SpO2)
def simulate_spo2():
    spo2 = random.uniform(95, 100)  # Normal oxygen saturation
    if random.random() < 0.05:  # 5% chance of low SpO2 due to health conditions
        spo2 = random.uniform(85, 90)  # Simulate hypoxemia
    return round(spo2, 1)

# Function to simulate respiratory rate (RR)
def simulate_rr(time_of_day, activity_level):
    base_rr = 14  # normal resting respiratory rate
    
    if time_of_day in range(6, 9):  # morning
        rr = base_rr - random.uniform(1, 2)  # slightly lower in the morning
    elif time_of_day in range(9, 12):  # daytime work
        rr = base_rr + random.uniform(1, 3)  # moderate increase
    elif time_of_day in range(12, 18):  # exercise
        rr = base_rr + random.uniform(5, 8)  # significant increase during exercise
    elif time_of_day in range(18, 22):  # evening
        rr = base_rr + random.uniform(2, 3)
    else:  # night (sleeping)
        rr = base_rr - random.uniform(1, 3)  # lower during sleep
    
    # Activity Level Impact
    if activity_level == "high":
        rr += random.uniform(3, 5)  # exercise increases breathing rate
    elif activity_level == "low":
        rr -= random.uniform(1, 2)  # low activity decreases rate
    
    return round(rr, 1)

# Function to simulate data for a single day (24 hours)
def generate_simulated_data(user_id, num_hours=24):
    data = []
    for hour in range(num_hours):
        # Activity levels: "high", "low", "moderate"
        activity_level = random.choice(["low", "moderate", "high"])
        
        hr = simulate_heart_rate(hour, activity_level)
        temp = simulate_temperature(hour)
        systolic, diastolic = simulate_blood_pressure(hour, activity_level)
        spo2 = simulate_spo2()
        rr = simulate_rr(hour, activity_level)
        
        timestamp = datetime.datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
        data.append([user_id, timestamp, hr, temp, systolic, diastolic, spo2, rr, activity_level])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["User_ID", "Timestamp", "Heart_Rate", "Body_Temperature", "Systolic_BP", "Diastolic_BP", "SpO2", "Respiratory_Rate", "Activity_Level"])
    return df

# Simulate data for a single user
user_id = 1
simulated_data = generate_simulated_data(user_id)

# Display the simulated data
print(simulated_data.head())

# Plot HR data for visualization
plt.plot(simulated_data["Timestamp"], simulated_data["Heart_Rate"])
plt.xlabel("Time")
plt.ylabel("Heart Rate (BPM)")
plt.title(f"Simulated Heart Rate for User {user_id}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
