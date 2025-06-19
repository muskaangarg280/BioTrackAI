import numpy as np
import pandas as pd
import random
import datetime
import matplotlib.pyplot as plt

# Function to create a randomized user profile (age, gender, health conditions, altitude, fitness level)
def create_user_profile():
    # Random age between 18 and 80, with higher chance of middle-aged
    age_weights = [0.15] * 10 + [0.25] * 13 + [0.25] * 20 + [0.15] * 10 + [0.05] * 10
    assert len(age_weights) == 63, f"Age weights should have 63 elements, but it has {len(age_weights)}"
    
    age = random.choices(range(18, 81), weights=age_weights, k=1)[0]
    gender = random.choice(['male', 'female'])  # Randomize gender
    health_condition = random.choices(
        ['healthy', 'stress', 'hypertension', 'asthma', 'sleep_apnea', 'diabetes', 'cardiovascular_disease', 'obesity', 'chronic_kidney_disease', 'arthritis'],
        weights=[0.60, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02, 0.03],  
        k=1
    )[0]
    
    altitude = random.choices([0, 1, 2], weights=[0.85, 0.10, 0.05], k=1)[0]  
    fitness_level = random.choices(['low', 'moderate', 'high'], weights=[0.4, 0.5, 0.1], k=1)[0]  
    medications = random.choices([None, 'beta_blockers', 'ace_inhibitors', 'bronchodilators'], weights=[0.85, 0.05, 0.05, 0.05], k=1)[0]  
    
    return age, gender, health_condition, altitude, fitness_level, medications

# Simulate Body Temperature
def simulate_temperature(time_of_day, health_condition, age, altitude, activity_level, last_meal_time=None):
    base_temp = random.uniform(36.3, 36.7)
    if age > 60:
        base_temp -= random.uniform(0.1, 0.2)  
    
    if altitude == 1:  
        base_temp -= random.uniform(0.1, 0.2)  
    elif altitude == 2:  
        base_temp -= random.uniform(0.2, 0.4)  
    
    if time_of_day in range(6, 9):  
        temp = base_temp - random.uniform(0.1, 0.3)
    elif time_of_day in range(9, 18):  
        temp = base_temp + random.uniform(0.1, 0.2)
        if activity_level == "high":
            temp += random.uniform(0.3, 0.5)  
    else:  
        temp = base_temp - random.uniform(0.2, 0.3)
    
    if health_condition == "stress":
        temp += random.uniform(0.2, 0.4)
    elif health_condition == "fever":
        temp += random.uniform(0.5, 1.0)
    elif health_condition == "obesity":
        temp += random.uniform(0.1, 0.3)
    
    if last_meal_time and time_of_day in last_meal_time:
        temp += random.uniform(0.1, 0.3)  
    
    return round(temp, 1)

# Simulate heart rate
def simulate_heart_rate(time_of_day, activity_level, age, health_condition, fitness_level, medications):
    base_hr = random.uniform(75, 85)
    
    if age > 50:
        base_hr -= random.uniform(5, 10) 
    
    if fitness_level == "high":
        base_hr -= random.uniform(5, 10)
    elif fitness_level == "low":
        base_hr += random.uniform(5, 10)
    
    if time_of_day in range(6, 9):  
        hr = base_hr + random.uniform(3, 12)
    elif time_of_day in range(9, 12):  
        hr = base_hr + random.uniform(8, 25)
    elif time_of_day in range(12, 18):  
        hr = base_hr + random.uniform(25, 45)
    elif time_of_day in range(18, 22):  
        hr = base_hr + random.uniform(2, 12)
    else:  
        hr = base_hr - random.uniform(3, 8)
    
    if activity_level == "high":
        hr += random.uniform(25, 35)
    elif activity_level == "low":
        hr -= random.uniform(5, 10)
    
    if health_condition == "stress":
        hr += random.uniform(10, 20)  
    elif health_condition == "hypertension":
        hr += random.uniform(5, 15)  
    elif health_condition == "diabetes":
        hr += random.uniform(5, 10)  
    
    if medications == "beta_blockers":
        hr -= random.uniform(10, 20)  
    elif medications == "ace_inhibitors":
        hr -= random.uniform(5, 10)  
    elif medications == "bronchodilators":
        hr += random.uniform(5, 10)  
    
    return round(hr, 1)

# Simulate blood pressure
def simulate_blood_pressure(time_of_day, activity_level, health_condition, age, medications, last_meal_time=None):
    systolic = 120  
    diastolic = 80
    
    if age > 50:
        systolic += random.uniform(5, 10)
        diastolic += random.uniform(3, 5)
    
    if activity_level == "low":
        systolic += random.uniform(5, 10)
        diastolic += random.uniform(5, 10)
    
    if health_condition == "hypertension":
        systolic += random.uniform(15, 30)
        diastolic += random.uniform(10, 20)
    elif health_condition == "stress":
        systolic += random.uniform(5, 15)
        diastolic += random.uniform(3, 10)
    
    if medications == "beta_blockers":
        systolic -= random.uniform(10, 20)
        diastolic -= random.uniform(5, 10)
    elif medications == "ace_inhibitors":
        systolic -= random.uniform(5, 10)
        diastolic -= random.uniform(3, 7)
    
    if time_of_day in range(6, 9):  
        systolic += random.uniform(5, 10)
        diastolic += random.uniform(3, 5)
    elif time_of_day in range(12, 18):  
        systolic += random.uniform(20, 40)
        diastolic += random.uniform(10, 15)
    elif time_of_day in range(18, 22):  
        systolic -= random.uniform(5, 10)
        diastolic -= random.uniform(3, 5)

    if last_meal_time and time_of_day in last_meal_time:
        systolic += random.uniform(5, 10)
        diastolic += random.uniform(3, 5)
    
    return round(systolic, 1), round(diastolic, 1)

# Function to simulate blood oxygen saturation (SpO2) based on altitude and health conditions
def simulate_spo2(altitude, health_condition):
    if altitude == 0:  
        spo2 = random.uniform(95, 100)
    elif altitude == 1:  
        spo2 = random.uniform(90, 95)
    else:  
        spo2 = random.uniform(85, 90)
    
    if health_condition == "asthma":
        spo2 = random.uniform(85, 95)
    elif health_condition == "sleep_apnea":
        spo2 = random.uniform(80, 90)
    
    return round(spo2, 1)

# Function to simulate respiratory rate (RR) based on activity, health condition, and age
def simulate_rr(time_of_day, activity_level, health_condition, age):
    base_rr = random.uniform(15, 18)  
    
    if age > 60:
        base_rr += random.uniform(1, 3)  
    
    if time_of_day in range(6, 9):  
        rr = base_rr - random.uniform(1, 2)
    elif time_of_day in range(9, 12):  
        rr = base_rr + random.uniform(1, 3)
    elif time_of_day in range(12, 18):  
        rr = base_rr + random.uniform(5, 8)
    elif time_of_day in range(18, 22):  
        rr = base_rr + random.uniform(2, 3)
    else:  
        rr = base_rr - random.uniform(1, 3)
    
    if activity_level == "high":
        rr += random.uniform(3, 5)
    elif activity_level == "low":
        rr -= random.uniform(1, 2)
    
    if health_condition == "stress":
        rr += random.uniform(2, 5)  
    
    return round(rr, 1)

# Function to generate data based on user_id and time
def generate_data(user_id=1, time=None):
    # If time is not passed, use the current time by default
    if time is None:
        time = datetime.datetime.now()

    # Get user profile
    age, gender, health_condition, altitude, fitness_level, medications = create_user_profile()
    
    # Random activity level for the user at this hour
    activity_level = random.choice(["low", "moderate", "high"])

    # Simulate health data
    hr = simulate_heart_rate(time, activity_level, age, health_condition, fitness_level, medications)
    temp = simulate_temperature(time, health_condition, age, altitude, activity_level)
    systolic, diastolic = simulate_blood_pressure(time, activity_level, health_condition, age, medications)
    spo2 = simulate_spo2(altitude, health_condition)
    rr = simulate_rr(time, activity_level, health_condition, age)

    # Return data as a list for the current user at the given time
    return [user_id, time, hr, temp, systolic, diastolic, spo2, rr, activity_level, age, gender, health_condition, fitness_level]

# Function to display data (generate data for multiple users)
def user_data(num_users):
    data = []
    current_time = datetime.datetime.now()
    for user_id in range(1, num_users + 1):
        user_data = generate_data(user_id, current_time)
        data.append(user_data)

    # Display the data in a table format
    df = pd.DataFrame(data, columns=["User_ID", "Timestamp", "Heart_Rate", "Body_Temperature", "Systolic_BP", "Diastolic_BP", "SpO2", "Respiratory_Rate", "Activity_Level", "Age", "Gender", "Health_Condition", "Fitness_Level"])
    return df

# Function to generate data across a given number of hours (does not loop back after 24 hours)
def hours_data(num_hours=24):
    data = []
    user_id = 1
    current_time = datetime.datetime.now()
    
    # Start from the current hour and generate the data for the given number of hours
    for hour in range(num_hours): 
        time = current_time + datetime.timedelta(hours=hour)  # Increase hour sequentially
        user_data = generate_data(user_id, time)
        data.append(user_data)

    df = pd.DataFrame(data, columns=["User_ID", "Timestamp", "Heart_Rate", "Body_Temperature", "Systolic_BP", "Diastolic_BP", "SpO2", "Respiratory_Rate", "Activity_Level", "Age", "Gender", "Health_Condition", "Fitness_Level"])

    return df

# Plot HR data for given number of users at current time
print(user_data(10))

# Plot HR data for one user for given number of hours
hours = hours_data(30)
plt.plot(hours["Timestamp"], hours["Heart_Rate"])
plt.xlabel("Time")
plt.ylabel("Heart Rate (BPM)")
plt.title(f"Simulated Heart Rate for User 1 on {datetime.datetime.now().strftime('%Y-%m-%d')}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
