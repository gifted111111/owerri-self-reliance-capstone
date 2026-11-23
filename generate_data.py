import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 600

device_tier = np.random.choice([0, 1, 2], size=n_samples, p=[0.35, 0.45, 0.20])
daily_power_hours = np.clip(np.random.normal(loc=6.5, scale=3.5, size=n_samples), 1, 18).round(1)
daily_data_mb = np.random.choice(
    [150, 300, 500, 1000, 1500, 2500, 5000],
    size=n_samples,
    p=[0.15, 0.25, 0.25, 0.15, 0.10, 0.05, 0.05],
)
campus_lab_access = np.random.choice([0, 1, 2], size=n_samples, p=[0.50, 0.35, 0.15])
peer_group_score = np.random.randint(1, 6, size=n_samples)
weekly_practice_hours = np.clip(
    1.5 * device_tier + 0.4 * daily_power_hours + 0.8 * peer_group_score + np.random.normal(0, 2, n_samples),
    1,
    30,
).round(1)

labels = []
for i in range(n_samples):
    if daily_power_hours[i] <= 4.5 or daily_data_mb[i] <= 300:
        if np.random.rand() < 0.90:
            labels.append(0)
        else:
            labels.append(1 if campus_lab_access[i] == 0 else 2)
    elif device_tier[i] <= 1 and campus_lab_access[i] <= 1:
        if np.random.rand() < 0.88:
            labels.append(1)
        else:
            labels.append(2)
    else:
        if np.random.rand() < 0.88:
            labels.append(2)
        else:
            labels.append(1)

df = pd.DataFrame({
    "device_tier": device_tier,
    "daily_power_hours": daily_power_hours,
    "daily_data_mb": daily_data_mb,
    "campus_lab_access": campus_lab_access,
    "peer_group_score": peer_group_score,
    "weekly_practice_hours": weekly_practice_hours,
    "target_phase": labels,
})

df.to_csv("student_constraints.csv", index=False)
print("Dataset saved to student_constraints.csv")
print("\nClass distribution:")
print(df["target_phase"].value_counts().sort_index())
