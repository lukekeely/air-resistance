import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import os

# Constants
B = 1.6e-4  # Linear drag coefficient (N s/m^2)
C = 0.25    # Quadratic drag coefficient (N s^2/m^4)
g = 9.81    # Acceleration due to gravity (m/s^2)
dt = 1e-5  # Time step (s)

densities = np.linspace(1000, 10000, 10)  # Density range
diameters = np.linspace(2e-5, 1e-4, 10)  # Diameter range
velocities = [1, 5, 10]  # Test velocities

angles = np.radians(np.linspace(1, 89, 89))  # Launch angles from 1 to 89 degrees
masses = np.array([density * (4/3) * np.pi * (D/2)**3 for density in densities for D in diameters]).reshape(len(densities), len(diameters))

output_dir = "optimum_angle"
os.makedirs(output_dir, exist_ok=True)
sns.set_theme()
collected_data = []

def projectile_motion_with_air_resistance(Vx, Vy, D, mass):
    total_velocity = np.sqrt(Vx**2 + Vy**2)
    bV = B * D * total_velocity
    cV2 = C * D**2 * total_velocity**2
    total_drag = bV + cV2
    air_resistance_force = total_drag / mass
    ax = -air_resistance_force * Vx
    ay = -g - air_resistance_force * Vy
    return ax, ay

def simulate_projectile_motion(theta, mass, D, V):
    Vx = V * np.cos(theta)
    Vy = V * np.sin(theta)
    X, Y = 0, 0
    while Y >= 0:
        ax, ay = projectile_motion_with_air_resistance(Vx, Vy, D, mass)
        Vx += ax * dt
        Vy += ay * dt
        X += Vx * dt
        Y += Vy * dt
    return X

def find_optimal_angles(V):
    optimal_angles_matrix = np.zeros((len(densities), len(diameters)))
    for i, mass_row in enumerate(tqdm(masses, desc=f"Velocity {V} m/s")):
        for j, mass in enumerate(mass_row):
            D = diameters[j]
            distances = [simulate_projectile_motion(theta, mass, D, V) for theta in angles]
            optimal_angle = angles[np.argmax(distances)]
            optimal_angles_matrix[i, j] = np.degrees(optimal_angle)
    return optimal_angles_matrix

for V in velocities:
    optimal_angles_matrix = find_optimal_angles(V)

    for i, density in enumerate(densities):
        for j, diameter in enumerate(diameters):
            collected_data.append([V, density, diameter, optimal_angles_matrix[i, j]])

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(optimal_angles_matrix, cmap="cubehelix_r", vmin=np.min(optimal_angles_matrix), vmax=np.max(optimal_angles_matrix))
    ax.set_title(f'Optimal Launch Angle Heatmap (Velocity = {V} m/s)')
    ax.set_xlabel('Diameter (m)')
    ax.set_ylabel('Density (kg/m³)')
    plt.xticks(ticks=np.linspace(0, len(diameters)-1, 6), labels=np.round(np.linspace(diameters.min(), diameters.max(), 6), 5))
    plt.yticks(ticks=np.linspace(0, len(densities)-1, 6), labels=np.round(np.linspace(densities.min(), densities.max(), 6), 0))
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig(f'{output_dir}/heatmap_velocity_{V}.png')
    plt.savefig(f'{output_dir}/heatmap_velocity_{V}.pdf')
    plt.show()

columns = ['Velocity', 'Density', 'Diameter', 'Optimal_Angle']
collected_data_df = pd.DataFrame(collected_data, columns=columns)
collected_data_df.to_csv(f"{output_dir}/optimum_angle_data.csv", index=False)
