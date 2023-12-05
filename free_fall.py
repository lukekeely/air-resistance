import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
B = 1.6e-4  # N s/m^2
C = 0.25
g = 9.81  # Acceleration due to gravity (m/s^2)
dt = 1e-5  # Time step (s)
D = 1e-2  # Diameter of the particle
H = 5  # Initial height (m)
densities = np.linspace(500, 5000, 20)  # Range of densities (kg/m^3) for 20 particles

def calculate_mass(density):
    radius = D / 2
    volume = (4/3) * np.pi * radius**3
    return density * volume

def dust_particle_motion_with_air_resistance(Vy):
    return -g + (B * D * Vy + C * D**2 * Vy**2)/mass

def simulate_fall_with_air_resistance():
    Vy = 0
    Y = H
    time = 0

    while Y > 0:
        Vy += dt * dust_particle_motion_with_air_resistance(Vy)
        Y += Vy * dt
        time += dt

    return time

def simulate_fall_without_air_resistance():
    return np.sqrt(2 * H / g)

# Prepare directory for plots
plot_dir = 'falling_particle_plots'
os.makedirs(plot_dir, exist_ok=True)

# Calculate time to reach the ground for different densities
times_with_air_resistance = []
times_without_air_resistance = []
masses = []

for density in densities:
    mass = calculate_mass(density)
    time_with_air = simulate_fall_with_air_resistance()
    time_without_air = simulate_fall_without_air_resistance()
    times_with_air_resistance.append(time_with_air)
    times_without_air_resistance.append(time_without_air)
    masses.append(mass)

# With Air Resistance
plt.figure(figsize=(5, 5), dpi=300)
plt.plot(masses, times_with_air_resistance, label='With Air Resistance', color='blue')
plt.scatter(masses, times_with_air_resistance, color='blue')
plt.xlabel('Mass (kg)')
plt.ylabel('Time (s)')
plt.title('Time to Reach the Ground vs. Mass (With Air Resistance)')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(plot_dir, 'Time_to_Ground_With_Air_Resistance.png'))
plt.savefig(os.path.join(plot_dir, 'Time_to_Ground_With_Air_Resistance.pdf'))
plt.show()

# Without Air Resistance
plt.figure(figsize=(5, 5), dpi=300)
plt.plot(masses, times_without_air_resistance, label='Without Air Resistance', color='green')
plt.scatter(masses, times_without_air_resistance, color='green')
plt.xlabel('Mass (kg)')
plt.ylabel('Time (s)')
plt.title('Time to Reach the Ground vs. Mass (Without Air Resistance)')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(plot_dir, 'Time_to_Ground_Without_Air_Resistance.png'))
plt.savefig(os.path.join(plot_dir, 'Time_to_Ground_Without_Air_Resistance.pdf'))
plt.show()

data = {
    'Mass (kg)': masses,
    'Time with Air Resistance (s)': times_with_air_resistance,
    'Time without Air Resistance (s)': times_without_air_resistance
}
df = pd.DataFrame(data)
print(df)
data_filename = os.path.join(plot_dir, 'particle_fall_times.csv')
df.to_csv(data_filename, index=False)
