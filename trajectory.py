import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Constants
B = 1.6e-4  # N s/m^2 (linear drag coefficient)
C = 0.25    # N s^2/m^4 (quadratic drag coefficient)
g = 9.81    # Acceleration due to gravity (m/s^2)
dt = 1e-3   # Time step (s)
D = 1e-4    # Diameter of the projectile
density = 8e4  # Density of the projectile (kg/m^3)
V = 10      # Launch velocity (m/s)
angles_degrees = [30, 45, 60]  # Small, medium, and large angles

radius = D / 2
volume = (4/3) * np.pi * radius**3
mass = density * volume


def projectile_motion(Vx, Vy, air_resistance_type, theta):
    if air_resistance_type == 'none':
        ax, ay = 0, -g
    else:
        bV = B * D if air_resistance_type in ['linear', 'both'] else 0
        cV2 = C * D**2 if air_resistance_type in ['quadratic', 'both'] else 0
        total_velocity = np.sqrt(Vx**2 + Vy**2)
        air_resistance_force = (bV + cV2 * total_velocity) / mass
        ax = -air_resistance_force * Vx
        ay = -g - air_resistance_force * Vy
    return ax, ay

def simulate_projectile_motion(air_resistance_type, theta):
    Vx = V * np.cos(theta)
    Vy = V * np.sin(theta)
    X, Y = 0, 0
    trajectory = []
    while Y >= 0:
        ax, ay = projectile_motion(Vx, Vy, air_resistance_type, theta)
        Vx += ax * dt
        Vy += ay * dt
        X += Vx * dt
        Y += Vy * dt
        trajectory.append((X, Y))
    return trajectory

output_dir = "trajectory"
os.makedirs(output_dir, exist_ok=True)

# Calculate the maximum X and Y values across all trajectories
max_X, max_Y = 0, 0
for angle in angles_degrees:
    theta = np.radians(angle)
    for resistance_type in ['none', 'linear', 'quadratic', 'both']:
        trajectory = simulate_projectile_motion(resistance_type, theta)
        max_X = max(max_X, max(trajectory, key=lambda x: x[0])[0])
        max_Y = max(max_Y, max(trajectory, key=lambda x: x[1])[1])

max_X += max_X / 100
max_Y += max_Y / 100

# Create separate plots for each launch angle
for angle in angles_degrees:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    theta = np.radians(angle)
    trajectory_data = []

    for resistance_type in ['none', 'linear', 'quadratic', 'both']:
        trajectory = simulate_projectile_motion(resistance_type, theta)

        for x, y in trajectory:
            trajectory_data.append([angle, resistance_type, x, y])
        ax.plot(*zip(*trajectory), label=f'{resistance_type.capitalize()} Air Resistance',
                linestyle='-' if resistance_type == 'both' else '--' if resistance_type == 'quadratic' else ':')
        ax.set_xlim(0, max_X)
        ax.set_ylim(0, max_Y)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True)

    df = pd.DataFrame(trajectory_data, columns=['Angle', 'Air Resistance', 'X', 'Y'])
    df.to_csv(f"{output_dir}/trajectory_data_angle_{angle}.csv", index=False)

    plot_filename = f"{output_dir}/trajectory_plot_angle_{angle}"
    plt.savefig(f"{plot_filename}.png")
    plt.savefig(f"{plot_filename}.pdf")
    plt.show()
