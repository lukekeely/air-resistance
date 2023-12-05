import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
B = 1.6e-4  # N s/m^2
C = 0.25    # N s/m^2
g = 9.81  # Acceleration due to gravity (m/s^2)
dt = 1e-3  # Time step (s)
D = 1e-2  # Constant diameter for all cases
tolerance = 1e-5
t_max = 60  # Maximum simulation time (s)
nsteps = int(t_max / dt)

# Function to calculate air resistance
def calculate_air_resistance(density, velocity):
    radius = D / 2
    volume = (4/3) * np.pi * radius**3
    mass = density * volume
    b = B * D
    c = C * D**2
    return (b * velocity + c * velocity*velocity)/mass

# cases
cases = {
    'Very Light Particle': {'density': 1e3, 'Vys': np.zeros(nsteps), 'terminal_velocity': None, 'time_to_terminal': None},
    'Light Particle': {'density': 5e3, 'Vys': np.zeros(nsteps), 'terminal_velocity': None, 'time_to_terminal': None},
    'Medium Particle': {'density': 1e4, 'Vys': np.zeros(nsteps), 'terminal_velocity': None, 'time_to_terminal': None},
    'Heavy Particle': {'density': 5e4, 'Vys': np.zeros(nsteps), 'terminal_velocity': None, 'time_to_terminal': None},
}

for step in range(nsteps):
    time = step * dt
    for name, properties in cases.items():
        Vy = properties['Vys'][step-1] if step > 0 else 0
        k1_Vy = dt * (-g + calculate_air_resistance(properties['density'], Vy))
        k2_Vy = dt * (-g + calculate_air_resistance(properties['density'], (Vy+ k1_Vy)))
        Vy_new = Vy + (k1_Vy + k2_Vy) / 2
        properties['Vys'][step] = Vy_new
        if properties['terminal_velocity'] is None:
            if abs(Vy_new - Vy) <= tolerance:
                properties['terminal_velocity'] = Vy_new
                properties['time_to_terminal'] = time

plot_dir = 'particle_motion_plots'
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(6, 4), dpi=300)
times = np.arange(0, dt * nsteps, dt)
line_colors = {}

for title, properties in cases.items():
    line, = plt.plot(times, properties['Vys'], label=title)
    line_colors[title] = line.get_color()
    if properties['terminal_velocity'] is not None:
        plt.axhline(y=properties['terminal_velocity'], color=line_colors[title], linestyle='--')
        plt.plot(properties['time_to_terminal'], properties['terminal_velocity'], 'o', color=line_colors[title])
    df = pd.DataFrame({'Time (s)': times, 'Velocity (m/s)': properties['Vys']})
    filename = f"{plot_dir}/{title.replace(' ', '_')}_data.csv"
    df.to_csv(filename, index=False)
    print(f"{title} - Density: {properties['density']} kg/m^3, Terminal Velocity: {properties['terminal_velocity']} m/s, Time to Terminal: {properties['time_to_terminal']} s")

plt.xlabel('Time (s)')
plt.ylabel('Vy (Velocity)')
plt.title('Mass Terminal Velocity Comparison')
plt.legend()
plt.grid(True)
plt.savefig(f"{plot_dir}/combined_particle_motion_plot.png")
plt.savefig(f"{plot_dir}/combined_particle_motion_plot.pdf")
plt.show()
