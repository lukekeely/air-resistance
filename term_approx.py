import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Constants
B = 1.6e-4  # N s/m^2
C = 0.25  # N s^2/m^4
THRESHOLD_PERCENT = 5  # percentage below which a term is considered negligible

def create_subfolder():
    subfolder_name = "term_approx"
    if not os.path.exists(subfolder_name):
        os.makedirs(subfolder_name)
    return subfolder_name

def plot_case(D, V_max, case_name, DxV_case, subfolder):
    sns.set_theme()
    V = np.linspace(0, V_max * 1.5, 100)  
    bV = B * D * V
    cV2 = C * D**2 * V**2
    total = bV + cV2
    DxV = D * V

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(DxV, bV, 'b-', label='$bV$')
    plt.plot(DxV, cV2, 'r-', label='$cV^2$')
    plt.plot(DxV, total, color='gray', linestyle='-', label='$bV + cV^2$')
    plt.plot(DxV, THRESHOLD_PERCENT/100 * bV, 'b--', label=f'{THRESHOLD_PERCENT}% of $bV$')
    plt.plot(DxV, THRESHOLD_PERCENT/100 * cV2, 'r--', label=f'{THRESHOLD_PERCENT}% of $cV^2$')
    plt.fill_between(DxV, bV, THRESHOLD_PERCENT/100 * bV, color='blue', alpha=0.3)
    plt.fill_between(DxV, cV2, THRESHOLD_PERCENT/100 * cV2, color='red', alpha=0.3)

    plt.axvline(DxV_case, color='gray', linestyle='--', label=f'DxV for {case_name} = {DxV_case:.2e}')

    plt.xlim(0, V_max * D * 1.5)  
    plt.ylim(0, max(max(bV), max(cV2)) * 1.2)

    plt.xlabel('$D \\times V$ (m$^2$/s)')
    plt.ylabel('Force Terms (N)')
    plt.title(f'{case_name}: Scaling of Air Resistance Terms with $D \\times V$')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(subfolder, f"{case_name}_force_terms_scaling.png"))
    plt.savefig(os.path.join(subfolder, f"{case_name}_force_terms_scaling.pdf"))
    plt.show()


subfolder = create_subfolder()
cases = {
    'Baseball': {'D': 0.07, 'V_max': 10},
    'Oil Drop': {'D': 1.5e-6, 'V_max': 1e-4},
    'Raindrop': {'D': 0.001, 'V_max': 3},
}


data_to_save = []

for name, params in cases.items():
    D, V_max = params['D'], params['V_max']
    DxV_case = D * V_max
    plot_case(D, V_max, name, DxV_case, subfolder)

    V = np.linspace(0, V_max, 100)
    bV = B * D * V
    cV2 = C * D**2 * V**2
    idx = np.argmin(np.abs(D * V - DxV_case))

    if min(bV[idx], cV2[idx]) > max(THRESHOLD_PERCENT/100 * bV[idx], THRESHOLD_PERCENT/100 * cV2[idx]):
        terms_to_use = "Both"
    else:
        terms_to_use = "Quadratic" if bV[idx] < cV2[idx] else "Linear"

    data_to_save.append({
        'Case': name,
        'D': D,
        'V_max': V_max,
        'DxV_case': DxV_case,
        'Terms to Use': terms_to_use,
        'bV_at_DxV': bV[idx],
        'cV2_at_DxV': cV2[idx]
    })

df = pd.DataFrame(data_to_save)
print(df)
csv_file = os.path.join(subfolder, "specific_cases.csv")
df.to_csv(csv_file, index=False)
print(f"Data saved to: {csv_file}")