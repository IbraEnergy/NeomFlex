import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Read the data
agent1_data = pd.read_csv('testing_record_agent_1.csv')
agent2_data = pd.read_csv('testing_record_agent_2.csv')

# Process sequential SOC data
def process_sequential_soc(df, soc_column, num_days=15):
    sequential_data = []
    for day in range(num_days):
        day_data = df[df['day'] == day][soc_column]
        sequential_data.extend(day_data)
    return sequential_data

# Convert RGB to matplotlib color format (0-1 scale)
color_agent1 = (79/255, 113/255, 190/255)  # For agent 1 battery
color_agent2 = (234/255, 51/255, 35/255)   # For agent 2 batteries

# Set up the plot
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.figure(figsize=(7, 2.5)), plt.gca()

# Create x-axis with all time steps
total_steps = 48 * 15  # 15 days with 48 steps each
time_steps = np.arange(total_steps)

# Process and plot SOC data
soc_agent1 = process_sequential_soc(agent1_data, 'battery_23_soc')
soc_agent2_battery1 = process_sequential_soc(agent2_data, 'battery_7_soc')  # Battery at bus 9
soc_agent2_battery2 = process_sequential_soc(agent2_data, 'battery_14_soc')  # Battery at bus 16

# Plot the SOC profiles
ax.plot(time_steps, soc_agent1, color=color_agent1, linewidth=1.2, 
        label='Agent 1 Battery')
ax.plot(time_steps, soc_agent2_battery1, color=color_agent2, linewidth=1.2,
        label='Agent 2 Battery 1')
ax.plot(time_steps, soc_agent2_battery2, color=color_agent2, linewidth=1.2,
        linestyle='--', label='Agent 2 Battery 2')

# Customize the plot
ax.set_xlabel('Time Steps', fontsize=10)
ax.set_ylabel('State of Charge', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=8, ncol=3, bbox_to_anchor=(0.5, 1.15), loc='center')

# Add day separators and labels
# for day in range(15):
#     step = day * 48
#     ax.axvline(x=step, color='gray', linestyle='--', alpha=0.5)
#     ax.text(step + 24, 0.35, f'Day {day}', 
#             horizontalalignment='center', fontsize=8)

# Set axis limits
ax.set_xlim(0, total_steps-1)
ax.set_ylim(0.2, 2)

# Tight layout to maximize space usage
plt.tight_layout()

# Save as PDF
with PdfPages('battery_soc_profile.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight')

plt.close()