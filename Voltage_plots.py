import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Read the data
data = pd.read_csv('testing_record_agent_1.csv')

# Function to get voltage columns
def get_voltage_columns(df):
    return [col for col in df.columns if 'voltage_bus_' in col]

# Process data for sequential days
def process_sequential_voltages(df, voltage_cols, num_days=15):
    sequential_data = {
        'max': [],
        'min': [],
        'mean': []
    }
    
    for day in range(num_days):
        day_data = df[df['day'] == day][voltage_cols]
        sequential_data['max'].extend(day_data.max(axis=1))
        sequential_data['min'].extend(day_data.min(axis=1))
        sequential_data['mean'].extend(day_data.mean(axis=1))
    
    return sequential_data

# Set up the plot
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.figure(figsize=(7, 2.5)), plt.gca()

# Get voltage columns
voltage_cols = get_voltage_columns(data)

# Process sequential statistics
sequential_stats = process_sequential_voltages(data, voltage_cols)

# Create x-axis with all time steps
total_steps = 48 * 15  # 15 days with 48 steps each
time_steps = np.arange(total_steps)

# Plot the data
ax.fill_between(time_steps, 
                sequential_stats['max'],
                sequential_stats['min'],
                alpha=0.5,
                color='lightblue')
ax.plot(time_steps,
        sequential_stats['mean'],
        color='navy',
        linewidth=0.8)

# Customize the plot
ax.set_xlabel('Time Steps', fontsize=10)
ax.set_ylabel('Voltage (p.u.)', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)

# Add day separators and labels
for day in range(15):
    step = day * 48
    ax.axvline(x=step, color='gray', linestyle='--', alpha=0.5)
    ax.text(step + 24, 0.965, f'Day {day}', 
            horizontalalignment='center', fontsize=8)

# Set axis limits
ax.set_xlim(0, total_steps-1)
ax.set_ylim(0.96, 1.02)

# Tight layout to maximize space usage
plt.tight_layout()

# Save as PDF
with PdfPages('voltage_profile_sequential.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight')

plt.close()