#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 04:21:09 2024

@author: ibrahimalsaleh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Read the data
agent1_data = pd.read_csv('testing_record_agent_1.csv')
agent2_data = pd.read_csv('testing_record_agent_2.csv')
agent3_data = pd.read_csv('testing_record_agent_3.csv')

# Process sequential VAR data
def process_sequential_var(df, var_column, num_days=15):
    sequential_data = []
    for day in range(num_days):
        day_data = df[df['day'] == day][var_column]
        sequential_data.extend(day_data)
    return sequential_data

# Convert RGB to matplotlib color format (0-1 scale)
color_agent1 = (79/255, 113/255, 190/255)   # Agent 1 PV
color_agent2 = (234/255, 51/255, 35/255)    # Agent 2 PV
color_agent3 = (78/255, 173/255, 91/255)    # Agent 3 PV

# Set up the plot
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.figure(figsize=(7, 2.5)), plt.gca()

# Create x-axis with all time steps
total_steps = 48 * 15  # 15 days with 48 steps each
time_steps = np.arange(total_steps)

# Process and plot VAR data
var_agent1 = process_sequential_var(agent1_data, 'pv_20_q')
var_agent2 = process_sequential_var(agent2_data, 'pv_16_q')
var_agent3 = process_sequential_var(agent3_data, 'pv_31_q')

# Plot the VAR profiles
ax.plot(time_steps, var_agent1, color=color_agent1, linewidth=0.8, 
        label='Agent 1 PV')
ax.plot(time_steps, var_agent2, color=color_agent2, linewidth=0.8,
        label='Agent 2 PV')
ax.plot(time_steps, var_agent3, color=color_agent3, linewidth=0.8,
        label='Agent 3 PV')

# Customize the plot
ax.set_xlabel('Time Steps', fontsize=10)
ax.set_ylabel('Reactive Power (VAR)', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=8, ncol=3, bbox_to_anchor=(0.5, 1.15), loc='center')

# Add day separators and labels
for day in range(15):
    step = day * 48
    ax.axvline(x=step, color='gray', linestyle='--', alpha=0.5)
    ax.text(step + 24, -0.8, f'Day {day}', 
            horizontalalignment='center', fontsize=8)

# Set axis limits
ax.set_xlim(0, total_steps-1)
ax.set_ylim(-1.2, 1.2)

# Tight layout to maximize space usage
plt.tight_layout()

# Save as PDF
with PdfPages('pv_var_profile.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight')

plt.close()