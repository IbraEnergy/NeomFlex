import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_episode_analysis(episode_num, save_dir='results/data'):
    """
    Analyze and visualize a single episode's data.
    
    Args:
        episode_num: Episode number to analyze
        save_dir: Directory containing episode data
    """
    # Read data
    df = pd.read_csv(f'{save_dir}/episode_{episode_num}.csv')
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))
    
    # 1. Voltage Profile Plot
    voltage_cols = [col for col in df.columns if col.startswith('voltage_bus_')]
    voltage_data = df[voltage_cols].values
    
    mean_voltage = np.mean(voltage_data, axis=0)
    min_voltage = np.min(voltage_data, axis=0)
    max_voltage = np.max(voltage_data, axis=0)
    
    ax1.plot(range(len(voltage_cols)), mean_voltage, 'b-', label='Mean', linewidth=2)
    ax1.fill_between(range(len(voltage_cols)), min_voltage, max_voltage, 
                     color='b', alpha=0.2, label='Min/Max Range')
    ax1.axhline(y=0.95, color='r', linestyle='--', label='Lower Limit')
    ax1.axhline(y=1.05, color='r', linestyle='--', label='Upper Limit')
    ax1.set_xlabel('Bus Number')
    ax1.set_ylabel('Voltage (p.u.)')
    ax1.set_title(f'Voltage Profile - Episode {episode_num}')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Net Load and Actions Plot
    time_steps = range(len(df))
    LLoad = df['load']
    net_load = df['load'] - df['solar']

    
    # Plot net load
    ax2.plot(time_steps, net_load, 'k-', label='Net Load', linewidth=2)
    
    # # Plot battery powers
    # bat_cols = [col for col in df.columns if col.startswith('bat_') and col.endswith('_power')]
    # for col in bat_cols:
    #     bus = col.split('_')[1]
    #     ax2.plot(time_steps, df[col], '--', label=f'Battery {bus}')
    
    # Plot PV reactive power
    pv_cols = [col for col in df.columns if col.startswith('pv_') and col.endswith('_q')]
    for col in pv_cols:
        bus = col.split('_')[1]
        ax2.plot(time_steps, df[col], ':', label=f'PV {bus} VAR')
    
    ax2.set_xlabel('Time Step (30-min intervals)')
    ax2.set_ylabel('Power (MW/MVAR)')
    ax2.set_title('Net Load and Control Actions')
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1))
    
    # 3. Net Load and SOC Plot
    ax3.plot(time_steps, net_load, 'k-', label='Net Load', linewidth=2)
    ax3.plot(time_steps, LLoad, 'b-', label='Load', linewidth=2)
    
    # Plot battery SOC
    soc_cols = [col for col in df.columns if col.startswith('bat_') and col.endswith('_soc')]
    for col in soc_cols:
        bus = col.split('_')[1]
        ax3.plot(time_steps, df[col], '--', label=f'Battery {bus} SOC')
    
    ax3.set_xlabel('Time Step (30-min intervals)')
    ax3.set_ylabel('Power (MW) / SOC (MWh)')
    ax3.set_title('Net Load and Battery SOC Profiles')
    ax3.grid(True)
    ax3.legend(bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage
    episode_number = 4977  # Change this to analyze different episodes
    
    # Create plot
    fig = plot_episode_analysis(episode_number)
    
    # Save plot
    plt.savefig(f'episode_{episode_number}_analysis.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"Analysis for episode {episode_number} completed and saved.")