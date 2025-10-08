import os
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple
import torch

import logging
import matplotlib.pyplot as plt
from grid_env_14 import GridEnv
from sac_agent_12 import SACAgent

class GridTester:
    def __init__(self, env, trained_agents, test_data_path='Testing.csv'):
        self.env = env
        self.agents = trained_agents
        self.test_data = pd.read_csv(test_data_path)
        self.total_days = len(self.test_data) // 48
        
        # Verify agents match environment dimensions
        for agent_id in env.territories:
            test_obs = env._get_obs(agent_id, np.ones(len(env.net.bus)))
            expected_dim = len(test_obs)
            actual_dim = trained_agents[agent_id].actor.fc1.in_features
            assert expected_dim == actual_dim, f"Dimension mismatch for {agent_id}: expected {expected_dim}, got {actual_dim}"
        
        # Initialize base environment for no-control comparison
        self.base_env = self._initialize_base_env()
        
        # Results storage
        self.results = {
            'controlled': {
                'voltages': [],
                'losses': [],
                'soc': [],
                'pv_var': [],
                'rewards': []
            },
            'base': {
                'voltages': [],
                'losses': []
            }
        }
        
    def _initialize_base_env(self):
        """Create a basic environment without VAR/Battery control."""
        base_env = pn.case33bw()
        # Copy basic network structure
        base_env.bus = self.env.net.bus.copy()
        base_env.line = self.env.net.line.copy()
        base_env.load = self.env.net.load.copy()
        
        # Add PV generators but without VAR control
        for pv_bus in self.env.pv_buses:
            pp.create_sgen(
                base_env,
                bus=pv_bus,
                p_mw=0.0,
                q_mvar=0.0,
                name=f"PV_{pv_bus}",
                controllable=False
            )
        
        return base_env
    
    
        
    def run_day_comparison(self, day_number: int) -> Dict:
        """Run comparison between controlled and base case for a specific day."""
        start_idx = day_number * 48
        end_idx = start_idx + 48
        
        day_data = self.test_data.iloc[start_idx:end_idx]
        
        # Run controlled case
        controlled_results = self._run_controlled_case(day_data)
        
        # Run base case
        base_results = self._run_base_case(day_data)
        
        return {
            'controlled': controlled_results,
            'base': base_results,
            'time_steps': list(range(48)),
            'day_data': day_data
        }
        
    def _run_controlled_case(self, day_data: pd.DataFrame) -> Dict:
        results = {agent_id: {
            'voltages': [],
            'losses': [],
            'soc': [],
            'pv_var': [],
            'rewards': []
        } for agent_id in self.agents}
        
        obs = self.env.reset()
        
        for step in range(48):
            actions = {}
            for agent_id, agent in self.agents.items():
                actions[agent_id] = agent.select_action(obs[agent_id], evaluate=True)
            
            next_obs, rewards, done, info = self.env.step(actions)
            
            # Record territorial results
            for agent_id in self.agents:
                territory_buses = self.env.territories[agent_id]
                results[agent_id]['voltages'].append(self.env.net.res_bus.vm_pu.values[territory_buses])
                
                # Calculate territorial losses
                territory_lines = [idx for idx, row in self.env.net.line.iterrows() 
                                 if row.from_bus in territory_buses or row.to_bus in territory_buses]
                territorial_losses = self.env.net.res_line.pl_mw[territory_lines].sum()
                results[agent_id]['losses'].append(territorial_losses)
                
                # Record territorial assets
                agent_batteries = [bus for bus in self.env.bat_buses 
                                 if self.env.bat_buses[bus] == agent_id]
                results[agent_id]['soc'].append({bus: self.env.bat_soc[bus] 
                                               for bus in agent_batteries})
                
                agent_pvs = [bus for bus in self.env.pv_buses 
                            if self.env.pv_buses[bus] == agent_id]
                results[agent_id]['pv_var'].append({bus: self.env.net.sgen.at[self.env.pv_sgen_indices[bus], 'q_mvar'] 
                                                  for bus in agent_pvs})
                
                results[agent_id]['rewards'].append(rewards[agent_id])
            
            obs = next_obs
        
        return results
        
    def _run_base_case(self, day_data: pd.DataFrame) -> Dict:
        """Run a day without VAR/Battery control."""
        results = {
            'voltages': [],
            'losses': []
        }
        
        for step in range(48):
            # Update loads
            load_factor = day_data.iloc[step]['Demand']
            solar_factor = day_data.iloc[step]['Solar']
            
            # Set loads
            self.base_env.load.p_mw = self.env.base_p * load_factor
            self.base_env.load.q_mvar = self.env.base_q * load_factor
            
            # Set PV real power only
            for pv_bus in self.env.pv_buses:
                sgen_idx = [i for i, name in enumerate(self.base_env.sgen.name) 
                           if name == f"PV_{pv_bus}"][0]
                self.base_env.sgen.at[sgen_idx, 'p_mw'] = self.env.pv_cap * solar_factor
            
            # Run power flow
            pp.runpp(self.base_env)
            
            # Record results
            results['voltages'].append(self.base_env.res_bus.vm_pu.values)
            results['losses'].append(self.base_env.res_line.pl_mw.sum())
            
        return results

class ResultVisualizer:
    def __init__(self, save_dir='results/testing'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Simple matplotlib style settings
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.grid': True,
            'grid.color': '0.8',
            'grid.linestyle': '--',
            'lines.linewidth': 2,
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
        
        # Set color scheme
        self.colors = {
            'controlled': {'line': '#1f77b4', 'fill': '#1f77b4'},  # Blue
            'base': {'line': '#d62728', 'fill': '#d62728'},        # Red
            'battery': ['#2ca02c', '#ff7f0e', '#9467bd'],          # Green, Orange, Purple
            'pv': ['#8c564b', '#e377c2', '#7f7f7f']               # Brown, Pink, Gray
        }
        
    def plot_day_comparison(self, results: Dict, day_number: int):
        """Create comprehensive plots for day comparison."""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 20))
        gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
        
        # 1. Voltage Profile Plot
        self._plot_voltage_comparison(fig.add_subplot(gs[0]), results, day_number)
        
        # 2. Losses Comparison
        self._plot_losses_comparison(fig.add_subplot(gs[1]), results)
        
        # 3. Battery SOC Plot
        self._plot_battery_soc(fig.add_subplot(gs[2]), results)
        
        # 4. PV VAR Support
        self._plot_pv_var_support(fig.add_subplot(gs[3]), results)
        
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.save_dir, f'day_{day_number}_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")  # Add this line to confirm saving
        
        # Display the plot
        plt.show()
        
        plt.close()
        
    def _plot_voltage_comparison(self, ax, results, day_number):
        """Plot voltage profiles with envelopes."""
        # Combine voltages from all territories for controlled case
        controlled_voltages_list = []
        for agent_id in results['controlled']:
            controlled_voltages_list.extend([np.array(results['controlled'][agent_id]['voltages'])])
        controlled_voltages = np.concatenate(controlled_voltages_list, axis=1)
        
        # Base case remains the same
        base_voltages = np.array(results['base']['voltages'])
        time_steps = np.arange(48) / 2
        
        # Compute statistics
        controlled_max = controlled_voltages.max(axis=1)
        controlled_min = controlled_voltages.min(axis=1)
        controlled_mean = controlled_voltages.mean(axis=1)
        
        base_max = base_voltages.max(axis=1)
        base_min = base_voltages.min(axis=1)
        base_mean = base_voltages.mean(axis=1)
        
        # Plot
        ax.fill_between(time_steps, controlled_min, controlled_max, alpha=0.3, 
                       color='blue', label='Controlled Range')
        ax.plot(time_steps, controlled_mean, 'b-', label='Controlled Mean')
        
        ax.fill_between(time_steps, base_min, base_max, alpha=0.3, 
                       color='red', label='Base Range')
        ax.plot(time_steps, base_mean, 'r-', label='Base Mean')
        
        ax.axhline(y=1.05, color='k', linestyle='--', alpha=0.5)
        ax.axhline(y=0.95, color='k', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Voltage (p.u.)')
        ax.set_title(f'Voltage Profiles Comparison - Day {day_number}')
        ax.legend()
        ax.grid(True)
        
    def _plot_losses_comparison(self, ax, results):
        """Plot network losses comparison."""
        # Combine territorial losses for controlled case
        territorial_losses = [results['controlled'][agent_id]['losses'] for agent_id in results['controlled']]
        controlled_losses = np.sum(territorial_losses, axis=0)  # Sum across territories
        
        # Base case remains the same
        base_losses = results['base']['losses']
        time_steps = np.arange(48) / 2
        
        ax.plot(time_steps, controlled_losses, 'b-', label='With Control')
        ax.plot(time_steps, base_losses, 'r-', label='Without Control')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Network Losses (MW)')
        ax.set_title('Network Losses Comparison')
        ax.legend()
        ax.grid(True)
        
    def _plot_battery_soc(self, ax, results):
       """Plot battery SOC evolution per agent."""
       time_steps = np.arange(48) / 2
       
       # Plot each agent's batteries
       for agent_id in results['controlled']:
           soc_data = results['controlled'][agent_id]['soc']
           if soc_data[0]:  # Only plot if agent has batteries
               for bus in soc_data[0].keys():
                   soc_values = [step[bus] for step in soc_data]
                   ax.plot(time_steps, soc_values, label=f'Battery {bus} (Agent {agent_id})')
               
       ax.set_xlabel('Time (hours)')
       ax.set_ylabel('State of Charge (MWh)')
       ax.set_title('Battery State of Charge by Territory')
       ax.legend()
       ax.grid(True)
    
    def _plot_pv_var_support(self, ax, results):
       """Plot PV reactive power support per agent."""
       time_steps = np.arange(48) / 2
       
       # Plot each agent's PV VAR support
       for agent_id in results['controlled']:
           var_data = results['controlled'][agent_id]['pv_var']
           if var_data[0]:  # Only plot if agent has PVs
               for bus in var_data[0].keys():
                   var_values = [step[bus] for step in var_data]
                   ax.plot(time_steps, var_values, label=f'PV {bus} (Agent {agent_id})')
               
       ax.set_xlabel('Time (hours)')
       ax.set_ylabel('Reactive Power (MVAr)')
       ax.set_title('PV Reactive Power Support by Territory')
       ax.legend()
       ax.grid(True)

def test_grid_performance(env, trained_agents, test_data_path='Testing.csv', day_to_analyze=0):
    print(f"Starting analysis for day {day_to_analyze}")
    
    # Initialize tester and visualizer
    tester = GridTester(env, trained_agents, test_data_path)
    visualizer = ResultVisualizer()
    
    print("Running day comparison...")
    # Run comparison for specific day
    results = tester.run_day_comparison(day_to_analyze)
    
    print("Creating visualizations...")
    # Create visualizations
    visualizer.plot_day_comparison(results, day_to_analyze)
    
    print("Testing completed!")
    return results


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler('testing_results.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Create environment
        env = GridEnv(data_path='Testing.csv')
        logging.info("Created testing environment")
        
        # Load trained agents
        trained_agents = {}
        results_dir = 'results/final_results'
        
        for agent_id in env.territories:
            # Create agent
            agent = SACAgent(
                state_dim=env.observation_spaces[agent_id].shape[0],
                action_dim=env.action_spaces[agent_id].shape[0]
            )
            
            # Load trained weights
            checkpoint_path = os.path.join(results_dir, f'{agent_id}.pt')
            agent.load(checkpoint_path)
            trained_agents[agent_id] = agent
            logging.info(f"Loaded trained agent: {agent_id}")
        
        # Create tester and visualizer
        tester = GridTester(env, trained_agents)
        visualizer = ResultVisualizer(save_dir='results/testing')
        
        # Test multiple days
        test_days = [0, 1, 2]  # Can test multiple days
        all_results = {}
        
        for day in test_days:
            logging.info(f"Testing day {day}")
            results = tester.run_day_comparison(day)
            visualizer.plot_day_comparison(results, day)
            all_results[day] = results
            
        # Calculate and log summary statistics
        for day in test_days:
            results = all_results[day]
            controlled_v = np.array(results['controlled']['voltages'])
            base_v = np.array(results['base']['voltages'])
            
            logging.info(f"\nDay {day} Statistics:")
            logging.info("Controlled Case:")
            logging.info(f"  Max Voltage: {controlled_v.max():.3f}")
            logging.info(f"  Min Voltage: {controlled_v.min():.3f}")
            logging.info(f"  Mean Voltage: {controlled_v.mean():.3f}")
            logging.info(f"  Voltage Violations: {np.sum((controlled_v < 0.95) | (controlled_v > 1.05))}")
            
            logging.info("Base Case:")
            logging.info(f"  Max Voltage: {base_v.max():.3f}")
            logging.info(f"  Min Voltage: {base_v.min():.3f}")
            logging.info(f"  Mean Voltage: {base_v.mean():.3f}")
            logging.info(f"  Voltage Violations: {np.sum((base_v < 0.95) | (base_v > 1.05))}")
            
        logging.info("\nTesting completed successfully")
        
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        raise
        
        
        
if __name__ == "__main__":
    from grid_env_14 import GridEnv
    from sac_agent_12 import SACAgent
    import matplotlib
    #matplotlib.use('Agg')  # For non-interactive backend
    
    # Create environment first
    env = GridEnv(data_path='Testing.csv')
    
    # Create agents with proper dimensions
    trained_agents = {}
    for agent_id in env.territories:
        # Get actual observation size from test observation
        test_obs = env._get_obs(agent_id, np.ones(len(env.net.bus)))
        state_dim = len(test_obs)  # Use actual observation size
        action_dim = env.action_spaces[agent_id].shape[0]
        
        print(f"Creating {agent_id} network with state_dim={state_dim}, action_dim={action_dim}")
        
        # Create agent
        agent = SACAgent(
            state_dim=state_dim,  # Use actual size
            action_dim=action_dim,
            hidden_dim=256
        )
        
        # Load trained weights
        checkpoint_path = os.path.join('results/final_results', f'{agent_id}.pt')
        print(f"Loading weights from: {checkpoint_path}")
        agent.load(checkpoint_path)
        trained_agents[agent_id] = agent
    
    # Run testing
    results = test_grid_performance(env, trained_agents, 'Testing.csv', day_to_analyze=2)
    
#     #%% Multiple day testing
    
# def test_grid_performance(env, trained_agents, test_data_path='Testing.csv', days_to_analyze=None):
#     """
#     Test grid performance for multiple days
#     Args:
#         days_to_analyze: List of days to analyze. If None, tests all available days
#     """
#     if days_to_analyze is None:
#         days_to_analyze = list(range(env.total_days))
    
#     print(f"Starting analysis for days: {days_to_analyze}")
    
#     # Initialize tester and visualizer
#     tester = GridTester(env, trained_agents, test_data_path)
#     visualizer = ResultVisualizer()
    
#     # Store results for all days
#     all_results = {}
    
#     for day in days_to_analyze:
#         print(f"\nAnalyzing day {day}")
#         # Run comparison for specific day
#         results = tester.run_day_comparison(day)
        
#         # Create visualizations
#         visualizer.plot_day_comparison(results, day)
        
#         # Store results
#         all_results[day] = results
        
#         # Print summary statistics for this day
#         controlled_v = np.array(results['controlled']['voltages'])
#         base_v = np.array(results['base']['voltages'])
        
#         print(f"\nDay {day} Statistics:")
#         print("Controlled Case:")
#         print(f"  Max Voltage: {controlled_v.max():.3f}")
#         print(f"  Min Voltage: {controlled_v.min():.3f}")
#         print(f"  Mean Voltage: {controlled_v.mean():.3f}")
#         print(f"  Voltage Violations: {np.sum((controlled_v < 0.95) | (controlled_v > 1.05))}")
        
#         print("Base Case:")
#         print(f"  Max Voltage: {base_v.max():.3f}")
#         print(f"  Min Voltage: {base_v.min():.3f}")
#         print(f"  Mean Voltage: {base_v.mean():.3f}")
#         print(f"  Voltage Violations: {np.sum((base_v < 0.95) | (base_v > 1.05))}")
    
#     print("\nTesting completed!")
#     return all_results

# if __name__ == "__main__":
#     from grid_env_14 import GridEnv
#     from sac_agent_12 import SACAgent
#     import matplotlib
#     matplotlib.use('Agg')  # For non-interactive backend
    
#     # Create environment first
#     env = GridEnv(data_path='Testing.csv')
    
#     # Create agents with proper dimensions
#     trained_agents = {}
#     for agent_id in env.territories:
#         # Get actual observation size from test observation
#         test_obs = env._get_obs(agent_id, np.ones(len(env.net.bus)))
#         state_dim = len(test_obs)  # Use actual observation size
#         action_dim = env.action_spaces[agent_id].shape[0]
        
#         print(f"Creating {agent_id} network with state_dim={state_dim}, action_dim={action_dim}")
        
#         # Create agent
#         agent = SACAgent(
#             state_dim=state_dim,
#             action_dim=action_dim,
#             hidden_dim=256
#         )
        
#         # Load trained weights
#         checkpoint_path = os.path.join('results/final_results', f'{agent_id}.pt')
#         print(f"Loading weights from: {checkpoint_path}")
#         agent.load(checkpoint_path)
#         trained_agents[agent_id] = agent
    
#     # Run testing for multiple days
#     days_to_test = [1, 2, 3, 4]  # Test specific days
#     # Or test all days:
#     # days_to_test = list(range(env.total_days))
    
#     results = test_grid_performance(
#         env=env, 
#         trained_agents=trained_agents, 
#         test_data_path='Testing.csv',
#         days_to_analyze=days_to_test
#     )
    
#     # Additional analysis across all tested days
#     print("\nOverall Statistics:")
#     for day, day_results in results.items():
#         controlled_v = np.array(day_results['controlled']['voltages'])
#         controlled_losses = np.array(day_results['controlled']['losses'])
        
#         base_v = np.array(day_results['base']['voltages'])
#         base_losses = np.array(day_results['base']['losses'])
        
#         print(f"\nDay {day} Summary:")
#         print(f"  Loss Reduction: {((np.sum(base_losses) - np.sum(controlled_losses))/np.sum(base_losses))*100:.2f}%")
#         print(f"  Voltage Violation Reduction: {np.sum((base_v < 0.95) | (base_v > 1.05)) - np.sum((controlled_v < 0.95) | (controlled_v > 1.05))}")
#%%
def record_testing_results(env, trained_agents, test_data_path='Testing.csv'):
   """Comprehensive testing and recording of both controlled and baseline cases."""
   print("Starting comprehensive testing and recording...")
   
   tester = GridTester(env, trained_agents, test_data_path)
   
   # Initialize per-agent records
   agent_records = {agent_id: [] for agent_id in trained_agents}
   baseline_records = []
   
   total_days = len(tester.test_data) // 48
   
   for day in range(total_days):
       print(f"\nProcessing day {day}/{total_days-1}")
       results = tester.run_day_comparison(day)
       
       # Record per-agent data
       for step in range(48):
           # Record baseline data
           base_record = {
               'day': day,
               'time_step': step,
               'time_of_day': step/2,
               'demand': results['day_data'].iloc[step]['Demand'],
               'solar': results['day_data'].iloc[step]['Solar'],
           }
           
           base_voltages = results['base']['voltages'][step]
           for bus in range(len(base_voltages)):
               base_record[f'voltage_bus_{bus}'] = base_voltages[bus]
           
           base_record['total_losses'] = results['base']['losses'][step]
           base_record['voltage_violations'] = np.sum(
               (base_voltages < 0.95) | (base_voltages > 1.05)
           )
           baseline_records.append(base_record)
           
           # Record per-agent controlled data
           for agent_id in trained_agents:
               record = {
                   'day': day,
                   'time_step': step,
                   'time_of_day': step/2,
                   'demand': results['day_data'].iloc[step]['Demand'],
                   'solar': results['day_data'].iloc[step]['Solar'],
               }
               
               # Record territorial voltages
               agent_voltages = results['controlled'][agent_id]['voltages'][step]
               for i, bus in enumerate(env.territories[agent_id]):
                   record[f'voltage_bus_{bus}'] = agent_voltages[i]
               
               # Record territorial losses
               record['territorial_losses'] = results['controlled'][agent_id]['losses'][step]
               
               # Record agent's batteries
               for bus, soc in results['controlled'][agent_id]['soc'][step].items():
                   record[f'battery_{bus}_soc'] = soc
                   
               # Record agent's PVs
               for bus, var in results['controlled'][agent_id]['pv_var'][step].items():
                   record[f'pv_{bus}_q'] = var
               
               # Record voltage violations in territory
               record['voltage_violations'] = np.sum(
                   (agent_voltages < 0.95) | (agent_voltages > 1.05)
               )
               
               agent_records[agent_id].append(record)
   
   # Save results
   for agent_id, records in agent_records.items():
       df = pd.DataFrame(records)
       df.to_csv(f'testing_record_{agent_id}.csv', index=False)
   
   pd.DataFrame(baseline_records).to_csv('baseline_record.csv', index=False)
   
   print("\nTesting completed!")
   print("Results saved to individual agent files and baseline_record.csv")
   
   return {agent_id: pd.DataFrame(records) for agent_id, records in agent_records.items()}, \
          pd.DataFrame(baseline_records)

if __name__ == "__main__":
    from grid_env_14 import GridEnv
    from sac_agent_12 import SACAgent
    import matplotlib
    matplotlib.use('Agg')
    
    # Create environment
    env = GridEnv(data_path='Testing.csv')
    
    # Create agents with proper dimensions
    trained_agents = {}
    for agent_id in env.territories:
        test_obs = env._get_obs(agent_id, np.ones(len(env.net.bus)))
        state_dim = len(test_obs)
        action_dim = env.action_spaces[agent_id].shape[0]
        
        print(f"Creating {agent_id} network with state_dim={state_dim}, action_dim={action_dim}")
        
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256
        )
        
        checkpoint_path = os.path.join('results/final_results', f'{agent_id}.pt')
        print(f"Loading weights from: {checkpoint_path}")
        agent.load(checkpoint_path)
        trained_agents[agent_id] = agent
    
    # Run comprehensive testing and recording
    agent_results, baseline_results = record_testing_results(
        env=env,
        trained_agents=trained_agents,
        test_data_path='Testing.csv'
    )
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for agent_id, df in agent_results.items():
        print(f"\n{agent_id}:")
        print(f"  Mean Territorial Losses: {df['territorial_losses'].mean():.4f}")
        print(f"  Total Voltage Violations: {df['voltage_violations'].sum()}")
        
    print("\nBaseline Case:")
    print(f"  Mean Total Losses: {baseline_results['total_losses'].mean():.4f}")
    print(f"  Total Voltage Violations: {baseline_results['voltage_violations'].sum()}")