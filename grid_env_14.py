import gym
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from typing import Dict, Tuple
from gym import spaces

class GridEnv(gym.Env):
    def __init__(self, data_path: str = 'Training.csv'):
        """
        Initialize the grid environment with comprehensive monitoring.
        
        Args:
            data_path (str): Path to CSV file containing demand and solar profiles
        """
        # Load data
        # Load data
        self.data = pd.read_csv(data_path)
        print(f"Loaded data with {len(self.data)} timesteps")  # Debug
        
        self.time_step = 0
        self.current_day = 0
        self.total_days = len(self.data) // 48
        print(f"Total days in data: {self.total_days}")  # Debug
        
        if len(self.data) % 48 != 0:
            print("WARNING: Data length is not a multiple of 48!")
        self.total_steps = len(self.data)  # Total timesteps available
        self.day_initial_soc = {}  # Store initial SOC for each day
        self.voltage_violation_occurred = False  # Flag for voltage violation
        
    
        # System parameters
        self.v_limits = (0.9, 1.1)
        self.pv_cap = 1.0  # PV capacity in MW
        self.bat_cap = 2.0  # Battery capacity in MWh
        self.bat_p_max = 0.5  # Max battery power in MW
        self.delta_t = 0.5  # 30-minute intervals in hours
        
        # Battery operational limits
        self.max_cap_SoC = 0.9 * self.bat_cap  # Maximum SOC limit (90%)
        self.min_cap_SoC = 0.2 * self.bat_cap  # Minimum SOC limit (20%)
    
        # Define territories and assets
        self.territories = {
            'agent_1': [0, 1, 2, 3, 4, 18, 19, 20, 21, 22, 23, 24],
            'agent_2': list(range(5, 18)),
            'agent_3': list(range(25, 33))
        }
    
        # Asset mapping
        self.pv_buses = {16: 'agent_2', 20: 'agent_1', 31: 'agent_3'}
        self.bat_buses = {23: 'agent_1', 7: 'agent_2', 14: 'agent_2'}
    
        # Create agent asset mappings
        self.agent_assets = {}
        for agent_id in self.territories.keys():
            batteries = [bus for bus, agent in self.bat_buses.items() if agent == agent_id]
            pvs = [bus for bus, agent in self.pv_buses.items() if agent == agent_id]
            self.agent_assets[agent_id] = {
                'batteries': batteries,
                'pvs': pvs,
                'action_map': {}
            }
            action_idx = 0
            for bus in batteries:
                self.agent_assets[agent_id]['action_map'][f'bat_{bus}'] = action_idx
                action_idx += 1
            for bus in pvs:
                self.agent_assets[agent_id]['action_map'][f'pv_{bus}'] = action_idx
                action_idx += 1
    
        # Initialize battery states
        self.bat_soc = {bus: self.bat_cap / 2 for bus in self.bat_buses}
    
        # Initialize metrics storage
        # Traditional metrics
        self.voltage_history = []     # Bus voltages
        self.soc_history = []         # Battery SOC
        self.loss_history = []        # Network losses
        self.action_history = []      # Raw actions
        
        # New detailed monitoring
        self.raw_actions_history = []       # Store raw agent actions
        self.pv_power_history = []          # PV real power output
        self.pv_var_history = []            # PV reactive power output
        self.battery_power_history = []      # Battery charge/discharge power
        self.power_setpoints_history = []    # All power setpoints
        self.constraint_violations = {       # Track constraint violations
            'voltage': [],                   # Voltage limit violations
            'soc': [],                       # Battery SOC limit violations
            'power': [],                     # Power limit violations
            'convergence': []                # Power flow convergence failures
        }
        
        # Network operation records
        self.operational_metrics = {
            'voltage_deviations': [],        # Mean voltage deviations
            'network_losses': [],            # Total network losses
            'pv_utilization': [],           # PV capacity utilization
            'battery_cycles': {bus: [] for bus in self.bat_buses},  # Battery cycling
            'reactive_power_support': []     # VAR support metrics
        }
    
        # Initialize network
        self.net = pn.case33bw()
        self.base_p = self.net.load.p_mw.copy()
        self.base_q = self.net.load.q_mvar.copy()
        
        # Create static sgen elements for PVs with proper limits
        for pv_bus in self.pv_buses:
            pp.create_sgen(
                self.net,
                bus=pv_bus,
                p_mw=0.0,
                q_mvar=0.0,
                name=f"PV_{pv_bus}",
                controllable=True,
                max_p_mw=self.pv_cap,
                min_p_mw=0,
                max_q_mvar=self.pv_cap,
                min_q_mvar=-self.pv_cap
            )
        
        # Create static sgen elements for batteries with proper limits
        for bat_bus in self.bat_buses:
            pp.create_sgen(
                self.net,
                bus=bat_bus,
                p_mw=0.0,
                q_mvar=0.0,
                name=f"Bat_{bat_bus}",
                controllable=True,
                max_p_mw=self.bat_p_max,     # Discharge limit
                min_p_mw=-self.bat_p_max     # Charge limit
            )
        
        # Store indices for quick access
        self.pv_sgen_indices = {
            bus: idx for idx in self.net.sgen.index 
            if self.net.sgen.at[idx, 'name'].startswith('PV_')
            for bus in [int(self.net.sgen.at[idx, 'name'].split('_')[1])]
        }
        
        self.bat_sgen_indices = {
            bus: idx for idx in self.net.sgen.index 
            if self.net.sgen.at[idx, 'name'].startswith('Bat_')
            for bus in [int(self.net.sgen.at[idx, 'name'].split('_')[1])]
        }
        
        # Data export setup
        self.record_columns = self._setup_record_columns()
        
        # Setup spaces
        self.setup_spaces()
        self.initialized = False  # Set to False initially
        
        # NEW
        self.use_solar_penalty = False  #NEW
        
        # # NEW: Add normalized time step to observation space
        # for agent_id in self.territories:
        #     # NEW
        #     self.observation_spaces[agent_id] = self._expand_obs_space(agent_id)
            
    def _update_timestep(self):
        """
        Update time_step and current_day while ensuring they stay within bounds.
        Returns True if a new day starts, False otherwise.
        """
        new_time_step = self.time_step + 1
        
        # First check if we've reached end of data
        if new_time_step >= len(self.data):
            self.time_step = 0
            self.current_day = 0
            return True
        
        # Update time_step
        self.time_step = new_time_step
        
        # Check if we're at day boundary (every 48 steps)
        if self.time_step % 48 == 0:
            self.current_day = (self.time_step // 48) % self.total_days
            return True
        
        return False
    
    def _setup_record_columns(self):
        """Setup columns for comprehensive data recording."""
        columns = ['timestamp', 'episode', 'step', 'load', 'solar']
        
        # Add voltage columns for all buses
        columns.extend([f'voltage_bus_{i}' for i in range(len(self.net.bus))])
        
        # Add PV columns
        for pv_bus in self.pv_buses:
            columns.extend([
                f'pv_{pv_bus}_p_mw',
                f'pv_{pv_bus}_q_mvar',
                f'pv_{pv_bus}_action'
            ])
        
        # Add battery columns
        for bat_bus in self.bat_buses:
            columns.extend([
                f'bat_{bat_bus}_power_mw',
                f'bat_{bat_bus}_soc_mwh',
                f'bat_{bat_bus}_action'
            ])
        
        # Add network metrics
        columns.extend([
            'total_losses_mw',
            'min_voltage_pu',
            'max_voltage_pu',
            'mean_voltage_deviation',
            'convergence_status'
        ])
        
        return columns

    def setup_spaces(self):
        self.observation_spaces = {}
        self.action_spaces = {}

        for agent_id in self.territories:
            # NEW: Use the expanded observation space that includes time
            self.observation_spaces[agent_id] = self._expand_obs_space(agent_id)
            
            n_batteries = len(self.agent_assets[agent_id]['batteries'])
            n_pvs = len(self.agent_assets[agent_id]['pvs'])

            # State space: [voltages, load, solar, battery_socs]
            obs_dim = len(self.territories[agent_id]) + 2 + n_batteries

            # Define observation space bounds
            voltage_low = np.full(len(self.territories[agent_id]), -0.2)
            voltage_high = np.full(len(self.territories[agent_id]), 0.2)
            demand_low = np.array([0.0])
            demand_high = np.array([1.0])
            solar_low = np.array([0.0])
            solar_high = np.array([1.0])
            soc_low = np.zeros(n_batteries)
            soc_high = np.ones(n_batteries)

            obs_low = np.concatenate([voltage_low, demand_low, solar_low, soc_low])
            obs_high = np.concatenate([voltage_high, demand_high, solar_high, soc_high])

            self.observation_spaces[agent_id] = spaces.Box(
                low=obs_low, high=obs_high, dtype=np.float32)

            # Action space: [-1,1] for each battery and PV
            self.action_spaces[agent_id] = spaces.Box(
                low=-1, high=1, shape=(n_batteries + n_pvs,), dtype=np.float32)

            
    # NEW method
    def _expand_obs_space(self, agent_id):
        # Original dimensions
        n_batteries = len(self.agent_assets[agent_id]['batteries'])
        obs_dim = len(self.territories[agent_id]) + 2 + n_batteries + 1  # +1 for time
        
        voltage_low = np.full(len(self.territories[agent_id]), -0.2)
        voltage_high = np.full(len(self.territories[agent_id]), 0.2)
        demand_low = np.array([0.0])
        demand_high = np.array([1.0])
        solar_low = np.array([0.0])
        solar_high = np.array([1.0])
        soc_low = np.zeros(n_batteries)
        soc_high = np.ones(n_batteries)
        time_low = np.array([0.0])  # Normalized time
        time_high = np.array([1.0])
        
        obs_low = np.concatenate([voltage_low, demand_low, solar_low, soc_low, time_low])
        obs_high = np.concatenate([voltage_high, demand_high, solar_high, soc_high, time_high])
        
        return spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)


    def step(self, actions: Dict) -> Tuple:
        print(f"Step - Current timestep: {self.time_step}, Day: {self.current_day}")  # Debug
        
        # Validate time_step before any operations
        if self.time_step >= len(self.data):
            print(f"Invalid timestep detected: {self.time_step}, Data length: {len(self.data)}")  # Debug
            obs = self.reset()
            rewards = {a: 0 for a in actions}
            done = {a: True for a in actions}
            info = {a: {'timestep_reset': True} for a in actions}
            return obs, rewards, done, info
            
        # Get current conditions
        load = self.data.iloc[self.time_step]['Demand']
        solar = self.data.iloc[self.time_step]['Solar']
        
        # Update base loads
        self.net.load.p_mw = self.base_p * load
        self.net.sgen.loc[:, ['p_mw', 'q_mvar']] = 0.0
        
        # Process each agent's actions
        for agent_id, agent_actions in actions.items():
            # Process battery actions
            for bus in self.agent_assets[agent_id]['batteries']:
                idx = self.agent_assets[agent_id]['action_map'][f'bat_{bus}']
                action = agent_actions[idx]
                
                # Scale action to power (-ve for charging, +ve for discharging)
                power = action * self.bat_p_max
                current_soc = self.bat_soc[bus]
                
                # Predict next SOC (note sign convention: +ve power = discharge = decrease SOC)
                next_soc = current_soc - (power * self.delta_t)
                
                # Adjust power to respect SOC limits
                if next_soc > self.max_cap_SoC:
                    power = -(self.max_cap_SoC - current_soc) / self.delta_t
                    next_soc = self.max_cap_SoC
                elif next_soc < self.min_cap_SoC:
                    power = -(self.min_cap_SoC - current_soc) / self.delta_t
                    next_soc = self.min_cap_SoC
                
                # Update battery state
                self.bat_soc[bus] = next_soc
                
                # Apply to network (use stored index)
                sgen_idx = self.bat_sgen_indices[bus]
                self.net.sgen.at[sgen_idx, 'p_mw'] = power
            
            # Process PV actions
            for bus in self.agent_assets[agent_id]['pvs']:
                idx = self.agent_assets[agent_id]['action_map'][f'pv_{bus}']
                action = agent_actions[idx]
                
                # Set PV real power
                p_pv = self.pv_cap * solar
                sgen_idx = self.pv_sgen_indices[bus]
                self.net.sgen.at[sgen_idx, 'p_mw'] = p_pv
                
                # Calculate and set reactive power
                q_max = np.sqrt(max(0, self.pv_cap**2 - p_pv**2))
                q_var = action * q_max
                self.net.sgen.at[sgen_idx, 'q_mvar'] = q_var


        # Run power flow
        try:
            pp.runpp(self.net)
            voltages = self.net.res_bus.vm_pu.values

            self.voltage_history.append(voltages)
            self.soc_history.append(self.bat_soc.copy())
            self.loss_history.append(self.net.res_line.pl_mw.sum())

            # Check for voltage violations
            voltage_violations = np.any(np.isnan(voltages)) or \
                                 np.any(voltages < self.v_limits[0]) or \
                                 np.any(voltages > self.v_limits[1])


        
            if voltage_violations:
                self.voltage_violation_occurred = True
                # Don't increment time_step, let reset handle it
                obs = {a: self._get_obs(a, voltages) for a in actions}
                rewards = {a: -100 for a in actions}
                done = {a: True for a in actions}
                info = {a: {'voltage_violation': True} for a in actions}
                return obs, rewards, done, info
                
            # Normal step progression
            # Get rewards before updating time_step
            rewards = {a: self._get_reward(a, voltages) for a in actions}
            
            # Update time_step after all computations
            old_time_step = self.time_step  # Save current time_step
            new_day = self._update_timestep()
            
            # Get observations using old time_step
            obs = {a: self._get_obs(a, voltages) for a in actions}
            done = {a: new_day for a in actions}
            info = {a: {
                'voltages': voltages[self.territories[a]],
                'soc': {b: self.bat_soc[b] for b in self.bat_buses if self.bat_buses[b] == a}
            } for a in actions}
            
            return obs, rewards, done, info
                
        except pp.powerflow.LoadflowNotConverged:
            obs = {a: self._get_obs(a, np.ones(len(self.net.bus))) for a in actions}
            rewards = {a: -100 for a in actions}
            done = {a: True for a in actions}
            info = {a: {'convergence_error': True} for a in actions}
            return obs, rewards, done, info

    def _get_reward(self, agent_id: str, voltages: np.ndarray) -> float:
        # Get voltages at agent's buses, excluding the slack bus (bus 0)
        agent_buses = [bus for bus in self.territories[agent_id] if bus != 0]
        v = voltages[agent_buses]
        
        # Calculate over-voltage and under-voltage penalties per bus
        over_voltage = np.maximum(v - 1.03, 0)
        under_voltage = np.maximum(0.97 - v, 0)
        
        # Sum the penalties for all buses in the agent's territory
        v_penalty = -100 * np.sum(over_voltage + under_voltage)
        
        # Keep the -100 penalty if any voltages are outside system-wide limits
        if np.any(v < self.v_limits[0]) or np.any(v > self.v_limits[1]):
            v_penalty -= 100
        
        # System losses penalty scaled by the agent's territory size
        loss_penalty = -5 * self.net.res_line.pl_mw.sum() * \
                       (len(self.territories[agent_id]) / len(self.net.bus))
                       
        # New solar charging penalty
        solar_penalty = 0
        if self.use_solar_penalty:  # Only apply when flag is True
            if 14 <= self.time_step % 48 <= 34:
                for bus in self.agent_assets[agent_id]['batteries']:
                    sgen_idx = self.bat_sgen_indices[bus]
                    power = self.net.sgen.at[sgen_idx, 'p_mw']
                    if power >= 0:
                        penalty_factor = (self.bat_p_max - power) / self.bat_p_max
                        solar_penalty += -30 * penalty_factor
        
        return v_penalty + loss_penalty + solar_penalty


    # def _get_obs(self, agent_id: str, voltages: np.ndarray) -> np.ndarray:
    #     # Normalize voltages by subtracting 1.0
    #     norm_voltages = voltages[self.territories[agent_id]] - 1.0  # centered around zero

    #     # Get Demand and Solar
    #     demand = self.data.iloc[self.time_step]['Demand']
    #     solar = self.data.iloc[self.time_step]['Solar']

    #     # Normalize SoC by dividing by bat_cap
    #     soc = np.array([self.bat_soc[bus] / self.bat_cap for bus in self.bat_buses if self.bat_buses[bus] == agent_id])

    #     # Add normalized time step (0 to 1)
    #     norm_time = (self.time_step % 48) / 47.0
        
    #     obs = np.concatenate([
    #         norm_voltages,
    #         [demand],
    #         [solar],
    #         soc,
    #         [norm_time]
    #     ]).astype(np.float32)
        

    #     return obs
    
    def _get_obs(self, agent_id: str, voltages: np.ndarray) -> np.ndarray:
        # Normalize voltages by subtracting 1.0
        norm_voltages = voltages[self.territories[agent_id]] - 1.0  # centered around zero
    
        # Get Demand and Solar
        demand = self.data.iloc[self.time_step]['Demand']
        solar = self.data.iloc[self.time_step]['Solar']
    
        # Normalize SoC by dividing by bat_cap
        soc = np.array([self.bat_soc[bus] / self.bat_cap 
                        for bus in self.bat_buses if self.bat_buses[bus] == agent_id])
    
        # Add normalized time step (0 to 1)
        norm_time = np.array([(self.time_step % 48) / 47.0])
        
        # Ensure all components are proper arrays
        obs = np.concatenate([
            norm_voltages.astype(np.float32),
            np.array([demand], dtype=np.float32),
            np.array([solar], dtype=np.float32),
            soc.astype(np.float32),
            norm_time.astype(np.float32)
        ])
        
        #print(f"Observation shape for {agent_id}: {obs.shape}")  # Debug
        return obs

        
    def reset(self):
        print(f"Reset - Current timestep: {self.time_step}, Day: {self.current_day}")
        
        if self.voltage_violation_occurred:
            # On voltage violation, go back to start of current day
            self.time_step = self.current_day * 48
            self.bat_soc = self.day_initial_soc.copy()
            self.voltage_violation_occurred = False
        else:
            # If we're at a day boundary, current_day is already updated
            # Just ensure timestep aligns with the current day
            new_time_step = self.current_day * 48
            if new_time_step >= len(self.data):
                self.current_day = 0
                self.time_step = 0
            else:
                self.time_step = new_time_step
                
            if self.current_day == 0:
                self.bat_soc = {bus: self.bat_cap / 2 for bus in self.bat_buses}
        
        # Safety check
        if self.time_step >= len(self.data):
            print(f"Reset safety triggered - Invalid timestep: {self.time_step}")
            self.time_step = 0
            self.current_day = 0
            self.bat_soc = {bus: self.bat_cap / 2 for bus in self.bat_buses}
        
        # Store initial SOC for this day
        self.day_initial_soc = self.bat_soc.copy()
        
        # Reset histories
        self.voltage_history = []
        self.soc_history = []
        self.loss_history = []
        self.action_history = []
        
        print(f"After reset - Timestep: {self.time_step}, Day: {self.current_day}")
        
        # Reset network state
        initial_load = self.data.iloc[self.time_step]['Demand']
        self.net.load.p_mw = self.base_p * initial_load
        self.net.load.q_mvar = self.base_q * initial_load
        self.net.sgen.loc[:, ['p_mw', 'q_mvar']] = 0.0
        
        try:
            pp.runpp(self.net)
            initial_voltages = self.net.res_bus.vm_pu.values
        except pp.powerflow.LoadflowNotConverged:
            initial_voltages = np.ones(len(self.net.bus))
        
        return {a: self._get_obs(a, initial_voltages) for a in self.territories}



