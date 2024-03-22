#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

def plot_state_space_and_trajectories(initial_states, stabilizer, steps):
    x1_range = np.linspace(-6, 6, 200)
    x2_range = np.linspace(-4, 4, 200)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    
    
    # Calculate the safe region (h > 0)
    h = -x1 - x2 + 2
    
    # Plotting the state space and the safe region
    plt.figure(figsize=(12, 8))
    plt.contourf(x1, x2, h, levels=[0, np.inf], colors='green', alpha=0.2, hatches=['/'])

    

    colors = ['b', 'purple', 'darkorange', 'r']  # Colors for trajectories
    styles = ['--', '-.', ':', '-']
    trajectory_labels = ['traj1', 'traj2', 'traj3', 'traj4']  # Labels for each trajectory


    for i, (initial_state, color, style, label) in enumerate(zip(initial_states, colors, styles, trajectory_labels)):
        _, state_traj, tmp_barrier_functions, tmp_Lyapunov_functions, tmp_relax_values, tmp_control_inputs = stabilizer.simulate(initial_state, np.array([0, 0]), steps)
        plt.scatter(initial_state[0], initial_state[1], color='black', marker='x', s=100, label='Start' if i == 0 else "")
        plt.plot(np.array(state_traj)[:, 0], np.array(state_traj)[:, 1], color=color, linestyle=style, linewidth=4, label=label)
            
        if color == 'b':
            barrier_functions = tmp_barrier_functions
            Lyapunov_functions = tmp_Lyapunov_functions
            relax_values = tmp_relax_values
            controls = tmp_control_inputs



    plt.xlabel('X1', fontsize=24)
    plt.ylabel('X2', fontsize=24)

    equilibrium_x = 0
    equilibrium_y = 0
    plt.scatter(equilibrium_x, equilibrium_y, color='green', marker='o', s=200, label='Equilibrium')  # Adjust 's' for size
    #plt.title('Polynomial System: State Space and Trajectories', fontsize=20)
    #safe_patch = mpatches.Patch(color='green', alpha=0.2, hatch='/', label='Safe Region')
    plt.xticks(fontsize=22)  # Adjust fontsize as needed for x-axis
    plt.yticks(fontsize=22)  # Adjust fontsize as needed for y-axis
    #plt.legend(handles=[safe_patch] + plt.gca().get_lines(), fontsize=16, loc='lower right')  # Combine safe region patch with trajectory lines in the legend

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('warm_up_switch.png', dpi=300)
    plt.show()
    
    return barrier_functions, Lyapunov_functions, relax_values, controls

def plot_values_and_control(barrier_functions, Lyapunov_functions, relax_values, control_inputs):
    time_steps = np.arange(len(barrier_functions))
    
    dt = 0.02  # Time discretization
    time = time_steps * dt  # Convert time steps to actual time

    # Find the first index where the barrier function becomes positive
    switch_index = np.argmax(np.array(barrier_functions) > 0)  # Using argmax to find the first True index
    switch_time = time[switch_index] if switch_index < len(time) else None  # Ensure index is within bounds

    # Create a new figure for the Lyapunov and barrier functions
    plt.figure(figsize=(12, 8))

    plt.plot(time, barrier_functions, 'g-.', label='Barrier Function', linewidth=4)
    plt.plot(time, Lyapunov_functions, 'b-', label='Lyapunov Function', linewidth=4)
    #plt.plot(time, relax_values, 'r-.', label='Relaxation', linewidth=4)
    plt.axhline(y=0, color='black', label = 'Zero Line', linestyle='--', linewidth=4)
    
    # Plot the vertical line if switch_time is valid
    if switch_time is not None:
        plt.axvline(x=switch_time, color='black', linestyle=':', linewidth=5)
        plt.text(switch_time, np.max(Lyapunov_functions), 'Switch', fontsize=24, verticalalignment='top')

    plt.xlabel('Time (Seconds)', fontsize=22)
    plt.ylabel('Function Value', fontsize=22)
    plt.legend(fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.tight_layout()
    plt.savefig('warmup_values_over_time.png', dpi=300)
    plt.show()
    



class WarmUpStabilizer:

    def __init__(self, dt, rateV=0.5, rateh=1.0):
        
        self.dt = dt  # time step

        self.rateV = rateV  #class K for CLF
        self.rateh = rateh  #class K for CBF
        
    def Lambda(self, s):
        if s < 2.01:
            return np.exp(-1 / (-s + 2.01))
        else:
            return 0

    def dynamics(self, state, control):
        dt = self.dt
        def f(current_state, u):
            x1, x2 = current_state
            x1_dot = x2 - x1 * self.Lambda(x1)
            x2_dot = -x1 + u
            return np.array([x1_dot, x2_dot])

        # Compute RK4 intermediate steps
        k1 = f(state, control)
        k2 = f(state + 0.5 * dt * k1, control)
        k3 = f(state + 0.5 * dt * k2, control)
        k4 = f(state + dt * k3, control)

        # Update the state using RK4 formula
        new_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return new_state

    def CLF_CBF_QP(self, current_state, desired_state):
        x1, x2 = current_state
        x1_d, x2_d = desired_state

        # Quadratic Lyapunov function
        V = 0.5 * (x1 - x1_d) ** 2 + 0.5 * (x2 - x2_d) ** 2

        # Derivative of V
        dVdstate = np.array([x1 - x1_d, x2 - x2_d])

        # Control inputs
        control = cp.Variable()
        delta = cp.Variable()

        # Dynamics of the polynomial system
        x1_dot = x2 - x1 * self.Lambda(x1)
        x2_dot = -x1 + control

        # Lie derivative of V
        dot_V = dVdstate @ np.array([x1_dot, x2_dot])

        # Constraint for the decrease rate of V
        baseline_constraints = []
        baseline_constraints.append(delta >= 0)
        baseline_constraints.append(dot_V + self.rateV * V <= delta)

        # Barrier function constraints

        h = -x1 - x2 + 2

        # Define dh_dstate based on the chosen barrier function h
        dh_dstate = np.array([-1, -1])

        # Derivative of h
        dot_h = dh_dstate @ np.array([x1_dot, x2_dot])

        epsilon_t = 0.001
        
        # uncomment the following line for CCLF-QP control
        baseline_constraints.append(dot_h + self.rateh * h >= epsilon_t)

        p2 = 1e3
        objective = cp.Minimize(cp.square(control) + p2 * cp.square(delta))

        constraints = baseline_constraints

        # Setup and solve the QP
        problem = cp.Problem(objective, constraints)
        problem.solve(solver='SCS', verbose=False)

        return control.value, delta.value, h, V

    def CLF_CBF_Switching_QP(self, current_state, desired_state, rateV = 0.5):
        x1, x2 = current_state
        x1_d, x2_d = desired_state

        # Quadratic Lyapunov function
        V = 0.5 * (x1 - x1_d)**2 + 0.5 * (x2 - x2_d)**2

        # Derivative of V
        dVdstate = np.array([x1 - x1_d, x2 - x2_d])

        # Control inputs
        control = cp.Variable()
        
        delta = cp.Variable()

        # Dynamics of the polynomial system
        x1_dot = x2 - x1 * self.Lambda(x1)
        x2_dot = -x1 + control

        # Lie derivative of V
        dot_V = dVdstate @ np.array([x1_dot, x2_dot])
        
        
        constraints = []
        
        
        
        # Barrier function constraints
        h = -x1 - x2 + 2

        # Define dh_dstate based on the chosen barrier function h
        dh_dstate = np.array([-1, -1])

        # Derivative of h
        dot_h = dh_dstate @ np.array([x1_dot, x2_dot])

        epsilon_t = 0.001

    
        
        
        
        if h < 0.0:
            # if in unsafe set, just solve the BNCBF-QP
            
            #delta = cp.Variable()
            #constraints.append(delta >= 0)
            #constraints.append(dot_V + rateV * V <= delta)
            #constraints.append(dot_h_3 + rateh * h3 >= 0)
            #constraints.append(dot_h_4 + rateh * h4 >= epsilon_t)
            
            constraints.append(dot_h + self.rateh * h >= epsilon_t)
            
            # print('h3:', h3)
            # print('h4:', h4)
            objective = cp.Minimize(cp.square(control))
            # Solve QP
            problem = cp.Problem(objective, constraints)
            problem.solve(solver='SCS', verbose=False)
            relax_value = 0   
        else:

            constraints.append(dot_V + self.rateV * V <= 0)
            constraints.append(dot_h + self.rateh * h >= 0)
            objective = cp.Minimize(cp.square(control))
            relax_value = 0
            problem = cp.Problem(objective, constraints)
            problem.solve(solver='SCS', verbose=False)

        
        #print('control:', control.value)
        

        return control.value, relax_value, h, V

    def simulate(self, initial_state, desired_state, steps):
        state = initial_state
        state_traj = []
        
        barrier_functions = []
        
        Lyapunov_functions = []
        
        relax_values = []
        
        control_inputs = []

        for step in range(steps):
            state_traj.append(state)
            '''
            Lipschitz controller, relaxed CLF constraint over time
            '''
            #control_input, delta , h, V = self.CLF_CBF_QP(state, desired_state)
            
            '''
            Switching strategy, switch to CLF-CBF QP (no relaxation) once CLF and CBF are compatiable
            '''
            
            control_input, delta , h, V = self.CLF_CBF_Switching_QP(state, desired_state)
            state = self.dynamics(state, control_input)
            #print('state:', state)
            barrier_functions.append(h)
            Lyapunov_functions.append(V)
            relax_values.append(delta)
            control_inputs.append(control_input)

            # Stopping condition (optional)
            if np.abs(state[0] - desired_state[0]) < 0.02 and np.abs(state[1] - desired_state[1]) < 0.02:
                print("System stabilized!")
                break
            
            # if state[0] > 2.0 and np.abs(state[1]-0.0) < 0.2:
            #     print("System failed!")  
            #     state_traj.append(np.array([state[0], 0]))
            #     break
            
            

        return state, state_traj, barrier_functions, Lyapunov_functions, relax_values, control_inputs

# Example usage
def main():
    initial_states = [
        [4, 3.2],    # Quadrant 1
        [3, -2],   # Quadrant 2
        [-4.2, -3.5],  # Quadrant 3
        [-2, 3]    # Quadrant 4
    ]
    
    dt = 0.01         # simulate time discretization
    steps = 1000          # total time step for simulation
    
    rateV = 0.5        #class K for CLF
    rateh = 1.0        #class K for CBF
    
    
    stabilizer = WarmUpStabilizer(dt, rateV, rateh)
    
    

    barrier_functions, Lyapunov_functions, relax_values, controls = plot_state_space_and_trajectories(initial_states,stabilizer, steps)
    
    plot_values_and_control(barrier_functions, Lyapunov_functions, relax_values, controls)


if __name__ == "__main__":
    main()
