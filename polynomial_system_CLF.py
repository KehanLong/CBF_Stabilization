#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

def plot_state_space_and_trajectories(initial_states, epsilon, psi, dt, steps):
    x1_range = np.linspace(-8, 8, 200)
    x2_range = np.linspace(-8, 8, 200)
    x1, x2 = np.meshgrid(x1_range, x2_range)

    # Calculate barrier function values for each constraint
    h1 = -x2 - (1 + epsilon) * x1 + np.sqrt(6 - psi) * (2 + epsilon)
    h2 = -x2 - (1 - epsilon) * x1 + np.sqrt(6 - psi) * (2 - epsilon)
    h3 = x2 + np.sqrt(6 - psi)
    h4 = x1 + np.sqrt(6 - psi)
    
    # Calculate the safe region (h > 0)
    h = np.minimum(np.minimum(h1, h2), np.minimum(h3, h4))
    
    # Plotting the state space and the safe region
    plt.figure(figsize=(8, 6))
    plt.contourf(x1, x2, h, levels=[0, np.inf], colors='green', alpha=0.3, hatches=['/'])


    # Plot trajectories from different quadrants
    stabilizer = PolynomialSystemStabilizer(epsilon, psi, dt)
    

    colors = ['b', 'purple', 'orange', 'r']  # Colors for trajectories
    styles = ['--', '-.', ':', '-']
    trajectory_labels = ['traj1', 'traj2', 'traj3', 'traj4']  # Labels for each trajectory


    for i, (initial_state, color, style, label) in enumerate(zip(initial_states, colors, styles, trajectory_labels)):
        _, state_traj, tmp_barrier_functions, tmp_Lyapunov_functions, tmp_relax_values, tmp_control_inputs = stabilizer.simulate(initial_state, np.array([0, 0]), steps)
        plt.plot(np.array(state_traj)[:, 0], np.array(state_traj)[:, 1], color=color, linestyle=style, linewidth=4, label=label)
            
        if color == 'b':
            barrier_functions = tmp_barrier_functions
            Lyapunov_functions = tmp_Lyapunov_functions
            relax_values = tmp_relax_values
            controls = tmp_control_inputs



    plt.xlabel('X1', fontsize=17)
    plt.ylabel('X2', fontsize=17)
    plt.title('Polynomial System: State Space and Trajectories', fontsize=20)
    safe_patch = mpatches.Patch(color='green', alpha=0.3, hatch='/', label='Safe Region')
    #plt.legend(handles=[safe_patch] + plt.gca().get_lines(), fontsize=16, loc='lower right')  # Combine safe region patch with trajectory lines in the legend

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('polynomial_system.png', dpi=300)
    plt.show()
    
    return barrier_functions, Lyapunov_functions, relax_values, controls

def plot_values_and_control(barrier_functions, Lyapunov_functions, relax_values, control_inputs):
    time_steps = np.arange(len(barrier_functions))

    # Create a new figure for the Lyapunov and barrier functions
    plt.figure(figsize=(8, 6))

    plt.plot(time_steps, barrier_functions, 'g--', label='Barrier Function', linewidth = 3)
    plt.plot(time_steps, Lyapunov_functions, 'b-', label='Lyapunov Function', linewidth = 2)
    plt.plot(time_steps, relax_values, 'r-.', label='Relaxation', linewidth = 3)
    plt.xlabel('Time Steps', fontsize = 17)
    plt.ylabel('Function Value', fontsize = 17)
    plt.title('Function Values Over Time for Trajectory', fontsize = 20)
    plt.legend(fontsize = 16)

    plt.tight_layout()
    plt.savefig('polynomial_system_function_values.png', dpi=300)
    plt.show()

class PolynomialSystemStabilizer:
    def __init__(self, epsilon, psi, dt):
        self.dt = dt  # time step
        self.epsilon = epsilon
        self.psi = psi

    def dynamics(self, state, control):
        dt = self.dt
        
        def f(current_state, u):
            x1, x2 = current_state
            x1_dot = u
            x2_dot = -x1 + (1/6) * x1**3 - u
            return np.array([x1_dot, x2_dot])
        
        # Compute RK4 intermediate steps
        k1 = f(state, control)
        k2 = f(state + 0.5 * dt * k1, control)
        k3 = f(state + 0.5 * dt * k2, control)
        k4 = f(state + dt * k3, control)
        
        # Update the state using RK4 formula
        new_state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return new_state

    def CLF_CBF_QP(self, current_state, desired_state, rateV = 1.0):
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
        x1_dot = control
        x2_dot = -x1 + (1/6) * x1**3 - control

        # Lie derivative of V
        dot_V = dVdstate @ np.array([x1_dot, x2_dot])
        
        # Constraint for the decrease rate of V
        baseline_constraints = []
        baseline_constraints.append(delta >= 0)
        baseline_constraints.append(dot_V + rateV * V <= delta)

        # Barrier function constraints
        rateh = 3.0
        
        h1 = -x2 - (1 + self.epsilon) * x1 + np.sqrt(6 - self.psi) * (2 + self.epsilon)
        h2 = -x2 - (1 - self.epsilon) * x1 + np.sqrt(6 - self.psi) * (2 - self.epsilon)
        h3 = x2 + np.sqrt(6 - self.psi)
        h4 = x1 + np.sqrt(6 - self.psi)
        h = min(h1, h2, h3, h4)
        
        
        # Define dh_dstate based on the chosen barrier function h
        dh1_dstate = np.array([-1 - self.epsilon, -1])
        dh2_dstate = np.array([-1 + self.epsilon, -1])
        dh3_dstate = np.array([0, 1])
        dh4_dstate = np.array([1, 0])

        if h == h1:
            #print('h1 active:', h)
            dh_dstate = dh1_dstate
        elif h == h2:
            #print('h2 active:', h)
            dh_dstate = dh2_dstate
        elif h == h3:
            #print('h3 active:', h)
            dh_dstate = dh3_dstate
        else:  # h == h4
            #print('h4 active:', h)
            dh_dstate = dh4_dstate

        # h = h4
        # print(h)
        # dh_dstate = dh4_dstate
        # Derivative of h
        dot_h = dh_dstate @ np.array([x1_dot, x2_dot])
        
        baseline_constraints.append(dot_h + rateh * h >= 0.0) 
        

        p2 = 1e2
        objective = cp.Minimize(cp.square(control) + p2 * cp.square(delta))



        constraints = baseline_constraints

        # Setup and solve the QP
        problem = cp.Problem(objective, constraints)
        problem.solve(solver='SCS', verbose=False)
        
        #print('control:', control.value)
        

        return control.value, delta.value, h, V

    def simulate(self, initial_state, desired_state, steps):
        state = initial_state
        state_traj = []
        
        barrier_functions = []
        
        Lyapunov_functions = []
        
        relax_values = []
        
        control_inputs = []

        for step in range(steps):
            state_traj.append(state)
            control_input, delta, h , V= self.CLF_CBF_QP(state, desired_state)
            state = self.dynamics(state, control_input)
            #print('state:', state)
            barrier_functions.append(h)
            Lyapunov_functions.append(V)
            relax_values.append(delta)
            control_inputs.append(control_input)

            # Stopping condition (optional)
            if np.abs(state[0] - desired_state[0]) < 0.05 and np.abs(state[1] - desired_state[1]) < 0.05:
                print("System stabilized!")
                break

        return state, state_traj, barrier_functions, Lyapunov_functions, relax_values, control_inputs

# Example usage
def main():
    initial_states = [
        [2, 0],    # Quadrant 1
        [3, -2],   # Quadrant 2
        [-2, -1],  # Quadrant 3
        [-2, 5]    # Quadrant 4
    ]
    
    dt = 0.02            # simulate time discretization
    steps = 1000           # total time step for simulation
    
    epsilon = 0.1
    psi = 0.1

    barrier_functions, Lyapunov_functions, relax_values, controls = plot_state_space_and_trajectories(initial_states, epsilon, psi, dt, steps)
    
    plot_values_and_control(barrier_functions, Lyapunov_functions, relax_values, controls)


if __name__ == "__main__":
    main()
