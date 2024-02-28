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
    x1_range = np.linspace(-4, 7, 200)
    x2_range = np.linspace(-5, 7, 200)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    
    
    epsilon = stabilizer.epsilon
    psi = stabilizer.psi

    # Calculate barrier function values for each constraint
    h1 = -x2 - (1 + epsilon) * x1 + np.sqrt(6 - psi) * (2 + epsilon)
    h2 = -x2 - (1 - epsilon) * x1 + np.sqrt(6 - psi) * (2 - epsilon)
    h3 = x2 + np.sqrt(6 - psi)
    h4 = x1 + np.sqrt(6 - psi)
    
    # Calculate the safe region (h > 0)
    h = np.minimum(np.minimum(h1, h2), np.minimum(h3, h4))
    
    # Plotting the state space and the safe region
    plt.figure(figsize=(8, 8))
    plt.contourf(x1, x2, h, levels=[0, np.inf], colors='green', alpha=0.2, hatches=['/'])

    

    colors = ['b', 'purple', 'orange', 'r']  # Colors for trajectories
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



    plt.xlabel('X1', fontsize=22)
    plt.ylabel('X2', fontsize=22)

    equilibrium_x = 0
    equilibrium_y = 0
    plt.scatter(equilibrium_x, equilibrium_y, color='green', marker='o', s=200, label='Equilibrium')  # Adjust 's' for size
    #plt.title('Polynomial System: State Space and Trajectories', fontsize=20)
    safe_patch = mpatches.Patch(color='green', alpha=0.2, hatch='/', label='Safe Region')
    plt.xticks(fontsize=20)  # Adjust fontsize as needed for x-axis
    plt.yticks(fontsize=20)  # Adjust fontsize as needed for y-axis
    #plt.legend(handles=[safe_patch] + plt.gca().get_lines(), fontsize=16, loc='lower right')  # Combine safe region patch with trajectory lines in the legend

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('polynomial_system_switch.png', dpi=300)
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

    plt.plot(time, barrier_functions, 'g--', label='Barrier Function', linewidth=4)
    plt.plot(time, Lyapunov_functions, 'b-', label='Lyapunov Function', linewidth=4)
    plt.plot(time, relax_values, 'r-.', label='Relaxation', linewidth=4)
    
    # Plot the vertical line if switch_time is valid
    if switch_time is not None:
        plt.axvline(x=switch_time, color='black', linestyle=':', linewidth=5)
        plt.text(switch_time, np.min(barrier_functions), 'Switch', fontsize=22, verticalalignment='bottom')

    plt.xlabel('Time (Seconds)', fontsize=22)
    plt.ylabel('Function Value', fontsize=22)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.savefig('polynomial_function_values_over_time.png', dpi=300)
    plt.show()
    
def plot_polynomial_system_feasibility_map(stabilizer, desired_state):
    x1_range = np.linspace(-8, 8, 100)
    x2_range = np.linspace(-8, 8, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Define your barrier function values here as per your stabilizer's definitions
    epsilon = stabilizer.epsilon
    psi = stabilizer.psi
    
    # Calculate barrier function values for each constraint
    H1 = -X2 - (1 + epsilon) * X1 + np.sqrt(6 - psi) * (2 + epsilon)
    H2 = -X2 - (1 - epsilon) * X1 + np.sqrt(6 - psi) * (2 - epsilon)
    H3 = X2 + np.sqrt(6 - psi)
    H4 = X1 + np.sqrt(6 - psi)
    
    # Calculate the minimum h for the safe region
    H = np.minimum(np.minimum(H1, H2), np.minimum(H3, H4))
    
    # temp_x1 = (np.sqrt(6 - psi) * (2 + epsilon) - np.sqrt(6 - psi) * (2 - epsilon)) / (2*epsilon)
    
    # temp_x2 = -(1+epsilon) * temp_x1 + np.sqrt(6 - psi) * (2 + epsilon)
    
    
    # state_h1_h2_zero = np.array([temp_x1, temp_x2])
    # state_h3_h4_zero = np.array([-np.sqrt(6-psi), -np.sqrt(6-psi)])
    
    # print('state_h1_h2:', state_h1_h2_zero)
    # print('state_h3_h4:', state_h3_h4_zero)
    
    # print(temp_x1 - 1/6 * temp_x1**3)
    # '''
    # sanity check for some states
    # '''
    # # Now perform CLF-CBF QP for these states
    # print("Checking state where h1 = h2 = 0")
    # control, delta, h, V = stabilizer.CLF_CBF_QP(state_h1_h2_zero, desired_state)
    # print("Control:", control, "Delta:", delta, "h:", h, "V:", V)

    # print("\nChecking state where h3 = h4 = 0")
    # control, delta, h, V = stabilizer.CLF_CBF_QP(state_h3_h4_zero, desired_state)
    # print("Control:", control, "Delta:", delta, "h:", h, "V:", V)

    
    # Plotting the state space
    plt.figure(figsize=(10, 8))
    plt.contourf(X1, X2, H, levels=[0, np.inf], colors='green', alpha=0.3, hatches=['/'])
    
    # Check feasibility for each state and scatter plot the feasible states
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            current_state = np.array([X1[i, j], X2[i, j]])
            try:
                control, delta, h, V = stabilizer.CLF_CBF_QP(current_state, desired_state)
                # Scatter plot the state if a control solution is found
                if control is not None:
                    plt.scatter(X1[i, j], X2[i, j], color='red', s=1)
            except Exception as e:
                pass  # Ignore errors and continue to the next state

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Polynomial System Feasibility plot')
    plt.grid(True)
    plt.savefig('polynomial_system_feasible_map.png', dpi=300)
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

    def CLF_CBF_QP(self, current_state, desired_state, rateV = 0.5):
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
        rateh = 0.8
        
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
        
        epsilon_t = 0.001
        
        baseline_constraints.append(dot_h + rateh * h >= epsilon_t) 
        
        
        dot_h_1 = dh1_dstate @ np.array([x1_dot, x2_dot])
        dot_h_2 = dh2_dstate @ np.array([x1_dot, x2_dot])
        dot_h_3 = dh3_dstate @ np.array([x1_dot, x2_dot])
        dot_h_4 = dh4_dstate @ np.array([x1_dot, x2_dot])
        
        # baseline_constraints.append(dot_h_1 + rateh * h1 >= 0) 
        # baseline_constraints.append(dot_h_2 + rateh * h2 >= 0) 
        
        # if(h1==h2):
        #     baseline_constraints.append(dot_h_1 + rateh * h1 >= epsilon_t) 
        #     baseline_constraints.append(dot_h_2 + rateh * h2 >= epsilon_t) 
        # else:
        #     baseline_constraints.append(dot_h + rateh * h >= epsilon_t) 
        
        #baseline_constraints.append(dot_h + rateh * h >= 0.0) 
        
        # if(h3==h4 and h3==0):
        #     print('h3==h4==0')
        #     baseline_constraints.append(dot_h_3 + 1 * h3 >= 0.0) 
        #     baseline_constraints.append(dot_h_4 + 2 * h4 >= 0.0) 
        # else:
            
        #    baseline_constraints.append(dot_h + rateh * h >= 0.0) 
        
        #baseline_constraints.append(dot_h_1 + rateh * h1 >= 0.0) 
        #baseline_constraints.append(dot_h_2 + rateh * h2 >= 0.0) 
        #baseline_constraints.append(dot_h_3 + rateh * h3 >= 0.0) 
        #baseline_constraints.append(dot_h_4 + rateh * h4 >= 0.0) 
        

        p2 = 1e2
        objective = cp.Minimize(cp.square(control) + p2 * cp.square(delta))



        constraints = baseline_constraints

        # Setup and solve the QP
        problem = cp.Problem(objective, constraints)
        problem.solve(solver='SCS', verbose=False)
        
        #print('control:', control.value)
        

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
        x1_dot = control
        x2_dot = -x1 + (1/6) * x1**3 - control

        # Lie derivative of V
        dot_V = dVdstate @ np.array([x1_dot, x2_dot])
        
        
        constraints = []


        # Barrier function constraints
        rateh = 0.8
        
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
        
        epsilon_t = 0.01
        
        
        
        dot_h_1 = dh1_dstate @ np.array([x1_dot, x2_dot])
        dot_h_2 = dh2_dstate @ np.array([x1_dot, x2_dot])
        dot_h_3 = dh3_dstate @ np.array([x1_dot, x2_dot])
        dot_h_4 = dh4_dstate @ np.array([x1_dot, x2_dot])
        
        #baseline_constraints.append(dot_h + rateh * h >= 0.0) 
        
        # if(h3==h4 and h3==0):
        #     print('h3==h4==0')
        #     baseline_constraints.append(dot_h_3 + 1 * h3 >= 0.0) 
        #     baseline_constraints.append(dot_h_4 + 2 * h4 >= 0.0) 
        # else:
            
        #    baseline_constraints.append(dot_h + rateh * h >= 0.0) 
        
        #baseline_constraints.append(dot_h_1 + rateh * h1 >= 0.0) 
        #baseline_constraints.append(dot_h_2 + rateh * h2 >= 0.0) 
        #baseline_constraints.append(dot_h_3 + rateh * h3 >= 0.0) 
        #baseline_constraints.append(dot_h_4 + rateh * h4 >= 0.0) 
        
        epsilon_t = 0.001
        
        
        
        if h < 0.0:
            delta = cp.Variable()
            constraints.append(delta >= 0)
            constraints.append(dot_V + rateV * V <= delta)
            constraints.append(dot_h + rateh * h >= epsilon_t)
            objective = cp.Minimize(cp.square(control) + 1e2 * cp.square(delta))
            # Solve QP
            problem = cp.Problem(objective, constraints)
            problem.solve(solver='SCS', verbose=False)
            relax_value = delta.value      
        else:
            rateV = 0.12   #make rate V small to ensure the compatibility 
            constraints.append(dot_V + rateV * V <= 0)
            constraints.append(dot_h + rateh * h >= 0)
            objective = cp.Minimize(cp.square(control))
            relax_value = 0
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
            if np.abs(state[0] - desired_state[0]) < 0.05 and np.abs(state[1] - desired_state[1]) < 0.05:
                print("System stabilized!")
                break

        return state, state_traj, barrier_functions, Lyapunov_functions, relax_values, control_inputs

# Example usage
def main():
    initial_states = [
        [1.0, 5.2],    # Quadrant 1
        [4, -4],   # Quadrant 2
        [-3, -1],  # Quadrant 3
        [-2, 4.5]    # Quadrant 4
    ]
    
    dt = 0.02           # simulate time discretization
    steps = 2000          # total time step for simulation
    
    epsilon = 0.1
    psi = 0.1
    
    desired_state = np.array([0, 0])
    
    stabilizer = PolynomialSystemStabilizer(epsilon, psi, dt)
    
    '''
    feasibility map plot
    '''
    #plot_polynomial_system_feasibility_map(stabilizer, desired_state)
    

    barrier_functions, Lyapunov_functions, relax_values, controls = plot_state_space_and_trajectories(initial_states,stabilizer, steps)
    
    plot_values_and_control(barrier_functions, Lyapunov_functions, relax_values, controls)


if __name__ == "__main__":
    main()
