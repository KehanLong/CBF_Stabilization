#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def plot_state_space_and_trajectories(initial_states, stabilizer, steps):
    theta_range = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    theta_dot_range = np.linspace(-6, 6, 200)
    theta, theta_dot = np.meshgrid(theta_range, theta_dot_range)
    
    epsilon1 = stabilizer.epsilon1
    epsilon2 = stabilizer.epsilon2

    # Calculate barrier function values
    h1 = -1/epsilon1 * theta - theta_dot
    h2 = theta_dot + epsilon2 * theta 
    h = np.minimum(h1, h2)
    h_tilde = np.minimum(-h1, -h2)

    # Plotting the state space
    plt.figure(figsize=(8, 6))
    plt.contourf(theta, theta_dot, h, levels=[0, np.inf], colors='green', alpha=0.2, hatches=['/'])
    
    #the following is for plotting the symmetric safe set
    #plt.contourf(theta, theta_dot, h_tilde, levels=[0, np.inf], colors='green', alpha=0.2, hatches=['/'])

    # Plot trajectories from different quadrants
    

    colors = ['b', 'purple', 'orange', 'r']  # Colors for trajectories
    styles = ['--', '-.', ':', '-']
    trajectory_labels = ['traj1', 'traj2', 'traj3', 'traj4']  # Labels for each trajectory


    for i, (initial_state, color, style, label) in enumerate(zip(initial_states, colors, styles, trajectory_labels)):
        _, state_traj, tmp_barrier_functions, tmp_Lyapunov_functions, tmp_relax_values, tmp_control_inputs = stabilizer.simulate(initial_state, np.array([0, 0]), steps)
        plt.scatter(initial_state[0], initial_state[1], color='black', marker='x', s=100, label='Start' if i == 0 else "")
        plt.plot(np.array(state_traj)[:, 0], np.array(state_traj)[:, 1], color=color, linestyle=style, linewidth=4, label=label)
            
        if color == 'orange':
            barrier_functions = tmp_barrier_functions
            Lyapunov_functions = tmp_Lyapunov_functions
            relax_values = tmp_relax_values
            controls = tmp_control_inputs

    equilibrium_x = 0
    equilibrium_y = 0
    plt.scatter(equilibrium_x, equilibrium_y, color='green', marker='o', s=200, label='Equilibrium')  # Adjust 's' for size



    plt.xlabel('Angle (rad)', fontsize=20)
    plt.ylabel('Angular Velocity (rad/s)', fontsize=20)
    #plt.title('Inverted Pendulum: Safe Region and Trajectories', fontsize=20)
    safe_patch = mpatches.Patch(color='green', alpha=0.2, hatch='/', label='Safe Region')
    plt.xticks(fontsize=18)  # Adjust fontsize as needed for x-axis
    plt.yticks(fontsize=18)  # Adjust fontsize as needed for y-axis
    # Create a custom legend entry for the initial states
    #initial_state_marker = plt.Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=10, label='Initial State')
    #equilibrium_marker = plt.Line2D([0], [0], marker='o', color='gold', linestyle='None', markersize=12, label='Equilibrium')

    # Combine custom entries with existing lines for the legend
    #plt.legend(handles=[safe_patch, initial_state_marker] + [plt.Line2D([0], [0], color=c, lw=4, linestyle=s, label=l) for c, s, l in zip(colors, styles, trajectory_labels)], fontsize=16, loc='lower left')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('inverted_pendulum_switch.png', dpi=300)
    plt.show()
    
    return barrier_functions, Lyapunov_functions, relax_values, controls

def plot_values_and_control(barrier_functions, Lyapunov_functions, relax_values, control_inputs):
    time_steps = np.arange(len(barrier_functions))
    
    dt = 0.01  # Time discretization
    time = time_steps * dt  # Convert time steps to actual time

    # Find the first index where the barrier function becomes positive
    switch_index = np.argmax(np.array(barrier_functions) > 0)  # Using argmax to find the first True index
    switch_time = time[switch_index] if switch_index < len(time) else None  # Ensure index is within bounds

    # Create a new figure for the Lyapunov and barrier functions
    plt.figure(figsize=(12, 6))

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
    plt.savefig('inverted_function_values_over_time.png', dpi=300)
    plt.show()
    
def create_standalone_legend():
    # Define legend elements
    safe_patch = mpatches.Patch(color='green', alpha=0.2, hatch='/', label='Safe Region')
    initial_state_marker = plt.Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=10, label='Initial States')  # Increased markersize
    equilibrium_marker = plt.Line2D([0], [0], marker='o', color='green', linestyle='None', markersize=10, label='Equilibrium')  # Increased markersize
    trajectory_lines = [
        mlines.Line2D([], [], color='b', linewidth=2, linestyle='--', label='Traj1'),
        mlines.Line2D([], [], color='purple', linewidth=2, linestyle='-.', label='Traj2'),
        mlines.Line2D([], [], color='orange', linewidth=2, linestyle=':', label='Traj3'),
        mlines.Line2D([], [], color='r', linewidth=2, linestyle='-', label='Traj4')
    ]

    # Create a figure specifically for the legend
    fig = plt.figure(figsize=(10, 2))  # Adjust figure size as needed
    ax = fig.add_subplot(111)
    ax.axis('off')

    legend_elements = [safe_patch, initial_state_marker, equilibrium_marker] + trajectory_lines
    # Create legend with larger fontsize
    legend = ax.legend(handles=legend_elements, loc='center', ncol=7, fontsize='large', frameon=False)

    # Adjust layout to minimize whitespace
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # Save the legend as a separate figure with minimal padding
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('legend_large.png', dpi=300, bbox_inches=bbox)
    
def plot_feasibility_map(stabilizer, desired_state):
    theta_range = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    theta_dot_range = np.linspace(-6, 6, 100)
    theta, theta_dot = np.meshgrid(theta_range, theta_dot_range)
    
    feasibility_map = np.zeros_like(theta, dtype=bool)

    feasible_theta = []
    feasible_theta_dot = []
    # Plotting the state space and safe regions
    plt.figure(figsize=(8, 6))
    
    epsilon1 = stabilizer.epsilon1
    epsilon2 = stabilizer.epsilon2
    h1 = 1/epsilon1 * theta + theta_dot
    h2 = -theta_dot - epsilon2 * theta 
    h = np.minimum(h1, h2)
    h_tilde = np.minimum(-h1, -h2)
    plt.contourf(theta, theta_dot, h, levels=[0, np.inf], colors='lightgreen', alpha=0.5)
    plt.contourf(theta, theta_dot, h_tilde, levels=[0, np.inf], colors='lightgreen', alpha=0.5)

    for i in range(len(theta_range)):
        for j in range(len(theta_dot_range)):
            current_state = np.array([theta[i, j], theta_dot[i, j]])
            try:
                control, _, h, V = stabilizer.CLF_CBF_QP(current_state, desired_state)
                if control is not None:  # Feasible if control is not None
                    feasible_theta.append(theta[i, j])
                    feasible_theta_dot.append(theta_dot[i, j])
                #else:
                    #print('h1_value:', h1[i,j])
                    #print('h2_value:', h2[i,j])         
            except Exception as e:
                pass  # Infeasible or error occurred



    # Scatter plot for feasible states
    plt.scatter(feasible_theta, feasible_theta_dot, color='red', s=1, label='Feasible States')

    plt.xlabel('Theta (rad)')
    plt.ylabel('Theta dot (rad/s)')
    plt.title('CLF-CBF QP Feasibility Map')
    plt.legend()
    plt.grid(True)
    plt.savefig('inverted_pendulum_feasible_map (rateh=30).png', dpi=300)
    plt.show()
    
def plot_control_policy(stabilizer, desired_state):
    theta_range = np.linspace(-np.pi, np.pi, 100)
    theta_dot_range = np.linspace(-6, 6, 100)
    Theta, Omega = np.meshgrid(theta_range, theta_dot_range)

    U_values = np.zeros_like(Theta)

    for i in range(len(theta_range)):
        for j in range(len(theta_dot_range)):
            current_state = np.array([Theta[i, j], Omega[i, j]])
            try:
                control, _, h, V = stabilizer.CLF_CBF_QP(current_state, desired_state)
                if control is not None:
                    U_values[i, j] = control
                else:
                    U_values[i, j] = np.nan  # Mark as NaN if infeasible or control is None
            except Exception as e:
                U_values[i, j] = np.nan  # Mark as NaN if error occurred
                

    # Plot the control policy
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(Theta, Omega, U_values, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Control input u')
    plt.xlabel('Theta (rad)')
    plt.ylabel('Theta dot (rad/s)')
    plt.title("Control Policy for Inverted Pendulum")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Inverted_pendulum_control_policy.png", dpi=300)
    plt.show()




class InvertedPendulumStabilizer:
    def __init__(self, epsilon1, epsilon2, dt):
        self.dt = dt  # time step
        self.g = 1.0  # gravity
        self.m = 1.0   # mass of the pendulum
        self.l = 1.0   # length of the pendulum
        
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2

    def dynamics(self, state, force):
        # Unpack the state
        theta, theta_dot = state
        
        # Equation of motion for inverted pendulum
        theta_ddot = (self.g / self.l) * np.sin(theta) + (1 / (self.m * self.l**2)) * force

        # Update the state
        new_theta = theta + theta_dot * self.dt
        new_theta_dot = theta_dot + theta_ddot * self.dt

        return np.array([new_theta, new_theta_dot])
    
    def CLF_QP(self, current_state, desired_state, rateV = 1.0):
        
        '''
        here defines a valid CLF
        '''
        theta, theta_dot = current_state
        theta_d, theta_dot_d = desired_state

        # Quadratic Lyapunov function
        V = np.cos(theta) - 1 + 1/4 * theta**2 + 1/2 * theta_dot**2 - 1/2 * theta * theta_dot

        # Derivative of V
        dVdstate = np.array([-np.sin(theta) + 1/2 * theta - 1/2 * theta_dot, theta_dot - 1/2 * theta])

        # Control inputs
        control = cp.Variable()
        
        delta = cp.Variable()

        # Dynamics of the inverted pendulum
        theta_ddot = (self.g / self.l) * np.sin(theta) + (1 / (self.m * self.l**2)) * control
        

        # Lie derivative of V
        dot_V = dVdstate @ np.array([theta_dot, theta_ddot])
        

        # Constraints for decrease rate of V
        
        baseline_constraints = [delta >= 0]
        
        baseline_constraints.append(dot_V + rateV * V <= 0)
        
        
        #print('lyapunov:', V)

        # Objective: Minimize control effort
        
        p2 = 1e2
        
        #objective = cp.Minimize(cp.square(control) + p2 * cp.square(delta))
        objective = cp.Minimize(cp.square(control) + p2 * cp.square(delta))
    
        constraints = baseline_constraints 

        # Setup and solve the QP
        problem = cp.Problem(objective, constraints)
        
        #problem.solve()
        problem.solve(solver='SCS', verbose=False)
        
        print('relax:', delta.value)
        
        

        return control.value, delta.value, 0, V
        
    
    def CLF_CBF_QP(self, current_state, desired_state, rateV = 1.0):
        theta, theta_dot = current_state
        theta_d, theta_dot_d = desired_state

        # Quadratic Lyapunov function
        V = (theta - theta_d)**2 + (theta_dot - theta_dot_d)**2

        # Derivative of V
        dVdstate = 2 * np.array([theta - theta_d, theta_dot - theta_dot_d])

        # Control inputs
        control = cp.Variable()
        
        delta = cp.Variable()

        # Dynamics of the inverted pendulum
        theta_ddot = (self.g / self.l) * np.sin(theta) + (1 / (self.m * self.l**2)) * control
        

        # Lie derivative of V
        dot_V = dVdstate @ np.array([theta_dot, theta_ddot])
        

        # Constraints for decrease rate of V
        
        baseline_constraints = [delta >= 0]
        
        baseline_constraints.append(dot_V + rateV * V <= delta)
        
        '''
        add a control constraint for verification purpose
        
        '''
        #baseline_constraints.append(cp.abs(control) <= 12)
        
        
        '''
        paper draft idea of defining h:
        '''
        
        rateh = 3.0
        
        h1 = -1/self.epsilon1 * theta - theta_dot
        h2 = theta_dot + self.epsilon2 * theta
        


        # Define the combined barrier function h
        h = min(h1, h2)
        #h = h1
        
        # Derivative of h
        dh_dstate = np.array([-1/self.epsilon1, -1]) if h1 <= h2 else np.array([self.epsilon2, 1])
        
        dot_h = dh_dstate @ np.array([theta_dot, theta_ddot])
        
        '''
        following is the symmetry safe set in the 4th quadrant, comment out to match with the proof in the paper
        '''
        
        # h_tilde = min(-h1, -h2)
        
        # dh_tilde_dstate = np.array([1/self.epsilon1, 1]) if -h1 <= -h2 else np.array([-self.epsilon2, -1])
        # dot_h_tilde = dh_tilde_dstate @ np.array([theta_dot, theta_ddot])
        
        # epsilon_t = 0.001
        # #if the initial state is at 4th quadrant 
        # if(h1 > 0):
        #     #print('barrier value:', h)
        #     baseline_constraints.append(dot_h + rateh * h >= 0) 
            
        #     h = h
            
        # if(h1 <= 0):
        #     baseline_constraints.append(dot_h_tilde + rateh * h_tilde >= 0) 
            
        #     h = h_tilde
        
        epsilon_t = 0.001
        
        '''
        comment the following line to check the controller behaviour with no CBF
        '''
        baseline_constraints.append(dot_h + rateh * h >= epsilon_t)

        p2 = 1e3
        
        objective = cp.Minimize(cp.square(control) + p2 * cp.square(delta))
        #objective = cp.Minimize(cp.square(control))
    
        constraints = baseline_constraints 

        # Setup and solve the QP
        problem = cp.Problem(objective, constraints)
        
        #problem.solve()
        problem.solve(solver='SCS', verbose=False) 
        

        return control.value, delta.value, h, V
    
    def CLF_CBF_Switching_QP(self, current_state, desired_state, rateV=1.0):
        theta, theta_dot = current_state
        theta_d, theta_dot_d = desired_state
        
        # Quadratic Lyapunov function
        V = (theta - theta_d)**2 + (theta_dot - theta_dot_d)**2
        
        # Derivative of V
        dVdstate = 2 * np.array([theta - theta_d, theta_dot - theta_dot_d])
        
        # Control input
        control = cp.Variable()
        
        # Dynamics
        theta_ddot = (self.g / self.l) * np.sin(theta) + (1 / (self.m * self.l**2)) * control
        
        # Lie derivative of V
        dot_V = dVdstate @ np.array([theta_dot, theta_ddot])
        
        # Barrier function
        constraints = []
        
        rateh = 3.0
        
        h1 = -1/self.epsilon1 * theta - theta_dot
        h2 = theta_dot + self.epsilon2 * theta
        

        # Define the combined barrier function h
        h = min(h1, h2)
        #h = h1
        
        # Derivative of h
        dh_dstate = np.array([-1/self.epsilon1, -1]) if h1 <= h2 else np.array([self.epsilon2, 1])
        
        dot_h = dh_dstate @ np.array([theta_dot, theta_ddot])
        
        '''
        following is the symmetry safe set in the 4th quadrant, comment out to match with the proof in the paper
        '''
        
        # h_tilde = min(-h1, -h2)
        
        # dh_tilde_dstate = np.array([1/self.epsilon1, 1]) if -h1 <= -h2 else np.array([-self.epsilon2, -1])
        # dot_h_tilde = dh_tilde_dstate @ np.array([theta_dot, theta_ddot])
        
        # epsilon_t = 0.001
        # #if the initial state is at 4th quadrant 
        # if(h1 > 0):
        #     #print('barrier value:', h)
        #     baseline_constraints.append(dot_h + rateh * h >= 0) 
            
        #     h = h
            
        # if(h1 <= 0):
        #     baseline_constraints.append(dot_h_tilde + rateh * h_tilde >= 0) 
            
        #     h = h_tilde
       

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
            
            constraints.append(dot_V + rateV * V <= 0)
            constraints.append(dot_h + rateh * h >= 0)
            objective = cp.Minimize(cp.square(control))
            relax_value = 0
            problem = cp.Problem(objective, constraints)
            problem.solve(solver='SCS', verbose=False)
            
        #print('control:', control.value)
        #print('barrier_value:', h)
        #print('currnet_state:', current_state)
        
        
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
            pure CLF-QP controller
            '''
            #force, delta,  h, V = self.CLF_QP(state, desired_state)
            
            
            '''
            Lipschitz controller, relaxed CLF constraint over time
            '''
            #force, delta , h, V = self.CLF_CBF_QP(state, desired_state)
            
            '''
            Switching strategy, switch to CLF-CBF QP (no relaxation) once CLF and CBF are compatiable
            '''
            
            force, delta , h, V = self.CLF_CBF_Switching_QP(state, desired_state)
            

            state = self.dynamics(state, force)
            
            
            barrier_functions.append(h)
            
            relax_values.append(delta)
            
            
            Lyapunov_functions.append(V)
            
            control_inputs.append(force)

            # Stopping condition (optional)
            if np.abs(state[0] - desired_state[0]) < 0.05 and np.abs(state[1]) < 0.05:
                print("Pendulum stabilized!")
                break
            
            # only for CLF-QP to compare results
            # if np.abs(state[1]) < 0.2:
            #     print("Actuation Lost!")
            #     print('state:', state)
            #     state_traj.append(np.array([state[0], 0]))
            #     break

        return state, state_traj, barrier_functions, Lyapunov_functions, relax_values, control_inputs


def main():
    
    initial_states = [
        [np.pi, 3],    # Quadrant 1
        [np.pi/1.5, -4],   # Quadrant 2
        [-np.pi/1.2, -3],  # Quadrant 3
        [-np.pi/2, 4]    # Quadrant 4
    ]
    
    dt = 0.01           # simulate time discretization
    steps = 500           # total time step for simulation

    epsilon1 = 0.1
    epsilon2 = 1.2
    
    
    desired_state = np.array([0, 0])
    
    stabilizer = InvertedPendulumStabilizer(epsilon1, epsilon2, dt)
    
    '''
    for feasibility check
    '''
    
    #plot_feasibility_map(stabilizer, desired_state)
    
    '''
    plot control policy
    '''
    
    #plot_control_policy(stabilizer, desired_state)

    barrier_functions, Lyapunov_functions, relax_values, controls = plot_state_space_and_trajectories(initial_states, stabilizer, steps)
    
    plot_values_and_control(barrier_functions, Lyapunov_functions, relax_values, controls)
    
    #create_standalone_legend()



if __name__ == "__main__":
    main()
