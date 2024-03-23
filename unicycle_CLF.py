#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import cvxpy as cp
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

def plot_unicycle_trajectories(initial_states, goal_state, stabilizer, steps):
    colors = ['b', 'purple', 'orange', 'r']  # Colors for trajectories
    styles = ['--', '-.', ':', '-']
    trajectory_labels = ['traj1', 'traj2', 'traj3', 'traj4']  # Labels for each trajectory

    plt.figure(figsize=(8, 8))
    for i, (initial_state, color, style, label) in enumerate(zip(initial_states, colors, styles, trajectory_labels)):
        _, state_traj, tmp_barrier_functions, tmp_Lyapunov_functions, tmp_relax_values, tmp_control_inputs = stabilizer.simulate(initial_state, goal_state, steps)
        
        plt.scatter(initial_state[0], initial_state[1], color='black', marker='x', s=100, label='Start' if i == 0 else "")
        plt.plot(np.array(state_traj)[:, 0], np.array(state_traj)[:, 1], color=color, linestyle=style, linewidth=4, label=label)

        # Plot arrows for orientation at initial and final states
        initial_orientation = initial_state[2]
        final_orientation = state_traj[-1][2]
       
        plt.arrow(initial_state[0], initial_state[1], 0.5*np.cos(initial_orientation), 0.5*np.sin(initial_orientation), head_width=0.2, head_length=0.2, fc=color, ec=color)
        plt.arrow(state_traj[-1][0], state_traj[-1][1], 0.5*np.cos(final_orientation), 0.5*np.sin(final_orientation), head_width=0.2, head_length=0.2, fc=color, ec=color)
        
        if color == 'b':
            barrier_functions = tmp_barrier_functions
            #print('h:', barrier_functions)
            Lyapunov_functions = tmp_Lyapunov_functions
            relax_values = tmp_relax_values
            controls = tmp_control_inputs

    # Plot goal region as a green disk
    goal_region = plt.Circle((goal_state[0], goal_state[1]), 0.4, color='green', alpha=0.2, label='Goal Region')
    plt.gca().add_patch(goal_region)

    plt.xlabel('X1', fontsize=20)
    plt.ylabel('X2', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.tight_layout()

    # Create a custom legend
    custom_legend_elements = [mpatches.Patch(color='green', alpha=0.2, label='Goal Region'),
                              plt.Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=10, label='Start'),
                              plt.Line2D([0], [0], color='b', linewidth=4, linestyle='--', label='Traj1'),
                              plt.Line2D([0], [0], color='purple', linewidth=4, linestyle='-.', label='Traj2'),
                              plt.Line2D([0], [0], color='orange', linewidth=4, linestyle=':', label='Traj3'),
                              plt.Line2D([0], [0], color='r', linewidth=4, linestyle='-', label='Traj4')]
    #plt.legend(handles=custom_legend_elements, fontsize=20)

    plt.savefig('unicycle_switch.png', dpi=300)
    plt.show()
    
    return barrier_functions, Lyapunov_functions, relax_values, controls

def plot_values_and_control(barrier_functions, Lyapunov_functions, relax_values, control_inputs):
    time_steps = np.arange(len(barrier_functions))
    
    dt = 0.01  # Time discretization
    time = time_steps * dt  # Convert time steps to actual time

    # Extract linear and angular velocities from control_inputs
    linear_velocities = [control[0] for control in control_inputs]
    angular_velocities = [control[1] for control in control_inputs]

    plt.figure(figsize=(12, 8))

    # Plot Barrier and Lyapunov Functions
    plt.subplot(2, 1, 1)
    plt.plot(time, barrier_functions, 'g--', label='Barrier Function', linewidth=4)
    plt.plot(time, Lyapunov_functions, 'b-', label='Lyapunov Function', linewidth=4)
    plt.plot(time, relax_values, 'r-.', label='Relaxation', linewidth=4)
    plt.xlabel('Time Steps', fontsize=20)
    plt.ylabel('Function Value', fontsize=20)
    plt.title('Function Values Over Time', fontsize=22)
    
    plt.legend(fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Plot Control Inputs: Linear and Angular Velocity
    plt.subplot(2, 1, 2)
    plt.plot(time, linear_velocities, 'c-', label='Linear Velocity', linewidth=4)
    plt.plot(time, angular_velocities, 'm-', label='Angular Velocity', linewidth=4)
    plt.xlabel('Time Steps', fontsize=20)
    plt.ylabel('Control Input', fontsize=20)
    plt.title('Control Input Over Time', fontsize=22)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    #plt.savefig('unicycle_function_values_and_controls_combined.png', dpi=300)
    plt.show()


class UnicycleStabilizer:
    def __init__(self, epsilon, dt):
        self.dt = dt  # time step
        
        self.prev_v = 0
        self.prev_omega = 0
        
        self.eps = epsilon
        
    def dynamics(self, state, linear_v, angular_w):
        # Unpack the state
        x, y, theta = state
        
        # Equation of motion 
        theta_dot = angular_w
        new_theta = theta + theta_dot * self.dt
        
        x_dot = linear_v * np.cos(new_theta)
        y_dot = linear_v * np.sin(new_theta)

        # Update the state
        new_x = x + x_dot * self.dt
        new_y = y + y_dot * self.dt
        
        return np.array([new_x, new_y, new_theta])
        
    def CLF_CBF_QP(self, current_state, desired_state, rateV = 0.1):
        x, y, theta = current_state
        x_d, y_d, theta_d = desired_state

        # Quadratic Lyapunov function
        V = (x - x_d)**2 +  (y - y_d)**2 + 1 * (theta - theta_d)**2

        # Derivative of V
        dVdstate = 2 * np.array([(x - x_d), (y - y_d), 1 * (theta - theta_d)])

        # Control inputs
        v = cp.Variable()
        omega = cp.Variable()
        
        delta = cp.Variable()

        # Lie derivative of V
        dot_V = dVdstate @ np.array([v * np.cos(theta), v * np.sin(theta), omega])
        
        #compute and plot \nabla V(x) * g(x)
        linear_gain = dVdstate[:2] @ np.array([np.cos(theta), np.sin(theta)])
        angular_gain = dVdstate[2] 

        # Constraints for decrease rate of V
        
        baseline_constraints = [delta >= 0]
        
        baseline_constraints.append(dot_V + rateV * V <= 0)
        
        
        #optional: some control constraints:
        # additional_constraints = [
        #     cp.abs(v) <= self.max_v,
        #     cp.abs(omega) <= self.max_omega
            
        # ]
        
        # idea: add the 'bad' state set as a CBF unsafe set 
        
        
        
        
        
        '''
        paper idea of defining h:
        '''
        
        
        h1 = self.eps - (-(x - x_d) * np.sin(theta) + (y - y_d) * np.sin(theta))**2
        
        h2 = (x-x_d)**2 + (y-y_d)**2 - 1.5**2 * self.eps
        
        # Calculating the partial derivatives
        dh1_dx = -2 * (- (x - x_d) * np.sin(theta) + (y - y_d) * np.cos(theta)) * (-np.sin(theta))
        dh1_dy = -2 * (- (x - x_d) * np.sin(theta) + (y - y_d) * np.cos(theta)) * np.cos(theta)
        dh1_dtheta = -2 * (- (x - x_d) * np.sin(theta) + (y - y_d) * np.cos(theta)) * (-(x - x_d) * np.cos(theta) - (y - y_d) * np.sin(theta))
        
        dh2_dx = 2 * (x-x_d)
        dh2_dy = 2 * (y-y_d)
        dh2_dtheta = 0


        '''
        add the safety constraint
        '''


        
        h = min(h1, h2)
        
        rateh = 0.005
        
        
        # Derivative of h
        if h==h1:
            
            dh_dstate = np.array([dh1_dx, dh1_dy, dh1_dtheta]) 
        else:
            dh_dstate = np.array([dh2_dx, dh2_dy, dh2_dtheta]) 
        
        dot_h = dh_dstate @ np.array([v * np.cos(theta), v * np.sin(theta), omega])
        
        epsilon_t = 0.005
        
        
        #comment the following line to remove the CBF constraint
        #baseline_constraints.append(dot_h + rateh * h >= epsilon_t) 
        
    

        # Objective: Minimize control effort
        
        p3 = 1e2
        
        objective = cp.Minimize(cp.square(v - self.prev_v) + cp.square(omega - self.prev_omega) + p3 * cp.square(delta))
        
        #objective = cp.Minimize(p3 * cp.square(delta))
        
        constraints = baseline_constraints 

        # Setup and solve the QP
        problem = cp.Problem(objective, constraints)
        
        #problem.solve()
        problem.solve(solver='SCS', verbose=False)
        
        
        self.prev_v = v.value
        self.prev_omega = omega.value

        return v.value, omega.value, linear_gain, angular_gain, delta.value, h, V
    
    def CLF_CBF_Switching_QP(self, current_state, desired_state, rateV = 0.1):
        x, y, theta = current_state
        x_d, y_d, theta_d = desired_state

        # Quadratic Lyapunov function
        V = (x - x_d)**2 +  (y - y_d)**2 + 0.1*(theta - theta_d)**2

        # Derivative of V
        dVdstate = 2 * np.array([(x - x_d), (y - y_d), 0.1*(theta - theta_d)])

        # Control inputs
        v = cp.Variable()
        omega = cp.Variable()
        
        delta = cp.Variable()

        # Lie derivative of V
        dot_V = dVdstate @ np.array([v * np.cos(theta), v * np.sin(theta), omega])
        
        #compute and plot \nabla V(x) * g(x)
        linear_gain = dVdstate[:2] @ np.array([np.cos(theta), np.sin(theta)])
        angular_gain = dVdstate[2] 

        constraints = []
        
        
        '''
        paper draft idea of defining h:
        '''
        
        
        h1 = self.eps - (-(x - x_d) * np.sin(theta) + (y - y_d) * np.sin(theta))**2
        
        h2 = (x-x_d)**2 + (y-y_d)**2 - 1.5**2 * self.eps
        
        # Calculating the partial derivatives
        dh1_dx = -2 * (- (x - x_d) * np.sin(theta) + (y - y_d) * np.cos(theta)) * (-np.sin(theta))
        dh1_dy = -2 * (- (x - x_d) * np.sin(theta) + (y - y_d) * np.cos(theta)) * np.cos(theta)
        dh1_dtheta = -2 * (- (x - x_d) * np.sin(theta) + (y - y_d) * np.cos(theta)) * (-(x - x_d) * np.cos(theta) - (y - y_d) * np.sin(theta))
        
        dh2_dx = 2 * (x-x_d)
        dh2_dy = 2 * (y-y_d)
        dh2_dtheta = 0


        '''
        add the safety constraint
        '''


        
        h = min(h1, h2)
        
        
        
        # Derivative of h
        if h==h1:
            
            dh_dstate = np.array([dh1_dx, dh1_dy, dh1_dtheta]) 
        else:
            dh_dstate = np.array([dh2_dx, dh2_dy, dh2_dtheta]) 
        
        dot_h = dh_dstate @ np.array([v * np.cos(theta), v * np.sin(theta), omega])
        
        epsilon_t = 0.002
        
        rateh = 0.005
        
        
        
        if h < 0.0:
            
            delta = cp.Variable()
            constraints.append(delta >= 0)
            constraints.append(dot_V + rateV * V <= delta)
            constraints.append(dot_h + rateh * h >= epsilon_t)
            objective = cp.Minimize(cp.square(v - self.prev_v) + cp.square(omega - self.prev_omega) + 2e2 * cp.square(delta))
            # Solve QP
            problem = cp.Problem(objective, constraints)
            problem.solve(solver='SCS', verbose=False)
            relax_value = 0    
        else: 
            

            constraints.append(dot_V + rateV * V <= 0)
            constraints.append(dot_h + rateh * h >= 0)
            objective = cp.Minimize(cp.square(v - self.prev_v) + cp.square(omega - self.prev_omega))
            relax_value = 0
            problem = cp.Problem(objective, constraints)
            problem.solve(solver='SCS', verbose=False)
            
        self.prev_v = v.value
        self.prev_omega = omega.value
        

        return v.value, omega.value, linear_gain, angular_gain, relax_value, h, V
    
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
            Lipschitz controller
            '''
            
            #linear_v, angular_w , _, _, delta, h, V = self.CLF_CBF_QP(state, desired_state)
            
            '''
            switching controller
            '''
            linear_v, angular_w , _, _, delta, h, V = self.CLF_CBF_Switching_QP(state, desired_state)

            state = self.dynamics(state, linear_v, angular_w)
            
            barrier_functions.append(h)
            
            relax_values.append(delta)
            
            
            Lyapunov_functions.append(V)
            
            control_inputs.append([linear_v, angular_w])
            
            #print('car_state:', state)

            # Stopping condition
            if np.linalg.norm(state[0:2] - desired_state[0:2]) < 0.4 and np.abs(state[2] - desired_state[2]) < 0.01:
                print("Car Reached the desired state!")
                state_traj.append(np.array([state[0], state[1], 0]))
                
                break
            
            # clf only stop
            # if np.abs(state[2] - desired_state[2]) < 0.009:
            #     print("car loss actuation")
            #     state_traj.append(np.array([state[0], state[1], 0]))
            #     break
            

        return state, state_traj, barrier_functions, Lyapunov_functions, relax_values, control_inputs





def main():

    # Initial and desired states
    
    epsilon = 0.04
    
    initial_states = [
        [3., 1., np.pi],    
        [2., -3, np.pi/2],   
        [-3, -1, -np.pi/4], 
        [-2, 3, np.pi/4]    
    ]
    
    dt = 0.02           # simulate time discretization
    steps = 2000           # total time step for simulation

    
    goal_state = np.array([0, 0, 0])
    
    stabilizer = UnicycleStabilizer(epsilon, dt)
    
    barrier_functions, Lyapunov_functions, relax_values, controls = plot_unicycle_trajectories(initial_states, goal_state, stabilizer, steps)
    
    #plot_values_and_control(barrier_functions, Lyapunov_functions, relax_values, controls)

        

if __name__ == "__main__":
    main()
