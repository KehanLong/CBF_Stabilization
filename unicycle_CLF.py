#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import cvxpy as cp
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

def plot_unicycle_trajectories(initial_state, goal_state, epsilon, dt, steps):
    stabilizer = UnicycleStabilizer(epsilon, dt)
    state_traj, barrier_functions, Lyapunov_functions, relax_values, controls = stabilizer.simulate(initial_state, goal_state, steps)
    
    # Plotting the trajectory of the unicycle
    plt.figure(figsize=(8, 6))
    plt.plot(np.array(state_traj)[:, 0], np.array(state_traj)[:, 1], 'b-', linewidth=2, label='Trajectory')
    plt.scatter([initial_state[0]], [initial_state[1]], color='g', marker='o', label='Start')
    plt.scatter([goal_state[0]], [goal_state[1]], color='r', marker='x', label='Goal')
    
    plt.xlabel('X Position', fontsize=14)
    plt.ylabel('Y Position', fontsize=14)
    plt.title('Unicycle Trajectory', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plotting function values and control inputs
    plot_values_and_control(barrier_functions, Lyapunov_functions, relax_values, controls)

def plot_values_and_control(barrier_functions, Lyapunov_functions, relax_values, control_inputs):
    time_steps = np.arange(len(barrier_functions))

    # Extract linear and angular velocities from control_inputs
    linear_velocities = [control[0] for control in control_inputs]
    angular_velocities = [control[1] for control in control_inputs]

    plt.figure(figsize=(12, 8))

    # Plot Barrier and Lyapunov Functions
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, barrier_functions, 'g--', label='Barrier Function', linewidth=2)
    plt.plot(time_steps, Lyapunov_functions, 'b-', label='Lyapunov Function', linewidth=2)
    plt.plot(time_steps, relax_values, 'r-.', label='Relaxation', linewidth=2)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Function Value', fontsize=14)
    plt.title('Function Values Over Time', fontsize=16)
    plt.legend(fontsize=12)

    # Plot Control Inputs: Linear and Angular Velocity
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, linear_velocities, 'c-', label='Linear Velocity', linewidth=2)
    plt.plot(time_steps, angular_velocities, 'm-', label='Angular Velocity', linewidth=2)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Control Input', fontsize=14)
    plt.title('Control Input Over Time', fontsize=16)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('unicycle_function_values_and_controls_combined.png', dpi=300)
    plt.show()


class UnicycleStabilizer:
    def __init__(self, epsilon, dt):
        self.max_v = 1.0
        self.max_omega = 2.0
        self.dt = dt  # time step
        
        self.prev_v = 0
        self.prev_omega = 0
        
        self.eps = epsilon
        
    def dynamics(self, state, linear_v, angular_w):
        # Unpack the state
        x, y, theta = state
        
        # Equation of motion for inverted pendulum
        theta_dot = angular_w
        new_theta = theta + theta_dot * self.dt
        
        x_dot = linear_v * np.cos(new_theta)
        y_dot = linear_v * np.sin(new_theta)

        # Update the state
        new_x = x + x_dot * self.dt
        new_y = y + y_dot * self.dt
        
        return np.array([new_x, new_y, new_theta])
        
    def CLF_CBF_QP(self, current_state, desired_state, rateV = 1.0, rateh = 0.05):
        x, y, theta = current_state
        x_d, y_d, theta_d = desired_state

        # Quadratic Lyapunov function
        V = (x - x_d)**2 +  (y - y_d)**2 + (theta - theta_d)**2

        # Derivative of V
        dVdstate = 2 * np.array([(x - x_d), (y - y_d), theta - theta_d])

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
        
        baseline_constraints.append(dot_V + rateV * V <= delta)
        
        
        #optional: some control constraints:
        # additional_constraints = [
        #     cp.abs(v) <= self.max_v,
        #     cp.abs(omega) <= self.max_omega
            
        # ]
        
        # idea: add the 'bad' state set as a CBF unsafe set 
        
        
        
        
        # '''
        # h: the only safe set is the line defined by the orientation and position of the desired state
        # '''
        # cos_hat = (x_d - x) / np.sqrt((x_d-x)**2 + (y_d-y)**2)
        
        
        # sin_hat = (y_d - y) / np.sqrt((x_d-x)**2 + (y_d-y)**2)
        
        
        # # Partial derivatives of cos_hat and sin_hat w.r.t x and y
        # # Derivatives of the numerator and denominator for cos_hat w.r.t x
        # exp1 = x_d - x
        # du_dx = -1
        # exp2 = np.sqrt((x_d-x)**2 + (y_d-y)**2)
        # dv_dx = (x - x_d) / exp2
        
        # # Derivative of cos_hat w.r.t x
        # d_cos_hat_dx = (du_dx * exp2 - exp1 * dv_dx) / (exp2**2)
        
        # # Similarly, compute d_sin_hat_dx, d_cos_hat_dy, and d_sin_hat_dy
        # dv_dy = -(y_d - y) / exp2  # Derivative of the denominator for sin_hat w.r.t y
        
        # # Derivative of sin_hat w.r.t x
        # d_sin_hat_dx = -dv_dx * (y_d - y) / (exp2**2)
        
        # # Derivative of cos_hat w.r.t y
        # d_cos_hat_dy = -dv_dy * (x_d - x) / (exp2**2)
        
        # # Derivative of sin_hat w.r.t y
        # exp1 = y_d - y
        # du_dy = -1
        # d_sin_hat_dy = (du_dy * exp2 - exp1 * dv_dy) / (exp2**2)

        
        # h = - self.eps * (cos_hat - np.cos(theta_d))**2 - self.eps * (sin_hat - np.sin(theta_d))**2
        
        # dh_dx =  -self.eps * 2 * (cos_hat - np.cos(theta_d)) * d_cos_hat_dx - self.eps * 2 * (sin_hat - np.sin(theta_d)) * d_sin_hat_dx
        # dh_dy =  -self.eps * 2 * (cos_hat - np.cos(theta_d)) * d_cos_hat_dy - self.eps * 2 * (sin_hat - np.sin(theta_d)) * d_sin_hat_dy
        
        
        # dh_dtheta = 0
        
        
        
        '''
        paper draft idea of defining h:
        '''
        
        
        h = self.eps - (-(x - x_d) * np.sin(theta) + (y - y_d) * np.sin(theta))**2
        
        # Calculating the partial derivatives
        dh_dx = -2 * (- (x - x_d) * np.sin(theta) + (y - y_d) * np.cos(theta)) * (-np.sin(theta))
        dh_dy = -2 * (- (x - x_d) * np.sin(theta) + (y - y_d) * np.cos(theta)) * np.cos(theta)
        dh_dtheta = -2 * (- (x - x_d) * np.sin(theta) + (y - y_d) * np.cos(theta)) * (-(x - x_d) * np.cos(theta) - (y - y_d) * np.sin(theta))


        '''
        add the safety constraint
        '''


        dhdstate = np.array([dh_dx, dh_dy, dh_dtheta]) 
        
        dot_h = dhdstate @ np.array([v * np.cos(theta), v * np.sin(theta), omega])
        
        
        
        baseline_constraints.append(dot_h + rateh * h >= 0) 
        
    

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
    
    def simulate(self, initial_state, desired_state, steps):
        state = initial_state
        
        state_traj = []
        
        barrier_functions = []
        
        Lyapunov_functions = []
        
        relax_values = []
        
        control_inputs = []

        for step in range(steps):
            
            state_traj.append(state)
            
            linear_v, angular_w , _, _, delta, h, V = self.CLF_CBF_QP(state, desired_state)

            state = self.dynamics(state, linear_v, angular_w)
            
            barrier_functions.append(h)
            
            relax_values.append(delta)
            
            
            Lyapunov_functions.append(V)
            
            control_inputs.append([linear_v, angular_w])

            # Stopping condition
            if np.linalg.norm(state - desired_state) < 0.1:
                print("Car Reached the desired state!")
                break

        return state_traj, barrier_functions, Lyapunov_functions, relax_values, control_inputs





def main():

    # Initial and desired states
    
    epsilon = 0.2
    
    dt = 0.02
    steps = 500
    
    initial_state = np.array([0., 0., 0], dtype=float)

    goal_state = np.array([3., 1., np.pi/2], dtype=float)
    
    
    
    plot_unicycle_trajectories(initial_state, goal_state, epsilon, dt, steps)

        

if __name__ == "__main__":
    main()
