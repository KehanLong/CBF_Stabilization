#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import cvxpy as cp
import numpy as np

import matplotlib.pyplot as plt

class UnicycleStabilizer:
    def __init__(self, dt):
        self.max_v = 1.0
        self.max_omega = 2.0
        self.dt = dt  # time step
        
    def Full_CLF_QP(self, current_state, desired_state, rateV):
        x, y, theta = current_state
        x_d, y_d, theta_d = desired_state

        # Quadratic Lyapunov function
        V = 1 * (x - x_d)**2 + 1 * (y - y_d)**2 + (theta - theta_d)**2

        # Derivative of V
        dVdx = 2 * np.array([1 * (x - x_d), 1 * (y - y_d), theta - theta_d])

        # Control inputs
        v = cp.Variable()
        omega = cp.Variable()
        
        delta = cp.Variable()

        # Lie derivative of V
        dot_V = dVdx @ np.array([v * np.cos(theta), v * np.sin(theta), omega])
        
        #compute and plot \nabla V(x) * g(x)
        linear_gain = dVdx[:2] @ np.array([np.cos(theta), np.sin(theta)])
        angular_gain = dVdx[2] 

        # Constraints for decrease rate of V
        baseline_constraints = [dot_V + rateV * V <= 0, delta >= 0]
        
        
        #append some control constraints:
        additional_constraints = [
            cp.abs(v) <= self.max_v,
            cp.abs(omega) <= self.max_omega
            
        ]
        
        # idea: add the 'bad' state set as a CBF unsafe set 
        
        eps = 0.1
        
        rateh = 10.0
        
        p3 = 1e3
        
        #h = linear_gain ** 2 + angular_gain **2 - eps
        
        
        
        
        
        # Compute partial derivatives of linear_gain with respect to x, y
        # partial_linear_gain_x = dVdx[0] * np.cos(theta)
        # partial_linear_gain_y = dVdx[1] * np.sin(theta)
        
        # # Compute partial derivative of linear_gain with respect to theta
        # partial_linear_gain_theta = -dVdx[0] * np.sin(theta) + dVdx[1] * np.cos(theta)
        
        # # Compute partial derivative of angular_gain with respect to theta
        # partial_angular_gain_theta = dVdx[2]
        
        # # Now compute the derivatives of h with respect to the state variables
        # dh_dx = 2 * linear_gain * partial_linear_gain_x
        # dh_dy = 2 * linear_gain * partial_linear_gain_y
        # dh_dtheta = 2 * linear_gain * partial_linear_gain_theta + 2 * angular_gain * partial_angular_gain_theta
        
        # # Combine them into the gradient of h
        # dhdx = np.array([dh_dx, dh_dy, dh_dtheta])

        
        #dot_h = dhdx @ np.array([v * np.cos(theta), v * np.sin(theta), omega])
        
        
        '''
        one idea that works
        '''
        
        # h = x_d - x + 0.3
        
        # dh_dx = -1
        # dh_dy = 0
        # dh_dtheta = 0
        

        
        '''
        new idea of defining h
        '''
        cos_hat = (x_d - x) / np.sqrt((x-x_d)**2 + (y-y_d)**2)
        
        
        sin_hat = (y_d - y) / np.sqrt((x-x_d)**2 + (y-y_d)**2)
        

        
        # Partial derivatives of cos_hat and sin_hat w.r.t x and y
        d_cos_hat_dx =  - ((y - y_d)**2) / ((x - x_d)**2 + (y - y_d)**2)**(3/2)
        d_sin_hat_dx = (x - x_d) * (y - y_d) / ((x - x_d)**2 + (y - y_d)**2)**(3/2)

        d_cos_hat_dy = (x - x_d) * (y - y_d) / ((x - x_d)**2 + (y - y_d)**2)**(3/2)
        d_sin_hat_dy = - ((x - x_d)**2) / ((x - x_d)**2 + (y - y_d)**2)**(3/2)

        
        h = (theta - theta_d) ** 2 - eps * (cos_hat - np.cos(theta_d))**2 - eps * (sin_hat - np.sin(theta_d))**2
        
        dh_dx =  -eps * 2 * cos_hat * d_cos_hat_dx - eps * 2 * sin_hat * d_sin_hat_dx
        dh_dy =  -eps * 2 * cos_hat * d_cos_hat_dy - eps * 2 * sin_hat * d_sin_hat_dy
        
        dh_dtheta = 2 * (theta - theta_d)
        
        
        
        
        dhdx = np.array([dh_dx, dh_dy, dh_dtheta])
        
        dot_h = dhdx @ np.array([v * np.cos(theta), v * np.sin(theta), omega])
        
        baseline_constraints.append(dot_h + rateh * h >= -delta) 
        
        
        
        print('barrier value:', h)
        
        print('lyapunov:', V)

        # Objective: Minimize control effort
        objective = cp.Minimize( cp.square(v - self.max_v) + cp.square(omega) + p3 * cp.square(delta)) 
        
        #objective = cp.Minimize(p3 * cp.square(delta)) 
        
        constraints = baseline_constraints 

        # Setup and solve the QP
        problem = cp.Problem(objective, constraints)
        
        #problem.solve()
        problem.solve(solver='SCS', verbose=False)
        
        print('relax:', delta.value)
        
        
        
        

        return v.value, omega.value, linear_gain, angular_gain, delta.value

    def Pos_CLF_QP(self, current_state, desired_pos, rateV):
        x, y, theta = current_state
        x_d, y_d = desired_pos

        # Quadratic Lyapunov function for position only
        V_pos = (x - x_d)**2 + (y - y_d)**2

        # Derivative of V_pos
        dVdx_pos = 2 * np.array([x - x_d, y - y_d])

        # Control inputs
        v = cp.Variable()
        omega = cp.Variable()

        # Lie derivative of V_pos
        dot_V = dVdx_pos @ np.array([v * np.cos(theta), v * np.sin(theta)])

        # Constraints for decrease rate of V_pos
        constraints = [dot_V + rateV * V_pos <= 0]

        # Objective: Minimize control effort
        objective = cp.Minimize( cp.square(v - self.max_v) + cp.square(omega))  

        # Setup and solve the QP
        problem = cp.Problem(objective, constraints)
        
        problem.solve()
        
        #problem.solve(solver='SCS', verbose=False)
        
        #problem.solve()
        


        return v.value, omega.value
        
        



def main():
    # Time step
    dt = 0.02
    rateV = 1.0  # Design parameter for the rate of decrease of the Lyapunov function

    # Create an instance of UnicycleStabilizer
    stabilizer = UnicycleStabilizer(dt)

    # Initial and desired states
    
    current_state = np.array([0., 0., 0], dtype=float)

    desired_state = np.array([3., 3., np.pi/2], dtype=float)  # x_d, y_d, theta_d

    # Lists to store simulation data for plotting
    x_list = []
    y_list = []
    theta_list = []
    v_list = []
    omega_list = []
    time_list = []
    
    linear_gain_list = []
    angular_gain_list = []
    
    delta_list = []

    # Simulation loop
    for step in range(1000):
        # Compute control inputs using one of the methods
        v, omega, linear_gain, angular_gain, delta = stabilizer.Full_CLF_QP(current_state, desired_state, rateV)
        # v, omega = stabilizer.Pos_CLF_QP(current_state, desired_state[:2], rateV)
    


        # Update the state using the computed control inputs
        current_state[0] += v * np.cos(current_state[2]) * dt

        
        #print(temp)
        current_state[1] += v * np.sin(current_state[2]) * dt
        current_state[2] += omega * dt
        
        
    
        # Append to lists for plotting
        x_list.append(current_state[0])
        y_list.append(current_state[1])
        theta_list.append(current_state[2])
        v_list.append(v)  
        omega_list.append(omega)  
        time_list.append(step  * dt)
        
        delta_list.append(delta)
        
        linear_gain_list.append(linear_gain)
        angular_gain_list.append(angular_gain)
    
        # Stopping condition
        if np.linalg.norm(current_state - desired_state) < 0.2:
            print("Reached the desired state!")
            break



    arrow_scale = 10
    arrow_width = 0.015
    # Plot the robot trajectory
    plt.figure(figsize=(18, 6), dpi = 300)
    plt.subplot(1, 3, 1)
    plt.plot(x_list, y_list, label='Trajectory')
    plt.plot(desired_state[0], desired_state[1], 'ro', label='Desired Position')
    plt.quiver(x_list[-1], y_list[-1], np.cos(theta_list[-1]), np.sin(theta_list[-1]), scale=arrow_scale, color='r', width=arrow_width)
    plt.title('Robot Trajectory')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.grid(True)
    plt.legend()

    # Plot the control inputs
    plt.subplot(1, 3, 2)
    plt.plot(time_list, v_list, label='v (linear velocity)')
    plt.plot(time_list, omega_list, label='omega (angular velocity)')
    plt.title('Control Inputs over Time')
    plt.xlabel('Time')
    plt.ylabel('Control Input')
    plt.grid(True)
    plt.legend()
    
    
    #plot for orientation
    plt.subplot(1, 3, 3)
    plt.plot(time_list, theta_list, label='Theta (orientation)')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (rad)')
    plt.title('Orientation Over Time')
    plt.legend()
    

    plt.tight_layout()
    plt.savefig('robot_traj', dpi = 300)

    plt.figure(dpi = 150)
    plt.plot(time_list, linear_gain_list, label='Linear Gain')
    plt.xlabel('Time (s)')
    plt.ylabel('Linear Gain')
    plt.title('Linear Gain Over Time')
    plt.legend()
    plt.savefig('linear_gain', dpi=300)
    
    # Plot for angular gain
    plt.figure(dpi = 150)
    plt.plot(time_list, angular_gain_list, label='Angular Gain')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Gain')
    plt.title('Angular Gain Over Time')
    plt.legend()
    plt.savefig('angular_gain', dpi=300)
    plt.show()
    
    # Plot for angular gain
    plt.figure(dpi = 150)
    plt.plot(time_list, delta_list, label='barrier penalty')
    plt.xlabel('Time (s)')
    plt.ylabel('barrier pennalty')
    plt.title('barrier penalty over time')
    plt.legend()
    plt.savefig('barrier_pnality', dpi=300)
    plt.show()
        

if __name__ == "__main__":
    main()
