import matplotlib.pyplot as plt
import control
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation


# global constants
g = 9.81  # Acceleration due to gravity in m/s^2 (Standard value) 

# local parameters
M = 0.5  # Mass of the cart in kilograms
m = 0.2  # Mass of the pendulum in kilograms
b = 0.1  # Coefficient of friction for the cart in N/m/sec
L = 0.3  # Length to pendulum center∏ of mass in meters
I = 0.006  # Mass moment of inertia of the pendulum in kg*m^2



def get_a_matrix(pend_mass=1, cart_mass=5, arm_length=2, g=-10, d=1, b=1):
    return np.array([
        [0, 1, 0, 0],
        [0, -d / cart_mass, b * pend_mass * g / cart_mass, 0],
        [0, 0, 0, 1],
        [0, -b * d / (cart_mass * arm_length), -b * (pend_mass + cart_mass) * g / (cart_mass * arm_length), 0]
    ])


def get_b_matrix(pend_mass=1, cart_mass=5, arm_length=2, g=-10, d=1, b=1):
    return np.array([[0, 1 / cart_mass, 0, b / (cart_mass * arm_length)]]).T

A = get_a_matrix()
B = get_b_matrix()
n = get_a_matrix().shape[0]

control_matrix = control.ctrb(A, B)
rank = np.linalg.matrix_rank(control_matrix)
print(f"Eigen values of matrix A: {list(np.linalg.eig(A)[0])}")
print(f"Rank of a control matrix C: rank={rank} (if rank={n} it is controllable)")


# Design LQR controller
Q = np.eye(4)   # np.diag(np.array([1, 1, 10, 100]))
R = .0001
K, S, E = control.lqr(A, B, Q, R)


def pendcart(x, t, pend_mass=1, cart_mass=5, arm_length=2, g=-10, d=1, s=1):
    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = pend_mass * (arm_length ** 2) * (pend_mass + cart_mass * (1 - Cx ** 2))
    
    wr = np.array([1, 0, np.pi, 0])
    
    u = np.dot(-K, (x - wr))[0]
    return np.array([
        x[1],
        (1/D)*(-(pend_mass**2)*(arm_length**2)*g*Cx*Sx+pend_mass*(arm_length**2)*(pend_mass*arm_length*(x[3]**2)*Sx-d*x[1]))+pend_mass*(arm_length**2)*(1/D)*u,
        x[3],
        (1/D)*((pend_mass+cart_mass)*pend_mass*g*arm_length*Sx-pend_mass*arm_length*Cx*(pend_mass*arm_length*(x[3]**2)*Sx-d*x[1]))-pend_mass*arm_length*Cx*(1/D)*u
    ])

t = np.arange(0, 10, 0.01)
x0 = np.array([-1, 0, np.pi + .1, 0])

sol = odeint(pendcart, x0, t)








fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(-5, 5)
ax.set_ylim(0, 5)

# Initial draw of elements, to be updated in animation
left_wheel = ax.scatter(-.5, .25, s=1000, c="k")
right_wheel = ax.scatter(.5, .25, s=1000, c="k")
ax.plot([-5, 5], [0, 0], "k-")  # x-axis
cart_body = ax.plot([-.8, .8], [.65, .65], "b-", linewidth=30)[0]  # cart body
pendulum_arm, = ax.plot([0, 0], [0.7, .7], "k-")  # pendulum arm
pendulum_bob = ax.scatter(0, .7, s=1000, c="r")  # pendulum

def update(frame):
    x = sol[frame][0]
    theta = sol[frame][2]
    
    # Update positions based on current data
    left_wheel.set_offsets([-.5 + x, .25])
    right_wheel.set_offsets([.5 + x, .25])
    cart_body.set_data([-.8 + x, .8 + x], [.65, .65])
    pendulum_arm.set_data([0 + x, -3 * np.sin(-theta) + x], [0.7, -3 * np.cos(-theta) + .7])
    pendulum_bob.set_offsets([-3 * np.sin(-theta) + x, -3 * np.cos(-theta) + .7])
    
    return left_wheel, right_wheel, cart_body, pendulum_arm, pendulum_bob

# Create animation
ani = FuncAnimation(fig, update, frames=len(sol), blit=True, interval=50, repeat=False)

plt.show()




