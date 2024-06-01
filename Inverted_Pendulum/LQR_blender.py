import bpy
import numpy as np
from mathutils import Euler, Vector
from scipy.integrate import odeint
import control


m = 0.3
M = 1.0
L = 3
g = 9.8
delta = 0.1

def dynamics(y, m, M, L, g, delta, u):
    """
    Compute the state derivatives.
    :param y: state vector
    :param m: pendulum mass
    :param M: cart mass
    :param L: pendulum length
    :param g: gravitational acceleration
    :param delta: friction damping
    :param u: control input
    :return: dy: state derivative
    """
    theta = y[2]
    theta_dot = y[3]
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    D = m * (L ** 2) * (M + m * (1 - cos_theta ** 2))    
    dy = np.array([
        y[1],
        (1 / D) * (
            -(m ** 2) * (L ** 2) * g * cos_theta * sin_theta 
            + m * (L ** 2) * (m * L * (theta_dot ** 2) * sin_theta - delta * y[1])
        ) + m * (L ** 2) * (1 / D) * u,
        y[3],
        (1 / D) * (
            (m + M) * m * g * L * sin_theta 
            - m * L * cos_theta * (m * L * (theta_dot ** 2) * sin_theta 
            - delta * y[1])
        ) - m * L * cos_theta * (1 / D) * u
    ])
    return dy

def get_a_matrix(m, M, L, g, delta):
    """
    :return A: state matrix
    """
    A = np.array([
        [0, 1, 0, 0],
        [0, -delta / M, m * g / M, 0],
        [0, 0, 0, 1],
        [0, -delta / (M * L), - (M + m) * g / (M * L), 0]
    ])
    return A

def get_b_matrix(m, M, L, g, delta):
    """
    :return B: input matrix
    """
    B = np.array([
        [0],
        [1 / M],
        [0],
        [1 / (M * L)]
    ])
    return B



def dynamics_with_control(y, t, m, M, L, g, delta, K):
    """
    Compute the state derivative with control input.
    :param K: state feedback gain matrix
    :return dy: state derivative
    """
    y_goal = np.array([0, 0, np.pi * 1, 0])
    u = np.dot(-K, y - y_goal )[0]
    return dynamics(y, m, M, L, g, delta, u)


def degrees_to_radians(y):
    return (y % (2 * np.pi)) / np.pi 

def plot_results(t, y):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(t, y[:, 0], label='Cart Position')
    axs[0].set_ylabel('Position (m)')
    axs[0].set_xlabel('Time (s)')
    axs[0].legend()
    axs[0].axhline(y=0, color='r', linestyle='--')

    axs[1].plot(t, degrees_to_radians(y[:, 2]), label='Pendulum Angle')
    axs[1].set_ylabel('Angle (rad)')
    axs[1].set_xlabel('Time (s)')
    axs[1].legend()
    axs[1].axhline(y=1, color='r', linestyle='--')
    plt.tight_layout()
    plt.show()


tf = 10
dt = float(1 / 30)
t = np.arange(0, tf, dt)


A = get_a_matrix(m, M, L, g, delta)
B = get_b_matrix(m, M, L, g, delta)
Q = np.diag([1,1,1,1])
R = np.array([[0.001]])

K, S, E = control.lqr(A, B, Q, R)

y0 = np.array([-3, 0, np.pi * (1 + 0.1), 0])
y = odeint(dynamics_with_control, y0, t, args=(m, M, L, g, delta, K))




##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################


# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Create cart object
bpy.ops.mesh.primitive_cube_add(size=1, location=(y0[0], 0, -0.25))
cart = bpy.context.object
cart.scale.x = 2
cart.scale.y = 1
cart.scale.z = 0.5
cart.location.x = y0[0]

# Create pendulum rod determined by two points


rod_start = Vector((y0[0], 0, 0))
rod_end = rod_start + Vector((L * np.sin(y0[2]), 0, -L * np.cos(y0[2])))

rod_location = rod_start + (rod_end - rod_start) / 2
rod_rotation_euler = Euler((0, -y0[2], 0))

bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1, location=rod_location)
rod = bpy.context.object

rod.rotation_euler = rod_rotation_euler
rod.scale.z = L

# Create the bob
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=rod_end, scale=(1, 1, 1))
bob = bpy.context.object

bpy.context.view_layer.update()


# Create keyframes for the animation
for frame, state in enumerate(y):
    time = frame * dt
    cart.location.x = state[0]
    
    # Calculate the end point of the rod
    rod_start = Vector((state[0], 0, 0))
    rod_end = rod_start + Vector((L * np.sin(state[2]), 0, -L * np.cos(state[2])))
    
    # Update rod position and rotation
    rod.location = rod_start + (rod_end - rod_start) / 2
    rod.rotation_euler = Euler((0, -state[2], 0))
    
    # Update bob position
    bob.location = rod_end
    
    # Insert keyframes
    cart.keyframe_insert(data_path="location", frame=frame)
    rod.keyframe_insert(data_path="location", frame=frame)
    rod.keyframe_insert(data_path="rotation_euler", frame=frame)
    rod.keyframe_insert(data_path="scale", frame=frame)
    bob.keyframe_insert(data_path="location", frame=frame)

# Set animation end frame
bpy.context.scene.frame_end = len(y)

# Set camera
bpy.ops.object.camera_add(location=(0, -100, 5))
camera = bpy.context.object
camera.rotation_euler = (np.pi / 3, 0, 0)
bpy.context.scene.camera = camera

# Set light
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
light = bpy.context.object
light.rotation_euler = (0, 0, 0)