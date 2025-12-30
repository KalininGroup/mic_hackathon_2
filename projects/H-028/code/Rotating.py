import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R

def rotate_vector_alpha(vector, angle_degrees):
    """
    Rotates a 3D vector around the x-axis using SciPy.

    :param vector: A 3D vector as a list or numpy array [x, y, z].
    :param angle_degrees: The angle of rotation in degrees.
    :return: The rotated vector as a numpy array.
    """
    # Convert angle to radians for internal use, though R.from_euler can use degrees
    # Define the rotation: 'X' axis, angle in degrees
    rotation = R.from_euler('x', angle_degrees, degrees=True)
    
    # Apply the rotation to the vector
    rotated_vector = rotation.apply(vector)
    
    return rotated_vector

def rotate_vector_beta(vector, angle_degrees):
    """
    Rotates a 3D vector around the y-axis using SciPy.

    Args:
        vector (list or np.array): The original 3D vector [x, y, z].
        angle_degrees (float): The rotation angle in degrees.

    Returns:
        np.array: The rotated vector.
    """
    rotation_vector = np.radians(angle_degrees) * np.array([0, 1, 0])
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(vector)
    return rotated_vec

def calculate_angle(v1, v2):
    """
    Calculates the angle between two vectors in radians.
    
    Args:
        v1 (list or np.array): The first vector.
        v2 (list or np.array): The second vector.
        
    Returns:
        float: The angle in radians (range [0, pi]).
    """
    # Convert lists to numpy arrays if necessary
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Compute the dot product
    dot_product = np.dot(v1, v2)
    
    # Compute the magnitudes (norms)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Calculate the cosine of the angle
    cosine_theta = dot_product / (norm_v1 * norm_v2)
    
    # Handle potential floating-point errors (clipping value to [-1, 1])
    # to avoid issues with np.arccos for values slightly outside this range
    cosine_theta = np.clip(cosine_theta, -1.0, 1.0)
    
    # Calculate the angle in radians using arccos
    angle_radians = np.arccos(cosine_theta)
    
    return angle_radians


def calculate_azimuth(normal):
    #Project norm into into xy plane
    project_normal = normal
    project_normal = np.array([normal[0], normal[1], 0])  #Just make the z coordinate zero, project it straight down
    #If you started with [001], this is going to cause some problems, so catch that error:
    if LA.norm(project_normal) == 0:
        phi_r = 0
        print("If")
        print(phi_r)
    else:
        print("Else")
        phi_r = calculate_angle(project_normal, np.array([1,0,0]))
    return phi_r
    