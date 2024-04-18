import math

# this assumes that the format of the orientation quarternions is [x, y, z, w]

def quart_norm(quarternion_orientation: list):

    x, y, z, w = quarternion_orientation
    return math.sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2)

def inverse_quarternion(quarternion_orientation: list):

    # assumes that [x, y, z, w]
    x, y, z, w = quarternion_orientation
    norm_q_sqrd = quart_norm(quarternion_orientation) ** 2

    return (-x/norm_q_sqrd, -y/norm_q_sqrd, -z/norm_q_sqrd, w/norm_q_sqrd)

def multiply_quarternions(quart_1: list, quart_2: list):

    x1, y1, z1, w1 = quart_1
    x2, y2, z2, w2 = quart_2

    return (
    w1*x2 + x1*w2 + y1*z2 - z1*y2,
    w1*y2 - x1*z2 + y1*w2 + z1*x2,
    w1*z2 + x1*y2 - y1*x2 + z1*w2,
    w1*w2 - x1*x2 - y1*y2 - z1*z2
    )

def adjust_orientation_origin(inverse_reference_quart: list, quart: list):
 
    return multiply_quarternions(inverse_reference_quart, quart)

def adjust_position_origin(ref_pose, pose):

    return [pose[i] - ref_pose[i] for i in range(3)]