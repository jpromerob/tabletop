from five_bar import FiveBar
import dynamixel_sdk as dx
import math
import time

DEG_TO_RAD = math.pi / 180
RAD_TO_DEG = 180 / math.pi

# Control table address
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_MIN_POSITION_LIMIT = 52
ADDR_MAX_POSITION_LIMIT = 48
ADDR_OPERATING_MODE = 11
ADDR_PROFILE_VELOCITY = 112
ADDR_PRESENT_POSITION = 132
left_trim = 0
right_trim = 210

# Protocol version
PROTOCOL_VERSION = 2

r1 = 23
r2 = 23
r3 = 23
r4 = 23
r5 = 10

left_id = 1
right_id = 2


center = 4095 // 2


'''
This function initializes the motors
'''
def initialize_dynamixel(torque_on):

    time.sleep(1)

    global port
    global handler
    port = dx.PortHandler("/dev/ttyACM0")

    # initialize_dynamixel PacketHandler Structs
    handler = dx.PacketHandler(2)

    # Open port
    port.openPort()

    # Set port baudrate
    port.setBaudRate(1000000)

    handler.reboot(port, right_id)
    handler.reboot(port, left_id)
    time.sleep(1)

    handler.write4ByteTxRx(port, right_id, ADDR_OPERATING_MODE, 4)
    handler.write4ByteTxRx(port, left_id, ADDR_OPERATING_MODE, 4)

    if torque_on:
        flag = 1
    else:
        flag = 0
    handler.write1ByteTxRx(port, left_id, ADDR_TORQUE_ENABLE, flag)
    handler.write1ByteTxRx(port, right_id, ADDR_TORQUE_ENABLE, flag)
    handler.write4ByteTxRx(port, right_id, ADDR_OPERATING_MODE, 4)
    handler.write4ByteTxRx(port, left_id, ADDR_OPERATING_MODE, 4)
    handler.write1ByteTxRx(port, left_id, ADDR_TORQUE_ENABLE, flag)
    handler.write1ByteTxRx(port, right_id, ADDR_TORQUE_ENABLE, flag)

'''
This function converts tip pose into motor angles
'''
def get_angles_from_xy(x, y):

    
    x += r5 / 2
    linkage = FiveBar(r1, r2, r3, r4, r5)
    linkage.inverse(x, y)
    if math.isnan(linkage.get_a11()) or math.isnan(linkage.get_a11()):
        return False
    left_angle  = -(math.pi - linkage.get_a11() - math.pi / 2)
    right_angle = -(math.pi / 2 - linkage.get_a42())


    return left_angle, right_angle

'''
This function converts angles in degrees to dynamixel's format 
'''
def degree_to_dx(angle):
    # minimum = -1_048_575
    maximum = 4095
    can_move = 2 * math.pi
    degress_per_step = can_move / maximum
    return int(angle/degress_per_step)

'''
This function requests motors to move to specific angular positions
The request is given as follows:
 - from a known current tip pose, 
 - move to a desired tip pose 
 - at a particular speed
'''
def move_to_from(new_x, new_y, cur_x, cur_y, velocity_rpm = 3):
    # print("in", x, y)
    # new_x = new_x - 0.5 # to correct for weird offset
    global port
    global handler

    new_left_angle, new_right_angle = get_angles_from_xy(new_x, new_y)
    cur_left_angle, cur_right_angle = get_angles_from_xy(cur_x, cur_y)

    left_discrete = center + degree_to_dx(new_left_angle) + left_trim
    right_discrete = center + degree_to_dx(new_right_angle) + right_trim


    velocity_discrete = int(velocity_rpm / 0.299)


    if 0 < left_discrete < 4095 and 0 < right_discrete < 4095:
        handler.write4ByteTxRx(port, right_id, ADDR_PROFILE_VELOCITY, velocity_discrete)
        handler.write4ByteTxRx(port, left_id, ADDR_PROFILE_VELOCITY, velocity_discrete)

        handler.write4ByteTxRx(port, right_id, ADDR_GOAL_POSITION, right_discrete)
        handler.write4ByteTxRx(port, left_id, ADDR_GOAL_POSITION, left_discrete)
        return True
    return False


'''
This function inquires the motor's angular position and returns the
corresponding tip pose
'''
def read_position():
    right_raw = handler.read4ByteTxRx(port, right_id, ADDR_PRESENT_POSITION)
    left_raw = handler.read4ByteTxRx(port, left_id, ADDR_PRESENT_POSITION)

    right_raw = right_raw[0]
    left_raw = left_raw[0]

    right_angle = (right_raw - right_trim - center) / 4095 * 2 * math.pi
    left_angle = (left_raw - left_trim - center) / 4095 * 2 * math.pi

    right_adjusted = right_angle + math.pi / 2
    left_adjusted = left_angle + math.pi / 2


    linkage = FiveBar(r1, r2, r3, r4, r5)
    linkage.forward(left_angle,right_angle)
    (x, y) = linkage.calculate_position(left_adjusted, right_adjusted)
    x -= r5 / 2 # center
    return (x, y)
