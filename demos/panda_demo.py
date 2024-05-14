import panda_py
import numpy as np
from os.path import abspath, dirname, join

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP

filepath = abspath(join(dirname(__file__), "../", "sens.txt"))

with open(filepath, 'r') as file:
    username = file.readline().strip()
    password = file.readline().strip()

if __name__ == "__main__":

    print("Connecting to Desk")

    # Connect to the robot
    desk = panda_py.Desk(SHOP_FLOOR_IP, username, password)

    while not desk.has_control():
        key = input("Another user has control\nPress any key (except q) when control has been relinquished to unlock FCI.\nOr press q to exit: ")
        if key == "q": exit(0)
        desk.take_control()

    print("Unlocking joints if locked")
    desk.unlock()       # don't need if already unlocked

    print("Activating FCI")
    desk.activate_fci() # don't need if already activated

    print("Conneting to Panda")
    panda = panda_py.Panda(SHOP_FLOOR_IP)

    # Define some joint positions pretty close to panda_py.constants.JOINT_POSITION_START
    pos0 = np.array([-0.1 , -0.79,  0.0 , -2.36,  0.0 , 1.57,  0.79])
    pos1 = np.array([ 0.0 , -0.79,  0.0 , -2.36,  0.0 , 1.57,  0.79])
    pos2 = np.array([ 0.0 , -0.89,  0.0 , -2.36,  0.0 , 1.57,  0.79])
    pos3 = np.array([ 0.0 , -0.69,  0.1 , -2.36,  0.0 , 1.57,  0.79])
    pos4 = np.array([ 0.0 , -0.69,  0.1 , -2.26,  0.0 , 1.57,  0.79])
    pos5 = np.array([ 0.0 , -0.69,  0.1 , -2.26,  0.1 , 1.57,  0.79])
    pos6 = np.array([ 0.0 , -0.69,  0.1 , -2.26,  0.1 , 1.67,  0.79])
    pos7 = np.array([ 0.0 , -0.69,  0.1 , -2.26,  0.1 , 1.67,  0.89])

    input("Press any key to move arm")

    # Move to panda_py.constants.JOINT_POSITION_START
    panda.move_to_start()

    # Move to the arm one joint at a time passing only a single 7x1 np.ndarray
    panda.move_to_joint_position(pos0)
    panda.move_to_joint_position(pos1)
    panda.move_to_joint_position(pos2)
    panda.move_to_joint_position(pos3)
    panda.move_to_joint_position(pos4)
    panda.move_to_joint_position(pos5)
    panda.move_to_joint_position(pos6)
    panda.move_to_joint_position(pos7)

    panda.move_to_start()

    # Or pass a list of 7x1 np.ndarrays as waypoints for a motion
    waypoints = [pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7]
    panda.move_to_joint_position(waypoints)

    # Move back to the neutral position
    panda.move_to_start()

    # Relinquish control
    desk.deactivate_fci()
    desk.release_control()


