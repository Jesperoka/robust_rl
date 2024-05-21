from panda_py import Panda, Desk

SHOP_FLOOR_IP = "10.0.0.2"

with open('sens.txt', 'r') as file:
    username = file.readline().strip()
    password = file.readline().strip()

if __name__ == "__main__":
    Desk(SHOP_FLOOR_IP, username, password).activate_fci()
    panda = Panda(SHOP_FLOOR_IP)
    robot_state = panda.get_state()
    print(robot_state.q)
