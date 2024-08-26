from rria_api.robot_enum import RobotEnum
from rria_api.robot_object import RobotObject

doc_name = 'doc_positions.txt'

robot = RobotObject('192.168.2.10', RobotEnum.GEN3_LITE)
robot.connect_robot()

conditions = True

try:
    doc_file_ = open(doc_name, 'w')
    doc_file_.close()

except FileNotFoundError:
    with open(doc_name, 'a+') as doc_file:
        doc_file.write('Position\tType\tJoints\n')

while conditions != 0:
    choice = input("Digite o nome da posição:\n")

    joint_position = robot.get_joints()
    cartesian_position = robot.get_cartesian()
    print(joint_position)
    print(cartesian_position)

    with open(doc_name, 'a+') as doc_file:

        doc_file.write(f'{choice}\tjoints\t{joint_position}\n')
        doc_file.write(f"{choice}\tcartesian\t{cartesian_position}\n")
