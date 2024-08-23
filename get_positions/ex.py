from rria_api.robot_enum import RobotEnum
from rria_api.robot_object import RobotObject


robot = RobotObject('192.168.2.10', RobotEnum.GEN3_LITE)
robot.connect_robot()

robot.move_joints(20.665, 350.727, 97.484, 267.707, 287.288, 115.2) # 'webcam_view'

