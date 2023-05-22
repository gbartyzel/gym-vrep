from pyrep.robots.end_effectors.gripper import Gripper


class RobotiqHande(Gripper):
    def __init__(self, joint_names: list, count: int = 0):
        super().__init__(count, "Robotiq_Hande", joint_names)
        self.initial_configuration = self.get_configuration_tree()
