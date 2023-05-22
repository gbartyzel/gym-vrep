from pyrep.robots.arms.arm import Arm


class UR5e(Arm):
    def __init__(self, count: int = 0):
        super().__init__(count, "UR5e", 6)
