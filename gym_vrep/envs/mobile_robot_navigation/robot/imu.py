import numpy as np
from gym_vrep.envs.vrep import vrep
from gym_vrep.envs.mobile_robot_navigation.robot.base import Base


class Imu(Base):
    def __init__(self, client, dt, objects_names, stream_names):
        super(Imu, self).__init__(client, dt, objects_names, stream_names)

    def get_accelerometer_values(self):
        _, packed_vec = vrep.simxReadStringStream(
            self._client, self._stream_names['accelerometer'],
            vrep.simx_opmode_buffer)

        data = vrep.simxUnpackFloats(packed_vec)

        return np.asarray(data)

    def get_gyroscope_values(self):
        _, packed_vec = vrep.simxReadStringStream(
            self._client, self._stream_names['gyroscope'],
            vrep.simx_opmode_buffer)

        data = vrep.simxUnpackFloats(packed_vec)

        return np.asarray(data)
