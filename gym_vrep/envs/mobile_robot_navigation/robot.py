import numpy as np

from gym_vrep.envs.vrep import vrep


class Robot(object):
    wheel_diameter = 0.06
    body_width = 0.156
    velocity_bound = np.array([0.0, 10.0])
    prox_sensor_bound = np.array([0.02, 2.0])

    def __init__(self, client, dt, objects_names, stream_names):
        self._object_names = objects_names

        self._stream_names = stream_names
        self._dt = dt
        self._objects_hanlders = dict()
        self._delta_phi = np.zeros(2)

        self._client = client

        self._proximity_reading = 2 * np.ones(5)
        self._encoder_reading = np.zeros(2)
        self._gyroscope_reading = np.zeros(3)
        self._accelerometer_reading = np.zeros(3)

    def reset(self):
        self._objects_hanlders = dict()
        self._delta_phi = np.zeros(2)

        for key, name in self._object_names.items():
            res, temp = vrep.simxGetObjectHandle(self._client, name,
                                                 vrep.simx_opmode_oneshot_wait)
            self._objects_hanlders[key] = temp

        for key, stream in self._stream_names.items():
            vrep.simxGetStringSignal(self._client, stream,
                                      vrep.simx_opmode_streaming)

        vrep.simxGetObjectPosition(self._client,
                                   self._objects_hanlders['robot'], -1,
                                   vrep.simx_opmode_streaming)

        vrep.simxGetObjectOrientation(self._client,
                                      self._objects_hanlders['robot'], -1,
                                      vrep.simx_opmode_streaming)

    def set_motor_velocities(self, velocities):
        if isinstance(velocities, list):
            velocities = np.asarray(velocities)

        velocities[velocities < self.velocity_bound[0]] = \
            self.velocity_bound[0]
        velocities[velocities > self.velocity_bound[1]] = \
            self.velocity_bound[1]

        vrep.simxSetJointTargetVelocity(
            self._client, self._objects_hanlders['left_motor'],
            velocities[0], vrep.simx_opmode_oneshot)

        vrep.simxSetJointTargetVelocity(
            self._client, self._objects_hanlders['right_motor'],
            velocities[1], vrep.simx_opmode_oneshot)

    def get_encoders_rotations(self):
        res, packed_vec = vrep.simxGetStringSignal(
            self._client, self._stream_names['encoders'],
            vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            self._encoder_reading = vrep.simxUnpackFloats(packed_vec)

        return np.round(self._encoder_reading, 3)

    def get_proximity_values(self):
        res, packed_vec = vrep.simxGetStringSignal(
            self._client, self._stream_names['proximity_sensor'],
            vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            self._proximity_reading = vrep.simxUnpackFloats(packed_vec)

        return np.round(self._proximity_reading, 3)

    def get_accelerometer_values(self):
        res, packed_vec = vrep.simxGetStringSignal(
            self._client, self._stream_names['accelerometer'],
            vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            self._accelerometer_reading = vrep.simxUnpackFloats(packed_vec)

        return np.asarray(self._accelerometer_reading)

    def get_gyroscope_values(self):
        res, packed_vec = vrep.simxGetStringSignal(
            self._client, self._stream_names['gyroscope'],
            vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            self._gyroscope_reading = vrep.simxUnpackFloats(packed_vec)
        return np.asarray(self._gyroscope_reading)

    def get_velocities(self):
        wheels_velocities = self._delta_phi / self._dt
        linear_velocity = (np.sum(
            wheels_velocities) * self.wheel_diameter / 2.0) / 2.0

        velocities = np.append(wheels_velocities, linear_velocity)

        return np.round(velocities, 3)

    def get_position(self):
        _, pos = vrep.simxGetObjectPosition(self._client,
                                            self._objects_hanlders['robot'],
                                            -1,
                                            vrep.simx_opmode_buffer)

        _, rot = vrep.simxGetObjectOrientation(self._client,
                                               self._objects_hanlders['robot'],
                                               -1,
                                               vrep.simx_opmode_buffer)

        return np.round(pos[0:2] + [rot[2]], 3)
