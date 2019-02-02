import numpy as np

from typing import List, Dict, NoReturn
from gym_vrep.envs.vrep import vrep


class Robot(object):
    wheel_diameter = 0.06
    body_width = 0.156
    nb_proximity_sensor = 5
    velocity_bound = np.array([0.0, 10.0])
    proximity_sensor_bound = np.array([0.02, 2.0])

    def __init__(self, client: int, dt: float, objects_names: Dict[str, str],
                 stream_names: Dict[str, str]):
        self._object_names = objects_names

        self._stream_names = stream_names
        self._dt = dt
        self._object_handlers = dict()

        self._client = client

        self._proximity_reading = 2 * np.ones(5)
        self._encoder_reading = np.zeros(2)
        self._gyroscope_reading = np.zeros(3)
        self._accelerometer_reading = np.zeros(3)

    def reset(self) -> NoReturn:
        self._object_handlers = dict()
        self._proximity_reading = 2 * np.ones(5)
        self._encoder_reading = np.zeros(2)
        self._gyroscope_reading = np.zeros(3)
        self._accelerometer_reading = np.zeros(3)

        for key, name in self._object_names.items():
            res, temp = vrep.simxGetObjectHandle(self._client, name, vrep.simx_opmode_oneshot_wait)
            self._object_handlers[key] = temp

        for key, stream in self._stream_names.items():
            vrep.simxGetStringSignal(self._client, stream, vrep.simx_opmode_streaming)

        if 'camera' in self._object_handlers:
            vrep.simxGetVisionSensorImage(
                self._client, self._object_handlers['camera'], False, vrep.simx_opmode_streaming)

        vrep.simxGetObjectPosition(
            self._client, self._object_handlers['robot'], -1, vrep.simx_opmode_streaming)

        vrep.simxGetObjectOrientation(
            self._client, self._object_handlers['robot'], -1, vrep.simx_opmode_streaming)

    def set_motor_velocities(self, velocities: np.ndarray) -> NoReturn:
        velocities = np.clip(velocities, self.velocity_bound[0], self.velocity_bound[1])

        vrep.simxSetJointTargetVelocity(self._client, self._object_handlers['left_motor'],
                                        velocities[0], vrep.simx_opmode_oneshot)

        vrep.simxSetJointTargetVelocity(self._client, self._object_handlers['right_motor'],
                                        velocities[1], vrep.simx_opmode_oneshot)

    def get_encoders_rotations(self) -> np.ndarray:
        res, packed_vec = vrep.simxGetStringSignal(
            self._client, self._stream_names['encoders'],
            vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            self._encoder_reading = np.asanyarray(vrep.simxUnpackFloats(packed_vec))

        return np.round(self._encoder_reading, 3)

    def get_image(self) -> np.ndarray:
        res, resolution, image = vrep.simxGetVisionSensorImage(
            self._client, self._object_handlers['camera'], False, vrep.simx_opmode_buffer)
        img = np.array(image, dtype=np.uint8)
        img.resize([resolution[1], resolution[0], 3])
        return img

    def get_proximity_values(self) -> np.ndarray:
        res, packed_vec = vrep.simxGetStringSignal(
            self._client, self._stream_names['proximity_sensor'],
            vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            self._proximity_reading = vrep.simxUnpackFloats(packed_vec)

        return np.round(self._proximity_reading, 3)

    def get_accelerometer_values(self) -> np.ndarray:
        res, packed_vec = vrep.simxGetStringSignal(
            self._client, self._stream_names['accelerometer'], vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            self._accelerometer_reading = vrep.simxUnpackFloats(packed_vec)

        return np.asarray(self._accelerometer_reading)

    def get_gyroscope_values(self) -> np.ndarray:
        res, packed_vec = vrep.simxGetStringSignal(
            self._client, self._stream_names['gyroscope'], vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            self._gyroscope_reading = vrep.simxUnpackFloats(packed_vec)
        return np.asarray(self._gyroscope_reading)

    def get_velocities(self) -> np.ndarray:
        radius = self.wheel_diameter / 2.0
        wheels_velocities = self._encoder_reading / self._dt

        linear_velocity = np.sum(wheels_velocities) * radius / 2.0
        angular_velocity = radius * np.diff(wheels_velocities) / self.body_width

        return np.round(np.array([linear_velocity, angular_velocity]), 3)

    def get_position(self) -> np.ndarray:
        _, pos = vrep.simxGetObjectPosition(
            self._client, self._object_handlers['robot'], -1, vrep.simx_opmode_buffer)

        _, rot = vrep.simxGetObjectOrientation(
            self._client, self._object_handlers['robot'], -1, vrep.simx_opmode_buffer)

        return np.round(pos[0:2] + [rot[2]], 3)
