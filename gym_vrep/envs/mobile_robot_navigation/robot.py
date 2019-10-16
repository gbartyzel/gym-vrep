from typing import Dict
from typing import NoReturn

import numpy as np

from gym_vrep.envs.vrep import vrep


class Robot(object):
    """A class that create interface between mobile robot model in simulation
    and remote API. Main functionality of this class is:
    * reading values from sensors (like proximity sensor, accelerometer etc.)
    * setting motors velocities
    * getting mobile robot pose from simulation engine
    """
    wheel_diameter = 0.06
    body_width = 0.156
    nb_proximity_sensor = 5
    velocity_bound = np.array([0.0, 10.0])
    proximity_sensor_bound = np.array([0.02, 2.0])

    def __init__(self, client: int, dt: float, objects_names: Dict[str, str],
                 stream_names: Dict[str, str]):
        """A robot class constructor.

        Args:
            client: A V-REP client id.
            dt: A delta time of the environment.
            objects_names: A dictionary of objects names that robot contains.
            stream_names: A dictionary of streams names that robot contains.
        """
        self._object_names = objects_names

        self._stream_names = stream_names
        self._dt = dt
        self._object_handlers = dict()

        self._client = client

        self._proximity_sensor_values = 2 * np.ones(5)
        self._encoder_ticks = np.zeros(2)
        self._gyroscope_values = np.zeros(3)
        self._accelerometer_values = np.zeros(3)

    def reset(self) -> NoReturn:
        """A method that reset robot model.  It is zeroing all class variables
        and also reinitialize handlers for newly loaded robot model into the
        environment.

        """
        self._object_handlers = dict()
        self._proximity_sensor_values = 2 * np.ones(5)
        self._encoder_ticks = np.zeros(2)
        self._gyroscope_values = np.zeros(3)
        self._accelerometer_values = np.zeros(3)

        for key, name in self._object_names.items():
            res, temp = vrep.simxGetObjectHandle(self._client, name,
                                                 vrep.simx_opmode_oneshot_wait)
            self._object_handlers[key] = temp

    def set_motor_velocities(self, velocities: np.ndarray) -> NoReturn:
        """A method that sets motors velocities and clips if given values exceed
        boundaries.

        Args:
            velocities: Target motors velocities in rad/s.

        """
        if np.shape(velocities)[0] != 2:
            raise ValueError(
                'Dimension of input motors velocities is incorrect!')

        velocities = np.clip(velocities, self.velocity_bound[0],
                             self.velocity_bound[1])

        vrep.simxSetJointTargetVelocity(self._client,
                                        self._object_handlers['left_motor'],
                                        velocities[0], vrep.simx_opmode_oneshot)

        vrep.simxSetJointTargetVelocity(self._client,
                                        self._object_handlers['right_motor'],
                                        velocities[1], vrep.simx_opmode_oneshot)

    def get_encoders_rotations(self) -> np.ndarray:
        """Reads encoders ticks from robot.

        Returns:
            encoder_ticks: Current values of encoders ticks.
        """
        res, packed_vec = vrep.simxReadStringStream(
            self._client, self._stream_names['encoders'],
            vrep.simx_opmode_oneshot)

        if res == vrep.simx_return_ok:
            self._encoder_ticks = vrep.simxUnpackFloats(packed_vec)
            self._encoder_ticks = np.round(self._encoder_ticks, 3)
        return self._encoder_ticks

    def get_image(self) -> np.ndarray:
        """Reads image from camera mounted to robot.

        Returns:
            img: Image received from robot
        """
        res, resolution, image = vrep.simxGetVisionSensorImage(
            self._client, self._object_handlers['camera'], False,
            vrep.simx_opmode_oneshot)
        img = np.array(image, dtype=np.uint8)
        img.resize([resolution[1], resolution[0], 3])
        return img

    def get_proximity_values(self) -> np.ndarray:
        """Reads proximity sensors values from robot.

        Returns:
            proximity_sensor_values: Array of proximity sensor values.
        """
        res, packed_vec = vrep.simxReadStringStream(
            self._client, self._stream_names['proximity_sensor'],
            vrep.simx_opmode_oneshot)

        if res == vrep.simx_return_ok:
            self._proximity_sensor_values = vrep.simxUnpackFloats(packed_vec)
            self._proximity_sensor_values = np.round(
                self._proximity_sensor_values, 3)
        return self._proximity_sensor_values

    def get_accelerometer_values(self) -> np.ndarray:
        """Reads accelerometer values from robot.

        Returns:
            accelerometer_values: Array of values received from accelerometer.
        """
        res, packed_vec = vrep.simxReadStringStream(
            self._client, self._stream_names['accelerometer'],
            vrep.simx_opmode_oneshot)

        if res == vrep.simx_return_ok:
            self._accelerometer_values = vrep.simxUnpackFloats(packed_vec)
            self._accelerometer_values = np.asarray(self._accelerometer_values)

        return self._accelerometer_values

    def get_gyroscope_values(self) -> np.ndarray:
        """Reads gyroscope values from robot.

        Returns:
            gyroscope_values: Array of values received from gyroscope.
        """
        res, packed_vec = vrep.simxReadStringStream(
            self._client, self._stream_names['gyroscope'],
            vrep.simx_opmode_oneshot)

        if res == vrep.simx_return_ok:
            self._gyroscope_values = vrep.simxUnpackFloats(packed_vec)
            self._gyroscope_values = np.asarray(self._gyroscope_values)
        return self._gyroscope_values

    def get_velocities(self) -> np.ndarray:
        """Computes current linear and angular velocity of robot. Velocities are
        computed using mobile robot kinematic model.

        Returns:
            velocities: Array of linear and angular mobile robot velocities.
        """
        radius = self.wheel_diameter / 2.0
        wheels_velocities = self.get_encoders_rotations() / self._dt

        linear_velocity = np.sum(wheels_velocities) * radius / 2.0
        angular_velocity = radius * np.diff(wheels_velocities) / self.body_width
        velocities = np.array([linear_velocity, angular_velocity],
                              dtype=np.float)

        return np.round(velocities, 3)

    def get_pose(self) -> np.ndarray:
        """Reads current pose of the robot.

        Returns:
            pose: Pose array containing x,y and yaw.
        """
        _, pos = vrep.simxGetObjectPosition(
            self._client, self._object_handlers['robot'], -1,
            vrep.simx_opmode_oneshot)

        _, rot = vrep.simxGetObjectOrientation(
            self._client, self._object_handlers['robot'], -1,
            vrep.simx_opmode_oneshot)

        pose = np.round(pos[0:2] + [rot[2]], 3)
        return pose
