from typing import NoReturn

import numpy as np

import gym_vrep.envs.vrep.vrep_lib as vlib


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
    velocity_limit = np.array([0.0, 15.0])
    ultrasonic_sensor_bound = np.array([0.02, 2.0])

    def __init__(self, client: int, dt: float, enable_vision: bool):
        """A robot class constructor.

        Args:
            client: A V-REP client id.
            dt: A delta time of the environment.
            enable_vision: Flag that set image acquisition from robot camera
        """
        self._dt = dt
        self._enable_vision = enable_vision

        self._robot_handle = -1
        self._camera_handle = -1
        self._ultrasonic_handles = [-1] * self.nb_proximity_sensor
        self._motor_handles = 2 * [-1]

        self._encoder_signals = ['leftEncoderSignal', 'rightEncoderSignal']
        self._imu_signals = ['accelerometerSignal', 'gyroscopeSignal']

        self._client = client

        self._ultrasonic_values = 2 * np.ones(5)
        self._encoder_ticks = np.zeros(2)
        self._gyroscope_values = np.zeros(3)
        self._accelerometer_values = np.zeros(3)

        self._initialize()

    def _initialize(self) -> NoReturn:
        """A method that initialize robot model.  It is zeroing all class
        variables and also reinitialize handlers for newly loaded robot model
        into the environment.

        """
        self._initialize_robot_handle()
        self._initialize_motors()
        self._initialize_ultrasonic()
        self._initialize_encoders_stream()
        self._initialize_imu_stream()
        if self._enable_vision:
            self._initialize_camera()

    def _initialize_robot_handle(self):
        """Initialize handle of robot body and also position and orientation
        stream.
        """
        self._robot_handle = vlib.get_object_handle(self._client, 'smartBot')
        vlib.get_object_position(self._client, self._robot_handle, stream=True)
        vlib.get_object_orientation(self._client, self._robot_handle,
                                    stream=True)

    def _initialize_motors(self):
        """Initialize handles of the motors.
        """
        self._motor_handles[0] = vlib.get_object_handle(self._client,
                                                        'smartBot_leftMotor')
        self._motor_handles[1] = vlib.get_object_handle(self._client,
                                                        'smartBot_rightMotor')

    def _initialize_ultrasonic(self):
        """Initialize handles of the ultrasonic sensors.
        """
        self._ultrasonic_values = 2 * np.ones(5)
        for i in range(self.nb_proximity_sensor):
            name = 'smartBot_ultrasonicSensor{}'.format(i + 1)
            object_handle = vlib.get_object_handle(self._client, name)
            vlib.read_proximity_sensor(self._client, object_handle, True)
            self._ultrasonic_handles[i] = object_handle

    def _initialize_camera(self):
        """Initialize handle of the camera.
        """
        object_handle = vlib.get_object_handle(self._client, 'smartBot_camera')
        vlib.get_image(self._client, object_handle, True)
        self._camera_handle = object_handle

    def _initialize_encoders_stream(self):
        """Initialize encoders streams.
        """
        self._encoder_ticks = np.zeros(2)
        for name in self._encoder_signals:
            vlib.read_float_stream(self._client, name, True)

    def _initialize_imu_stream(self):
        """Initialize IMU streams.
        """
        self._gyroscope_values = np.zeros(3)
        self._accelerometer_values = np.zeros(3)
        for name in self._imu_signals:
            vlib.read_string_stream(self._client, name, True)

    def set_motor_velocities(self, velocities: np.ndarray) -> NoReturn:
        """A method that sets motors velocities and clips if given values exceed
        boundaries.

        Args:
            velocities: Target motors velocities in rad/s.

        """
        if np.shape(velocities)[0] != 2:
            raise ValueError(
                'Dimension of input motors velocities is incorrect!')

        velocities = np.clip(velocities, self.velocity_limit[0],
                             self.velocity_limit[1])

        for i in range(2):
            vlib.set_joint_velocity(
                self._client, self._motor_handles[i], velocities[i])

    def get_encoders_rotations(self) -> np.ndarray:
        """Reads encoders ticks from robot.

        Returns:
            encoder_ticks: Current values of encoders ticks.
        """
        ticks = [vlib.read_float_stream(self._client, name)
                 for name in self._encoder_signals]
        if None not in ticks:
            self._encoder_ticks = np.round(ticks, 3)
        return self._encoder_ticks

    def get_image(self) -> np.ndarray:
        """Reads image from camera mounted to robot.

        Returns:
            img: Image received from robot
        """
        return vlib.get_image(self._client, self._camera_handle)

    def get_proximity_values(self) -> np.ndarray:
        """Reads proximity sensors values from robot.

        Returns:
            proximity_sensor_values: Array of proximity sensor values.
        """
        for i in range(self.nb_proximity_sensor):
            dist = vlib.read_proximity_sensor(
                self._client, self._ultrasonic_handles[i])
            if dist == -1.0:
                self._ultrasonic_values[i - 1] = self.ultrasonic_sensor_bound[1]
            else:
                self._ultrasonic_values[i - 1] = dist
        return self._ultrasonic_values

    def get_accelerometer_values(self) -> np.ndarray:
        """Reads accelerometer values from robot.

        Returns:
            accelerometer_values: Array of values received from accelerometer.
        """
        value = vlib.read_string_stream(self._client, self._imu_signals[0])
        if value is not None and value.shape != (0,):
            self._accelerometer_values = np.round(value, 6)
        return self._accelerometer_values

    def get_gyroscope_values(self) -> np.ndarray:
        """Reads gyroscope values from robot.

        Returns:
            gyroscope_values: Array of values received from gyroscope.
        """
        value = vlib.read_string_stream(self._client, self._imu_signals[1])
        if value is not None and value.shape != (0,):
            self._gyroscope_values = np.round(value, 6)
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
                              dtype=np.float32)

        return np.round(velocities, 3)

    def get_pose(self) -> np.ndarray:
        """Reads current pose of the robot.

        Returns:
            pose: Pose array containing x,y and yaw.
        """
        position = vlib.get_object_position(self._client, self._robot_handle)
        rotation = vlib.get_object_orientation(self._client, self._robot_handle)
        return np.array([position[0], position[1], rotation[2]])

    @property
    def robot_handle(self) -> int:
        return self._robot_handle
