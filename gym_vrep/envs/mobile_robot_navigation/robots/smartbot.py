import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.robots.mobiles.nonholonomic_base import NonHolonomicBase
from pyrep.sensors.accelerometer import Accelerometer
from pyrep.sensors.gyroscope import Gyroscope


class SmartBot(NonHolonomicBase):
    velocity_limit = np.array([0.0, 15.0])
    nb_ultrasonic_sensor = 5
    ultrasonic_sensor_bound = np.array([0.02, 2.0])

    def __init__(self, count: int = 0, enable_vision: bool = False):
        super().__init__(count, 2, 'smartbot')

        self._enable_vision = enable_vision

        self._camera = VisionSensor('{}_camera'.format(self.get_name()))
        self._gyroscope = Gyroscope('{}_gyro_sensor'.format(self.get_name()))
        self._accelerometer = Accelerometer(
            '{}_accelerometer'.format(self.get_name()))

        self._ultrasonic_values = 2 * np.ones(self.nb_ultrasonic_sensor)
        self._ultrasonic_sensors = [ProximitySensor(
            '{}_ultrasonic_sensor_{}'.format(self.get_name(), i + 1))
            for i in range(self.nb_ultrasonic_sensor)]
        
        self._previous_joint_positions = self.get_joint_positions()

        self.initial_configuration = self.get_configuration_tree()

    @property
    def ultrasonic_distances(self) -> np.ndarray:
        """Reads proximity sensors values from robot.

        Returns:
            proximity_sensor_values: Array of proximity sensor values.
        """
        for i in range(self.nb_ultrasonic_sensor):
            dist = self._ultrasonic_sensors[i].read()
            if dist == -1.0:
                self._ultrasonic_values[i - 1] = self.ultrasonic_sensor_bound[1]
            else:
                self._ultrasonic_values[i - 1] = dist
        return self._ultrasonic_values

    @property
    def image(self) -> np.ndarray:
        """Reads image from camera mounted to robot.

        Returns:
            img: Image received from robot
        """
        return self._camera.capture_rgb()

    @property
    def accelerations(self) -> np.ndarray:
        """Reads accelerometer values from robot.

        Returns:
            accelerometer_values: Array of values received from accelerometer.
        """
        values = self._accelerometer.read()
        return np.round(values, 4)

    @property
    def angular_velocities(self) -> np.ndarray:
        """Reads gyroscope values from robot.

        Returns:
            gyroscope_values: Array of values received from gyroscope.
        """
        values = self._gyroscope.read()
        return np.round(values, 4)

    @property
    def encoder_ticks(self) -> np.ndarray:
        """Reads encoders ticks from robot.
        Returns:
            encoder_ticks: Current values of encoders ticks.
        """
        dphi = np.asarray(self.get_joint_positions()) \
               - np.asarray(self._previous_joint_positions)
        self._previous_joint_positions = self.get_joint_positions()

        def correct_angle(angle):
            new_angle = None
            if angle >= 0:
                new_angle = np.fmod((angle + np.pi), (2 * np.pi)) - np.pi

            if angle < 0:
                new_angle = np.fmod((angle - np.pi), (2 * np.pi)) + np.pi

            new_angle = np.round(angle, 3)
            return new_angle

        return np.asarray([correct_angle(angle) for angle in dphi])
