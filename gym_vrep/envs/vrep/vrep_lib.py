import os
from typing import NoReturn
from typing import Sequence
from typing import Union

import numpy as np

from gym_vrep.envs.vrep import vrep


def connect(port: int, synchronous: bool) -> int:
    print('Connecting to V-REP on port {}'.format(port))
    client = vrep.simxStart('127.0.0.1', port, True, True, 1000, 5)
    assert client >= 0, 'Connection to V-REP failed!'

    vrep.simxSynchronous(client, synchronous)

    return client


def start_simulation(client: int, synchronous: bool):
    vrep.simxSynchronous(client, synchronous)
    vrep.simxStartSimulation(client, vrep.simx_opmode_blocking)


def stop_simulation(client: int):
    vrep.simxStopSimulation(client, vrep.simx_opmode_blocking)


def disconnect(client: int):
    vrep.simxStopSimulation(client, vrep.simx_opmode_blocking)
    vrep.simxCloseScene(client, vrep.simx_opmode_blocking)
    vrep.simxFinish(client)


def trigger_simulation_step(client):
    vrep.simxSynchronousTrigger(client)
    vrep.simxGetPingTime(client)


def load_scene(client: int, scene: str, path: str):
    """Loads given scene in V-REP simulator.

    Args:
        client:
        scene: Name of the scene that will be loaded.
        path:

    """
    scenes_list = os.listdir(path)
    assert (scene + '.ttt' in scenes_list), 'Scene not found!'

    scene_path = os.path.join(path, scene + '.ttt')
    res = vrep.simxLoadScene(client, scene_path, 1, vrep.simx_opmode_blocking)
    assert (res == vrep.simx_return_ok), 'Could not load scene!'


def load_model(client: int, model: str, path: str):
    """Loads given model into V-REP scene

    Args:
        client:
        model: Name of the model taht will be loaded.
        path:

    """
    models_list = os.listdir(path)
    assert (model + '.ttm' in models_list), 'Model not found!'

    model_path = os.path.join(path, model + '.ttm')
    res, _ = vrep.simxLoadModel(client, model_path, 1,
                                vrep.simx_opmode_blocking)
    assert (res == vrep.simx_return_ok), 'Could not load model!'


def is_headless(client: int) -> bool:
    return _get_boolparam(vrep.sim_boolparam_headless, client)


def set_thread_rendering(client: int, value: bool):
    _set_boolparam(parameter=vrep.sim_boolparam_threaded_rendering_enabled,
                   value=value,
                   client=client)


def set_hierarchy_visibility(client: int, value: bool):
    _set_boolparam(vrep.sim_boolparam_hierarchy_visible, value, client)


def set_console_visibility(client: int, value: bool):
    _set_boolparam(vrep.sim_boolparam_console_visible, value, client)


def set_browser_visbility(client: int, value: bool):
    _set_boolparam(vrep.sim_boolparam_browser_visible, value, client)


def set_simulation_time_step(client: int, dt: float):
    _set_floatparam(vrep.sim_floatparam_simulation_time_step, dt, client)


def get_object_handle(client: int, name: str) -> int:
    res, object_handle = vrep.simxGetObjectHandle(client, name,
                                                  vrep.simx_opmode_blocking)
    assert (res == vrep.simx_return_ok or
            res == vrep.simx_return_novalue_flag), \
        'Could not obtain handle of {}! Error code: {}'.format(name, res)
    return object_handle


def set_object_position(client: int,
                        object_handle: int,
                        position: Union[Sequence[float], np.ndarray]):
    assert len(position) == 3, 'Wrong size of the position array!'
    res = vrep.simxSetObjectPosition(
        client, object_handle, -1, position, vrep.simx_opmode_blocking)
    assert (res == vrep.simx_return_ok or
            res == vrep.simx_return_novalue_flag), \
        'Could not set model position! Error code: {}'.format(res)


def set_object_orientation(client: int,
                           object_handle: int,
                           orientation: Union[Sequence[float], np.ndarray]):
    assert len(orientation) == 3, 'Wrong size of the orientation array!'
    res = vrep.simxSetObjectOrientation(
        client, object_handle, -1, orientation, vrep.simx_opmode_blocking)
    assert (res == vrep.simx_return_ok or
            res == vrep.simx_return_novalue_flag), \
        'Could not set model orientation! Error code: {}'.format(res)


def set_joint_velocity(client: int, object_handle: int, velocity: float):
    res = vrep.simxSetJointTargetVelocity(client, object_handle, velocity,
                                          vrep.simx_opmode_oneshot)
    assert (res == vrep.simx_return_ok or
            res == vrep.simx_return_novalue_flag), \
        'Could not set joint target velocity! Error code: {}'.format(res)


def read_proximity_sensor(client: int,
                          object_handle: int,
                          stream: bool = False) -> Union[float, None]:
    res, state, points, _, _ = vrep.simxReadProximitySensor(
        client, object_handle, _switch_streaming_buffer(stream))
    if res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag:
        if state:
            dist = np.sqrt(points[0] ** 2 + points[1] ** 2 + points[2] ** 2)
            return np.round(dist, 3)
        return -1.0
    print('Could not read proximity sensor! Error code: {}'.format(res))
    return None


def read_string_stream(client: int,
                       name: str,
                       stream: bool = False) -> Union[np.ndarray, None]:
    res, value = vrep.simxReadStringStream(client, name,
                                           _switch_streaming_buffer(stream))
    if res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag:
        return np.asarray(vrep.simxUnpackFloats(value))
    print('Could not read string stream {}! Error code: {}'.format(name, res))
    return None


def read_float_stream(client, name, stream=False):
    res, value = vrep.simxGetFloatSignal(client, name,
                                         _switch_streaming_buffer(stream))
    if res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag:
        return value
    print('Could not read float stream {}! Error code: {}'.format(name, res))
    return None


def get_image(client, object_hadle, stream=False):
    res, resolution, image = vrep.simxGetVisionSensorImage(
        client, object_hadle, False, _switch_streaming_buffer(stream))
    if res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag:
        img = np.array(image, dtype=np.uint8)
        img.resize([resolution[1], resolution[0], 3])
        return img
    print('Could not get camera image! Error code: {}'.format(res))
    return None


def get_object_position(client: int,
                        object_handle: int,
                        reference: int = -1,
                        stream: bool = False):
    res, value = vrep.simxGetObjectPosition(
        client, object_handle, reference, _switch_streaming_buffer(stream))
    if res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag:
        return np.round(value, 3)
    print('Could not read object position! Error code: {}'.format(res))
    return None


def get_object_orientation(client: int,
                           object_handle: int,
                           reference: int = -1,
                           stream: bool = False):
    res, value = vrep.simxGetObjectOrientation(
        client, object_handle, reference, _switch_streaming_buffer(stream))
    if res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag:
        return np.round(value, 3)
    print('Could not read object orientation! Error code: {}'.format(res))
    return None


def _set_boolparam(parameter: int, value: bool, client: int) -> NoReturn:
    """Sets boolean parameter of V-REP simulation.

    Args:
        parameter: Parameter to be set.
        value: Boolean value to be set.

    """
    res = vrep.simxSetBooleanParameter(client, parameter, value,
                                       vrep.simx_opmode_oneshot)
    assert (res == vrep.simx_return_ok or
            res == vrep.simx_return_novalue_flag), \
        'Could not set boolean parameter! Error code: {}'.format(res)


def _set_floatparam(parameter: int, value: float, client: int) -> NoReturn:
    """Sets float parameter of V-REP simulation.

    Args:
        parameter: Parameter to be set.
        value: Float value to be set.

    """
    res = vrep.simxSetFloatingParameter(client, parameter, value,
                                        vrep.simx_opmode_oneshot)
    assert (res == vrep.simx_return_ok or
            res == vrep.simx_return_novalue_flag), \
        'Could not set float parameter! Error code: {}'.format(res)


def _get_boolparam(parameter: int, client: int) -> bool:
    res, value = vrep.simxGetBooleanParameter(client, parameter,
                                              vrep.simx_opmode_oneshot)
    assert (res == vrep.simx_return_ok or
            res == vrep.simx_return_novalue_flag), \
        'Could not get boolean parameter! Error code: {}'.format(res)
    return value


def _switch_streaming_buffer(stream: bool = False):
    if stream:
        return vrep.simx_opmode_streaming
    return vrep.simx_opmode_buffer
