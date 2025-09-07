"""
A client script for controlling a robot using vision-based actions from a diffusion model server.
This script captures observations from a RealSense camera and the robot's end-effector pose,
sends them to a server, and executes the received actions on the robot.

The script uses the Fairo/Polymetis library for robot control and the RealSense SDK for camera interfacing.
"""

import base64
import time
import traceback
import cv2
import numpy as np
import pyrealsense2 as rs
import requests
import torch

from PIL import Image
from polymetis import RobotInterface, GripperInterface
from scipy.spatial.transform import Rotation as R


# configuration
class Config:
    # server configuration
    SERVER_URL = "http://10.69.55.168:8777"

    # task configuration
    INSTRUCTION = "Slide the red cylinder across the table."

    # robot configuration
    FRANKA_IP = "172.16.0.1"
    GRIPPER_IP = "172.16.0.4"

    # gripper parameters
    GRIPPER_SPEED = 0.1
    GRIPPER_FORCE = 60
    GRIPPER_MAX_WIDTH = 0.08570
    GRIPPER_TOLERANCE = 0.01

    ACTION_STEPS = 1 # number of actions to execute at each step
    OBS_STEPS = 2  # number of observations to send


# create config instance
configs = Config()


# utility functions for pose conversion
def quat2euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to euler angles (xyz) in radians by way of rotation matrix.

    Args:
        quat: Quaternion array in [x, y, z, w] format.

    Returns:
        Euler angles in [x, y, z] format (radians).
    """
    return R.from_quat(quat).as_euler("xyz")

def euler2quat(euler: np.ndarray) -> np.ndarray:
    """Convert euler angles (xyz) in radians to quaternion by way of rotation matrix.

    Args:
        euler: Euler angles in [x, y, z] format (radians).

    Returns:
        Quaternion array in [x, y, z, w] format.
    """
    return R.from_euler("xyz", euler).as_quat()

def add_euler(delta: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Add euler angles by way of conversion to rotation matrix.

    Args:
        delta: Delta euler angles in [x, y, z] format (radians).
        source: Source euler angles in [x, y, z] format (radians).

    Returns:
        Combined euler angles in [x, y, z] format (radians).
    """
    return (R.from_euler("xyz", delta) * R.from_euler("xyz", source)).as_euler("xyz")


class RobotActionExecutor:
    """
    A class for executing robot actions by managing end-effector pose control.

    This class maintains an internal state representation of the robot's expected
    end-effector pose and executes actions by updating the desired pose incrementally.
    It acts as an interface between high-level action commands and low-level robot
    control, similar to the frankaenv environment.

    Attributes:
        robot: The robot interface object used for pose control.
        expected_ee_euler (np.ndarray): Internal state tracking the expected end-effector
            pose as [x, y, z, roll, pitch, yaw].
    """
    def __init__(self, robot):
        self.robot = robot

        # initialize expected pose from robot
        ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])  # [x, y, z, qx, qy, qz, qw]
        position = ee_pose[:3]
        quat = ee_pose[3:]
        euler = quat2euler(quat)  # [roll, pitch, yaw]

        # maintain internal state like frankaenv
        self.expected_ee_euler = np.concatenate([position, euler])  # [x, y, z, roll, pitch, yaw]

    def send_action(self, action: np.ndarray):
        """
        Send an action command to the robot to update its end-effector pose.

        This method takes a delta position action and updates the robot's expected
        end-effector position, then sends the new pose command to the robot controller.

        Args:
            action (np.ndarray): Action array where the first 3 elements represent
                delta position [dx, dy, dz] in Cartesian space. Additional elements
                are currently unused.

        Note:
            - Rotation deltas are currently set to zero (not implemented).
            - Grasp action is not implemented.
            - The method updates internal state tracking and sends pose commands
              to the robot controller.
        """
        dpos = action[:3]
        drotation = [0, 0, 0]
        grasp = 0 # TODO: not implemented

        # update internal state like frankaenv.step()
        self.expected_ee_euler[:3] += dpos
        self.expected_ee_euler[3:] = add_euler(drotation, self.expected_ee_euler[3:])  # shape-safe

        # convert to quaternion and send
        next_pos = self.expected_ee_euler[:3]
        next_quat = euler2quat(self.expected_ee_euler[3:])

        self.robot.update_desired_ee_pose(
            position=torch.from_numpy(next_pos).float(),
            orientation=torch.from_numpy(next_quat).float(),
        )

        print(f"Sent action to robot: dpos {dpos}, drotation {drotation}, grasp {grasp}")


def decode_numpy_json(obj):
    """
    Decode a JSON object that may contain base64-encoded NumPy arrays.

    This function recursively processes JSON objects to reconstruct NumPy arrays
    that were previously encoded using a custom format with base64 encoding.

    Args:
        obj: The JSON object to decode. Can be a dict, list, or any other type.
             If dict contains "__numpy__" key, it's treated as an encoded array.

    Returns:
        numpy.ndarray or original type: If the object is an encoded NumPy array,
        returns the reconstructed array. If it's a list, returns a NumPy array
        of decoded elements. Otherwise, returns the original object unchanged.
    """
    if isinstance(obj, dict) and "__numpy__" in obj:
        array_bytes = base64.b64decode(obj["__numpy__"])
        return np.frombuffer(array_bytes, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    elif isinstance(obj, list):
        return np.array([decode_numpy_json(x) for x in obj])
    else:
        return obj


def get_observation(pipeline, robot):
    obs = {}

    # wait for frames
    frames = pipeline.wait_for_frames()

    # add camera image (as numpy array)
    color_frame = frames.get_color_frame()

    # add camera depth
    depth_frame = frames.get_depth_frame()

    if not color_frame:
        print("Warning: No color frame received.")

    if not depth_frame:
        print("Warning: No depth frame received.")

    # convert to numpy arrays
    color_image = np.zeros((480, 640, 3), dtype=np.uint8) if not color_frame else np.asanyarray(color_frame.get_data())
    depth_image = np.zeros((480, 640, 1), dtype=np.float32) if not depth_frame else np.asanyarray(depth_frame.get_data())

    rgb_color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # convert to proper data types
    raw_rgb_array = np.array(rgb_color_image).astype(np.uint8)
    raw_depth_array = np.array(depth_image).astype(np.float32)

    # construct the obs
    obs["agentview_rgb"] = raw_rgb_array.tolist()
    obs["agentview_depth"] = raw_depth_array.tolist()
    obs["eef_pos"] = robot.get_ee_pose()[0].tolist()

    return obs


def get_server_action(observation_buffer: list):
    """
    Send observations to the server and retrieve the corresponding actions.

    This function makes a POST request to the server's '/act' endpoint with the
    provided observation data and returns the decoded actions response.

    Args:
        observation_buffer (list): The observation buffer to send to the server. Each observation
            should be JSON-serializable data that the server expects.

    Returns:
        list or None: The decoded action array from the server response,
            or None if the request failed or returned an error status code.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    response = requests.post(f"{configs.SERVER_URL}/act", json={"observations": observation_buffer})
    action_raw = response.json()

    if response.status_code != 200:
        print("Error from server:", action_raw)
        return None

    # get decoded actions, expecting shape (num_actions, action_dim)
    actions = decode_numpy_json(action_raw)

    # convert a [num_actions, action_dim] numpy array to a list of actions
    actions_list = [actions[i] for i in range(actions.shape[0])]
    return actions_list


def apply_action(actions, executor, num_actions, step_time=0.5):
    """
    Apply a sequence of actions to an executor with specified timing.

    TODO: Check the default step_time value.

    Args:
        actions: A sequence of actions to be executed
        executor: The executor object that will receive and process the actions
        num_actions (int): The number of actions to execute from the beginning of the actions sequence
        step_time (float, optional): The time delay in seconds between each action execution. Defaults to 0.5.

    Note:
        - This function will block execution for (num_actions * step_time) seconds total.
        - Each action is sent sequentially with the specified delay between them.
    """
    for action in actions[:num_actions]:
        executor.send_action(action)
        time.sleep(step_time)


def main():
    """Main execution loop for robot control with vision-based actions."""
    print("Initializing robot system")

    # initialize robot interfaces
    robot = RobotInterface(ip_address=configs.FRANKA_IP)
    gripper = GripperInterface(ip_address=configs.GRIPPER_IP)
    gripper_state = 1  # 1: open, 0: closed
    robot.start_cartesian_impedance()

    # initialize RealSense camera
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(rs_config)

    # initialize action executor
    robot_action_executor = RobotActionExecutor(robot)
    print(f"Starting diffusion client with instruction: '{configs.INSTRUCTION}'")
    print(f"Server URL: {configs.SERVER_URL}")
    print("-" * 50)

    observation_buffer = []

    try:
        while True:
            # get current observation
            current_obs = get_observation(pipeline, robot)
            current_obs["instruction"] = configs.INSTRUCTION

            # add to buffer
            observation_buffer.append(current_obs)

            if len(observation_buffer) > configs.OBS_STEPS:
                print(f"Buffer full, popping oldest observation")
                observation_buffer.pop(0)

            if len(observation_buffer) < configs.OBS_STEPS:
                print(f"Repeating observations since buffer len is {len(observation_buffer)}")

                # populate buffer with repeated observations
                while len(observation_buffer) < configs.OBS_STEPS:
                    observation_buffer.insert(0, current_obs)

            # get action from server
            print(f"Requesting action from server by sending {len(observation_buffer)} observations")
            actions_list = get_server_action(observation_buffer)

            if actions_list is not None:
                print("-" * 30)
                apply_action(actions_list, robot_action_executor, configs.ACTION_STEPS, step_time=0.5)
            else:
                print("Failed to get action from server")
                break

    except KeyboardInterrupt:
        print("\nShutting down client...")

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()

    finally:
        print("Shutting down robot system...")
        pipeline.stop()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
