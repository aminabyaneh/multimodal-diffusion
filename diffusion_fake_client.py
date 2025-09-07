import base64
import requests
import numpy as np
import traceback

class Config:
    """
    Configuration class for the diffusion fake client.
    Contains server URL, instruction, action steps, and observation steps.

    Note: Adjust the params before running the client.
    """
    SERVER_URL = "http://localhost:8777"
    INSTRUCTION = "Slide the red cylinder across the table."
    action_steps = 1
    OBS_STEPS = 2

configs = Config()


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


def get_fake_observation():
    """
    Generate fake observation data for testing purposes.

    Creates mock sensor data including RGB and depth images from an agent's viewpoint,
    as well as end-effector position information.

    Returns:
        dict: A dictionary containing:
            - "agentview_rgb" (list): Flattened RGB image data as a list of shape (640, 480, 3)
            - "agentview_depth" (list): Flattened depth image data as a list of shape (640, 480, 1)
            - "eef_pos" (list): End-effector position as [x, y, z] coordinates, defaults to [0.0, 0.0, 0.0]
    """
    obs = {}
    color_image = np.zeros((640, 480, 3), dtype=np.uint8)
    depth_image = np.zeros((640, 480, 1), dtype=np.uint8)

    # fake observations
    obs["agentview_rgb"] = color_image.tolist()
    obs["agentview_depth"] = depth_image.tolist()
    obs["eef_pos"] = [0.0, 0.0, 0.0]

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


if __name__ == "__main__":
    print(f"Starting diffusion client with instruction: '{configs.INSTRUCTION}'")
    print(f"Server URL: {configs.SERVER_URL}")
    print("-" * 50)

    observation_buffer = []

    try:
        # Main loop
        while True:
            # Get fake observation data
            observation = get_fake_observation()
            observation["instruction"] = configs.INSTRUCTION
            observation_buffer.append(observation)

            if len(observation_buffer) > configs.OBS_STEPS:
                print(f"Buffer full, popping oldest observation")
                observation_buffer.pop(0)

            if len(observation_buffer) < configs.OBS_STEPS:
                print(f"Repeating observations since buffer len is {len(observation_buffer)}")

                # populate buffer with repeated observations
                while len(observation_buffer) < configs.OBS_STEPS:
                    observation_buffer.insert(0, observation)

            print(f"Sending observation with keys: {list(observation.keys())} and len buffer: {len(observation_buffer)}")

            # Send to server and get action
            action_list = get_server_action(observation_buffer)

            if action_list is not None:
                for i, action in enumerate(action_list):
                    print(f"Received action for observation {i}: {action}")
                print("-" * 30)
            else:
                print("Failed to get action from server")
                break

    except KeyboardInterrupt:
        print("\nShutting down client...")

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()
