# agents/utils.py

import yaml
import os
import logging
import json
from difflib import SequenceMatcher
import autogen
import unittest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_data(train = True):
    #Read in all training SAR data
    
    sars = []
    for filename in os.listdir("./data/input"):
        if 'train' in filename and filename.endswith('.txt'):
            file_path = os.path.join("./data/input", filename)
            with open(file_path,'r') as file:
                content = file.read()
                logging.info(f" Read '{filename}' ")
                sars.append(content)

    return sars

import os

def read_file(filename, base_dir="./data/input"):
    """
    Reads a specific SAR text file from a given directory.

    Args:
        filename (str): The name of the file to read.
        base_dir (str): The base directory where the file is located. Defaults to "./data/input".

    Returns:
        str: The content of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For other errors encountered during file processing.
    """
    file_path = os.path.join(base_dir, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Further processing can be done here if needed.
            return content
    except FileNotFoundError:
        # Raising an error can be more flexible than printing, as it allows the caller to handle it.
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while processing the file: {e}")


def load_config(file_path):
    """
    Load configuration from a YAML file.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing configuration values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If there is an error parsing the YAML file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The configuration file '{file_path}' does not exist.")

    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing the YAML configuration file '{file_path}': {e}")

def load_agents_from_single_config(file_path):
    """
    Load and parse a single YAML configuration file containing multiple agents.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        list of dict: A list containing configuration dictionaries for each agent.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config.get('agents', [])
    except FileNotFoundError:
        print(f"Configuration file {file_path} not found.")
        raise
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file {file_path}: {exc}")
        raise

def load_single_agent_config(config_file, agent_name):
    """
    Load and parse the configuration for a single agent from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.
        agent_name (str): Name of the agent to load.

    Returns:
        dict: Configuration dictionary for the specified agent.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        KeyError: If the specified agent is not found in the configuration.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            agents = config.get('agents', {})
            if agent_name not in agents:
                raise KeyError(f"Agent '{agent_name}' not found in configuration.")
            agent_config = agents[agent_name]
            logging.info(f"Loaded configuration for agent '{agent_name}' from {config_file}.")
            return agent_config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_file} not found.")
        raise
    except yaml.YAMLError as exc:
        logging.error(f"Error parsing YAML file {config_file}: {exc}")
        raise
    except KeyError as ke:
        logging.error(ke)
        raise

def get_agent_config(agent_configs, agent_name):
    """
    Retrieve the configuration for a specific agent by name.

    Args:
        agents (list of dict): List of agent configurations.
        agent_name (str): Name of the agent to retrieve.

    Returns:
        dict: Configuration dictionary for the specified agent.

    Raises:
        ValueError: If the agent is not found.
    """
    for agent_config in agent_configs:
        if agent_config.get('name') == agent_name:
            logging.info(f"Found configuration for agent '{agent_name}'.")
            return agent_config
    raise ValueError(f"Agent '{agent_name}' not found.")   

def write_dict_to_json_file(data_dict, file_path):
    """
    Writes the provided dictionary to a JSON file at the specified path.
    Ensures that the target directory exists.
    """
    # Extract directory from file path
    directory = os.path.dirname(file_path)
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except OSError as e:
            raise OSError(f"Failed to create directory {directory}: {e}") from e
    
    # Write dictionary to JSON file with pretty formatting
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data_dict, json_file, indent=4)
            print(f"Successfully wrote data to {file_path}")
    except IOError as e:
        raise IOError(f"Failed to write to file {file_path}: {e}") from e


# Helper function for approximate matching
def approximate_match_ratio(str_a, str_b):
    """
    Returns a similarity ratio in the range [0.0, 1.0].
    1.0 = identical text, 0.0 = completely different.
    """
    return SequenceMatcher(None, str_a, str_b).ratio()


def get_config_list():
    config_list = autogen.config_list_from_dotenv(
    dotenv_file_path="./.env",
    model_api_key_map={
        "gpt-4o": "OPENAI_API_KEY", 
        "gpt-4o-mini": "OPENAI_API_KEY"
    },
    filter_dict={"model":["gpt-4o-mini"]}
    )

    return config_list
    
def compare_dicts(test_case, actual, expected, path="root"):
    """
    Recursively compare two dictionaries, raising detailed AssertionError
    if there is any mismatch in keys or values, indicating where the mismatch occurs.
    """

    # Check all keys in the expected dictionary
    for key in expected:
        new_path = f"{path}['{key}']"  # Path to this key
        test_case.assertIn(key, actual, f"Missing key at {new_path} in the actual dictionary")

        # If the value is itself a dict, recurse; otherwise, do a direct comparison
        if isinstance(expected[key], dict):
            test_case.assertIsInstance(
                actual[key], dict,
                f"Type mismatch at {new_path}: expected dict, got {type(actual[key])}"
            )
            compare_dicts(test_case, actual[key], expected[key], new_path)
        else:
            test_case.assertEqual(
                actual[key],
                expected[key],
                f"Value mismatch at {new_path}: expected {expected[key]}, got {actual[key]}"
            )

    # Check if there are any extra keys in the actual dictionary that are not in expected
    for key in actual:
        new_path = f"{path}['{key}']"
        test_case.assertIn(key, expected, f"Unexpected key at {new_path} found in actual dictionary")

    
