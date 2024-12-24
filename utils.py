# agents/utils.py

import yaml
import os
from autogen import ConversableAgent

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

def instantiate_agents(configs):
    """
    Instantiate ConversableAgent objects from a list of configuration dictionaries.

    Args:
        configs (list of dict): List of agent configurations.

    Returns:
        dict: A dictionary of agent instances keyed by their names.
    """
    agents = {}
    for config in configs:
        name = config.get('name', 'Default_Agent_Name')
        system_message = config.get('system_message', 'Default system message.')
        llm_config = config.get('llm_config', {})
        human_input_mode = config.get('human_input_mode', 'ALWAYS')  # Default to 'ALWAYS' if not specified

        agent = ConversableAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
        )
        agents[name] = agent
    return agents