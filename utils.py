# agents/utils.py

import yaml
import os
import logging
import json
from difflib import SequenceMatcher
import autogen
from datetime import datetime
import pandas as pd
import re
import unicodedata
from typing import Any

# Configure logger
logger = logging.getLogger(__name__)

def read_data(train = True):
    #Read in all training SAR data
    
    sars = []
    # Gather only training files
    files = [f for f in os.listdir("./data/input") if 'train' in f and f.endswith('.txt')]
    # Sort files by the numeric suffix (e.g., 'sar_train_04.txt' -> 4)
    files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))
    for filename in files:
        file_path = os.path.join("./data/input", filename)
        with open(file_path,'r') as file:
            content = file.read()
            logger.info(f" Read '{filename}' ")
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
            logger.info(f"Loaded configuration for agent '{agent_name}' from {config_file}.")
            return agent_config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_file} not found.")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file {config_file}: {exc}")
        raise
    except KeyError as ke:
        logger.error(ke)
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
            logger.info(f"Found configuration for agent '{agent_name}'.")
            return agent_config
    raise ValueError(f"Agent '{agent_name}' not found.")   

def write_data_to_file(data, file_path):
    """
    Writes the provided data to a file. If the data is a dictionary, writes to a JSON file.
    If the data is a pandas DataFrame, writes to a CSV file.
    Ensures that the target directory exists.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        except OSError as e:
            raise OSError(f"Failed to create directory {directory}: {e}") from e

    try:
        # If data is a JSON string, parse it into a Python object
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                data = parsed
            except json.JSONDecodeError:
                # If it's a plain string (not valid JSON), write as text
                with open(file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(data)
                    logger.info(f"Successfully wrote text data to {file_path}")
                return
        if isinstance(data, dict):
            with open(file_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=4)
                logger.info(f"Successfully wrote JSON data to {file_path}")
        elif isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
            logger.info(f"Successfully wrote DataFrame to CSV file at {file_path}")
        else:
            raise TypeError("Unsupported data type. Only dict and pandas DataFrame are supported.")
    except Exception as e:
        raise IOError(f"Failed to write to file {file_path}: {e}") from e


# Helper function for approximate matching
def approximate_match_ratio(str_a, str_b):
    """
    Returns a similarity ratio in the range [0.0, 1.0].
    1.0 = identical text, 0.0 = completely different.
    """
    return SequenceMatcher(None, str_a, str_b).ratio()


def get_config_list():
    '''
    Prepare Open AI API keys in a format suitable for GPT Assistant Agent
    '''
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

    
def flatten_nested_mapping(nested_dict):
    """
    Convert a nested dictionary like
        {
            'Bank1': {'Acct1': 'Cust1', 'Acct2': 'Cust2'},
            'Bank2': {'Acct3': 'Cust3'}
        }
    into a set of tuples:
        {('Bank1', 'Acct1', 'Cust1'),
         ('Bank1', 'Acct2', 'Cust2'),
         ('Bank2', 'Acct3', 'Cust3')}
    """
    flat_set = set()
    for bank, acct_dict in nested_dict.items():
        for acct, cust_id in acct_dict.items():
            flat_set.add((bank, acct, cust_id))
    return flat_set



def generate_dynamic_output_file_name(filename, output_file_type="json", output_folder="./data/output"):
    """
    Function that creates a dynamic file name so that the file
    will be written into the appropriate folder, with an optional filetype.
    """
    if filename.endswith(".txt"):
        # Get the base name without extension, e.g., "sar1_train" from "sar1_train.txt"
        base_name = os.path.splitext(os.path.basename(filename))[0]
    else:
        base_name = filename

    #Ensure output folder exists, if not create it
    # Ensure the folder exists (or handle the error appropriately)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a timestamp in the format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate the new file name using the provided filetype
    output_filename = f"results_{base_name}_{timestamp}.{output_file_type}"

    # Create the full output file path
    output_file_path = os.path.join(output_folder, output_filename)

    return output_file_path

def concatenate_trxn_sets(gold_narrative):
    """
    Takes a dictionary of the form:
    {
        'account_id': {
            'Trxn_Set_1': 'narrative 1',
            'Trxn_Set_2': 'narrative 2',
            ...
        },
        ...
    }
    and returns a dictionary:
    {
        'account_id': 'narrative 1 <space> narrative 2 <space> ...',
        ...
    }
    """
    new_dict = {}
    for account_id, trxn_sets in gold_narrative.items():
        # Concatenate all transaction set narratives into one string
        combined_narratives = " ".join(trxn_sets.values())
        new_dict[account_id] = combined_narratives
    return new_dict

def split_dictionary_into_subnarratives(data: dict) -> list:
    """
    Given the input dictionary 'data', this function returns a list of new dictionaries.
    Each returned dictionary retains the same fields/keys as the original dictionary,
    except that the 'Narrative' field is narrowed down to exactly one AccountID
    and one Transaction Set.

    :param data: The original dictionary containing 'Entities', 'Account_IDs',
                 'Acct_to_FI', 'Acct_to_Cust', 'FI_to_Acct_to_Cust', and 'Narrative'.
    :return: A list of dictionaries, each having exactly one Narrative entry
             corresponding to one (AccountID, Trxn_Set) pair.
    """
    results = []
    original_narrative = data.get("Narratives", {})

    for acct_id, trxn_sets in original_narrative.items():
        for trxn_set_label, narration_text in trxn_sets.items():
            # Copy all top-level fields except Narrative
            new_dict = {
                "Entities": data["Entities"],
                "Account_IDs": data["Account_IDs"],
                "Acct_to_FI": data["Acct_to_FI"],
                "Acct_to_Cust": data["Acct_to_Cust"],
                "FI_to_Acct_to_Cust": data["FI_to_Acct_to_Cust"],
                # Narrow the Narrative down to one (acct_id, trxn_set_label)
                "Narratives": {
                    acct_id: {
                        trxn_set_label: narration_text
                    }
                }
            }
            results.append(new_dict)

    return results



def assert_dict_structure(testcase, expected, actual):
    """
    Recursively verify that 'actual' has at least the same keys (and sub-keys) as 'expected'.
    This checks structural integrity (i.e., key presence and dict nesting), but not exact values.
    """
    testcase.assertIsInstance(actual, dict, "Expected a dictionary for this level of structure.")
    
    for key, expected_value in expected.items():
        # Check the key exists in the actual dictionary
        testcase.assertIn(key, actual, f"Key '{key}' is missing in the actual dictionary.")
        
        # If the expected value is also a dict, recurse into the structure
        if isinstance(expected_value, dict):
            testcase.assertIsInstance(actual[key], dict, 
                                      f"Key '{key}' should map to a dict in the actual dictionary.")
            assert_dict_structure(testcase, expected_value, actual[key])
        else:
            # If not a dict, we only check that the key is present. 
            # You could also check type: 
            # testcase.assertIsInstance(actual[key], type(expected_value), ...)
            pass


def convert_dict_to_df(i:int,nested_dict:dict):
    """
    Convert a nested dictionary into a pandas DataFrame, 
    using each inner dictionary as a row.
    """
    df = pd.DataFrame.from_dict(nested_dict, orient='index')
    df.reset_index(drop=True, inplace=True)  # Drop the original keys, if you only want the rows
    #df["Transaction_Set"] = i
    return df


##Functions to normalize string output from LLM to regular ASCII

def normalize_entity(text: str) -> str:
    # 1. Unicode compatibility fold
    text = unicodedata.normalize('NFKC', text)
    # 2. Strip diacritics (café → cafe)
    text = ''.join(ch for ch in unicodedata.normalize('NFKD', text)
                   if not unicodedata.combining(ch))
    # 3. Canonicalize punctuation
    reps = {
        '\u2018':"'", '\u2019':"'",  # curly single quotes
        '\u201c':'"', '\u201d':'"',  # curly double quotes
        '\u2013':'-', '\u2014':'-',  # en/em dashes
        '\u2026':'...',              # ellipsis
        '\u00A0':' ', '\u200B':'',   # non-breaking & zero-width spaces
    }
    for orig, repl in reps.items():
        text = text.replace(orig, repl)
    # 4. Remove control/nonprintables
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    # 5. Normalize thousands separators in numbers
    text = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)
    # 6. Collapse whitespace and trim,
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_dict(obj: Any) -> Any:
    if isinstance(obj, str):
        return normalize_entity(obj)
    elif isinstance(obj, dict):
        return { key: normalize_dict(val) for key, val in obj.items() }
    elif isinstance(obj, list):
        return [ normalize_dict(item) for item in obj ]
    else:
        return obj  # leave ints, floats, bools, None untouched

