from autogen import ConversableAgent , initiate_chats, GroupChat , GroupChatManager
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.function_utils import get_function_schema
import logging
import json
import ast
from utils import get_agent_config, get_config_list
from agents.tools import generate_transactions


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def instantiate_all_base_agents(configs):
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

        logging.info(f"Loaded configuration for agent '{name}' ")

        if name == "Transaction_Generation_Agent_w_Tool":
            continue  #Need to replace with instantiation of GPTAsistantAgent

        agent = ConversableAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
        )

        logging.info(f"Instantiated '{name}' ")
        agents[name] = agent

        
    return agents

def instantiate_base_agent(agent_name, config):
    """
    Instantiate a ConversableAgent with the given configuration.

    Args:
        agent_name (str): Name of the agent.
        config (dict): Configuration dictionary for the agent.

    Returns:
        ConversableAgent: An instance of ConversableAgent.
    """
    try:
        # Extract configuration parameters
        system_message = config['system_message']
        llm_config = config['llm_config']
        human_input_mode = config['human_input_mode']

        # Instantiate the agent
        agent = ConversableAgent(
            name=agent_name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
        )
        logging.info(f"Agent '{agent_name}' instantiated successfully.")
        return agent
    except Exception as e:
        logging.error(f"Failed to instantiate agent '{agent_name}': {e}")
        raise


def create_two_agent_chat(sender_agent , receiver_agent, message,summary_prompt):
    '''
    Creates a two way chat where the sender sends a message which is acted on by the receiver
    
    '''
    
    chat_results = sender_agent.initiate_chats(
    [
        {
            "recipient":  receiver_agent,
            "message": message,
            "max_turns": 1,
            "summary_method": "reflection_with_llm",
            "summary_args": {
            "summary_prompt" :  summary_prompt                                                               
                },
        } ] )
    

    results = chat_results[0].summary
    logging.info("Results returned by LLM")
    cleaned_results = results.strip("```python\n").strip("```")
    logging.info(f"Results cleaned")
    # Convert to dictionary
    results_dict = ast.literal_eval(cleaned_results)
    assert isinstance(results_dict,dict), "results is not a dictionary"
    logging.info(f"Results converted to a Python dictionary")
    print(results_dict)
    return results_dict


def instantiate_agents_for_trxn_generation(configs):
    '''
    Instantiates all agents necessary for trxn generation from a narrative and other inputs
    '''
    agents = {}
    ###############################################################################
    # Agent 1: Instantiate SAR agent who sends trxns to the trxn generation agents
    ##############################################################################
    SAR_Agent_2_config = get_agent_config(configs, agent_name = "SAR_Agent_2")
    try:
        # Extract configuration parameters
        agent_name = SAR_Agent_2_config.get('name', 'Default_Agent_Name')
        system_message = SAR_Agent_2_config.get('system_message')
        llm_config = SAR_Agent_2_config.get('llm_config')
        human_input_mode = SAR_Agent_2_config.get('human_input_mode')
        code_execution_config = SAR_Agent_2_config.get('code_execution_config')

        logging.info(f"Loaded configuration for SAR Agent 2")

        # Instantiate the agent
        agent1 = ConversableAgent(
            name=agent_name,
            system_message=system_message,
            llm_config=llm_config,
            code_execution_config = code_execution_config,
            human_input_mode=human_input_mode,
        )
        logging.info("SAR Agent 2 instantiated successfully.")
        agents["SAR_Agent_2"] = agent1
    except Exception as e:
        logging.error("Failed to instantiate SAR Agent 2")
        raise
    
    ##########################################################
    # Agent 2: Instantiate Transaction Generation without tool
    ##########################################################

    trxn_generation_agent_config = get_agent_config(configs, agent_name = "Transaction_Generation_Agent")
    try:
        # Extract configuration parameters
        agent_name = trxn_generation_agent_config.get('name', 'Default_Agent_Name')
        system_message = trxn_generation_agent_config.get('system_message')
        llm_config = trxn_generation_agent_config.get('llm_config')
        human_input_mode = trxn_generation_agent_config.get('human_input_mode')
        code_execution_config = trxn_generation_agent_config.get("trxn_generation_agent_config", False)
        # summary_method = trxn_generation_agent_config.get('summary_method')
        # summary_prompt = trxn_generation_agent_config.get('summary_prompt')

        logging.info(f"Loaded configuration for Trxn Generation Agent")

        # Instantiate the agent
        agent2 = ConversableAgent(
            name=agent_name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
            code_execution_config =code_execution_config
        )
        logging.info("Transaction_Generation_Agent instantiated successfully.")

        agents["Transaction_Generation_Agent"] = agent2
    except Exception as e:
        logging.error("Failed to instantiate Transaction_Generation_Agent")
        raise

    # logging.info
    # config_list = config_list_from_dotenv(
    # dotenv_file_path="../.env",
    # model_api_key_map={
    #     "gpt-4o": "OPENAI_API_KEY", 
    #     "gpt-4o-mini": "OPENAI_API_KEY"
    # },
    # filter_dict={"model":["gpt-4o-mini"]}
    #  )

    #Assistant API Tool Schema for Trxn Generation
    generate_transactions_schema = get_function_schema(
    generate_transactions,
    name = "generate_transactions",
    description = " A function for generating transactions when a large number of transactions have to be synthesizes"
                )

    ###########################################################
    # Agent 3: Instantiate Transaction Generation with tool use
    ###########################################################

    config_list = get_config_list()

    trxn_generation_agent_w_tool_config = get_agent_config(configs, agent_name = "Transaction_Generation_Agent")
    try:
        # Extract configuration parameters
        agent_name = trxn_generation_agent_w_tool_config.get('name', 'Default_Agent_Name')
        instructions = trxn_generation_agent_w_tool_config.get('instructions')
        llm_config = trxn_generation_agent_w_tool_config.get('llm_config')
        # overwrite_instructions = trxn_generation_agent_w_tool_config.get('overwrite_instructions')
        # overwrite_tools = trxn_generation_agent_w_tool_config.get('overwrite_tools')

        logging.info(f"Loaded configuration for Trxn Generation Agent with Tool")

        # Instantiate the agent
        agent3 = GPTAssistantAgent(
            name=agent_name,
            instructions=instructions,
            llm_config= {
                    "config_list":config_list,
                    "tools":[generate_transactions_schema]

                },
        )
        logging.info("Transaction_Generation_Agent with tool  instantiated successfully.")
        agents["Transaction_Generation_Agent_w_tool"] = agent3
    except Exception as e:
        logging.error("Failed to instantiate Transaction_Generation_Agent with tool")
        raise


    return agents


    