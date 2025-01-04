from autogen import ConversableAgent , initiate_chats
import logging
import json

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

        agent = ConversableAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
        )
        agents[name] = agent

        logging.info(f"Loaded configuration for agent '{name}' ")
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

    try:
        results_dict = json.loads(results)
    except json.JSONDecodeError as e:
         print("Failed to decode JSON:", e)

    return results_dict