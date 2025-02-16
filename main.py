# main.py
from autogen import initiate_chats ,GroupChat, GroupChatManager
from utils import load_agents_from_single_config , get_agent_config, write_dict_to_json_file, read_file, get_config_list
from agents.agents import instantiate_all_base_agents, instantiate_agents_for_trxn_generation
from dotenv import load_dotenv
import os
import json
from autogen import Cache
from typing import  Dict, Any
import ast
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

def run_agentic_workflow1(sar_text: str,config_file:str) -> Dict[str, Any]:
    '''
    Runs the full agentic workflow and returns results as a dictionary. 
    To be updated when more agents are added to the workflow.
    
    '''

    assert sar_text and sar_text.strip(), "SAR narrative must not be empty or whitespace only"

    agent_configs = load_agents_from_single_config(config_file)
    agents = instantiate_all_base_agents(agent_configs)
    logging.info("All agents instantiated successfully")

    #Dictionary to store other relevant config of each agent
    agent_config_dict = {}

    # Iterate over agents and  configure 
    for name, _ in agents.items():
        agent_config_dict[name] = {} # Dictonary to store other relevant configs for each agent
        agent_config = get_agent_config(agent_configs, name) #Get configuratins for the specific agent
        agent_config_dict[name]["summary_prompt"] = agent_config.get("summary_prompt")
        agent_config_dict[name]["summary_method"] = agent_config.get("summary_method")
        agent_config_dict[name]["max_turns"] = agent_config.get("max_turns",1)
        logging.info("  Additional configs for Agent %s read successfully",name)
        
    

    # Use DiskCache as cache
    with Cache.disk() as cache:
        
        chat_results = agents["SAR_Agent"].initiate_chats(
              [
                {
                    "recipient": agents["Entity_Extraction_Agent"],
                    "message": sar_text,
                    "max_turns": agent_config_dict["Entity_Extraction_Agent"]["max_turns"],
                    "summary_method": agent_config_dict["Entity_Extraction_Agent"]["summary_method"],
                    "summary_args": {
                        "summary_prompt" : agent_config_dict["Entity_Extraction_Agent"]["summary_prompt"]                                                                
                    },
                },

                {
                    "recipient": agents["Entity_Resolution_Agent"],
                    "message": sar_text,
                    "max_turns": agent_config_dict["Entity_Resolution_Agent"]["max_turns"],
                    "summary_method": agent_config_dict["Entity_Resolution_Agent"]["summary_method"],
                    "summary_args": {
                        "summary_prompt" : agent_config_dict["Entity_Resolution_Agent"]["summary_prompt"] 
                                        },
                },
                {
                    "recipient": agents["Narrative_Extraction_Agent"],
                    "message": sar_text,
                    "max_turns": agent_config_dict["Narrative_Extraction_Agent"]["max_turns"],
                    "summary_method": agent_config_dict["Narrative_Extraction_Agent"]["summary_method"],
                    "summary_args": {
                        "summary_prompt" : agent_config_dict["Narrative_Extraction_Agent"]["summary_prompt"] 
                                        },
                },
                

            ] 

                                                             )


    results0 = chat_results[0].summary
    results1 = chat_results[1].summary
    results2 = chat_results[2].summary

    
    cleaned_results0 = results0.strip("```python\n").strip("```")
    cleaned_results1 = results1.strip("```python\n").strip("```")
    cleaned_results2 = results2.strip("```python\n").strip("```")
    # Convert to dictionary
    results_dict0 = ast.literal_eval(cleaned_results0)
    logging.info("Results from Entity Extraction Agent converted to a dictionary")
    results_dict1 = ast.literal_eval(cleaned_results1)
    logging.info("Results from Entity Resolution Agent converted to a dictionary")
    results_dict2 = ast.literal_eval(cleaned_results2)
    logging.info("Results from Narrative Extraction Agent converted to a dictionary")

    # Combine results from first agentic workflow
    results = {**results_dict0,**results_dict1,**results_dict2}
    
    return results

# Not passing input dict to check instantiation
def run_agentic_workflow2(input:Dict, config_file:str) -> Dict[str, Dict[int, Dict[str, Any]]] :
    #Convert Dict to string to pass to LLM
    input = repr(input)
    agent_configs = load_agents_from_single_config(config_file)
    agents = instantiate_agents_for_trxn_generation(agent_configs)
    n_agents = len(agents)
    assert len(agents)==3 , f"The 3 agents required for trxn generation have not been passed. Only {n_agents} agents have been created"

    logging.info("All agents instantiated successfully")
    agents_list = list(agents.values())

    ##########################################################
    # Instantiate Group Chat Manager Agent
    ##########################################################

    group_chat_manager_config = get_agent_config(config_file, agent_name = "Group_Chat_Manager")
    try:
        # Extract configuration parameters
        
        llm_config = group_chat_manager_config.get('llm_config')
        summary_method = group_chat_manager_config.get("summary_method")
        summary_prompt = group_chat_manager_config.get("summary_prompt")
        logging.info(f"Loaded configuration for Group Chat and Group Chat Manager")
        groupchat = GroupChat(agents = agents_list,messages=[],max_round=2)
        manager = GroupChatManager(groupchat=groupchat, llm_config = llm_config)

    except Exception as e:
        logging.error("Failed to instantiate Group Chat Manager")
        raise

    # Use DiskCache as cache
    with Cache.disk() as cache:
        chat_results = agents["SAR_Agent_2"].initiate_chat(
                    manager,
                    message = input,
                    summary_method= summary_method,
                    summary_args = {
                        "summary_prompt" : summary_prompt
                    } )
    results = chat_results.summary
    cleaned_results = results.strip("```python\n").strip("```")
    # Convert to dictionary
    results_dict = ast.literal_eval(cleaned_results)
    logging.info("Results from  Transaction Generation Agent converted to a dictionary")

    return results_dict

def main(filename):
    load_dotenv()
    api_key = os.getenv("OPEN_API_KEY")
    config_file = 'configs/agents_config.yaml'  # Update this path as needed
    
    # train_sars = read_data(train=True)
    message = read_file(filename)

    # Run first agentic workflow
    results1 = run_agentic_workflow1(message,config_file)
    
    # Run second agentic worklfow
    results2 = run_agentic_workflow2(results1,config_file)
    
    return results1, results2
    # output_file_path = "./data/output/results0.json"
    # write_dict_to_json_file(results_dict, output_file_path)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read and process a file.")
    parser.add_argument("filename", help="Name of the file to be read and processed")
    args = parser.parse_args()
    main(args.filename)
