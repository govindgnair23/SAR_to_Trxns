# main.py
from autogen import initiate_chats
from utils import load_agents_from_single_config , get_agent_config, write_dict_to_json_file, read_file
from agents.agents import instantiate_all_base_agents
from dotenv import load_dotenv
import os
import json
from autogen import Cache
from typing import  Dict, Any
import ast
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

def run_agentic_workflow(sar_text: str,config_file:str) -> Dict[str, Any]:
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

        # chat_results = agents["SAR_Agent"].initiate_chats(
        # [
        #     {
        #         "recipient":  agents["Entity_Extraction_Agent"],
        #         "message": sar_text,
        #         "max_turns": 1,
        #         "summary_method": "reflection_with_llm",
        #         "summary_args": {
        #         "summary_prompt" :  ee_summary_prompt                                                               
        #             },
        #     } ] )
        
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

def main(filename):
    load_dotenv()
    api_key = os.getenv("OPEN_API_KEY")
    config_file = 'configs/agents_config.yaml'  # Update this path as needed
    
    # train_sars = read_data(train=True)
    message = read_file(filename)
    results = run_agentic_workflow(message,config_file)
    print(results)
  
    # output_file_path = "./data/output/results0.json"
    # write_dict_to_json_file(results_dict, output_file_path)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read and process a file.")
    parser.add_argument("filename", help="Name of the file to be read and processed")
    args = parser.parse_args()
    main(args.filename)
