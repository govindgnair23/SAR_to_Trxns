from autogen import GroupChat, GroupChatManager
from utils import load_agents_from_single_config , get_agent_config, split_dictionary_into_subnarratives,convert_trxn_dict_to_df,generate_dynamic_output_file_name , write_data_to_file, normalize_dict
from agents.agents import instantiate_all_base_agents, instantiate_agents_for_trxn_generation
from autogen import Cache
from typing import  Dict, Any, List
import ast
import logging
import json
import pandas as pd
import copy

logger = logging.getLogger(__name__)

def run_agentic_workflow1(sar_text: str,config_file:str) -> Dict[str, Any]:
    '''
    Runs the full agentic workflow and returns results as a dictionary. 
    To be updated when more agents are added to the workflow.
    
    '''

    assert sar_text and sar_text.strip(), "SAR narrative must not be empty or whitespace only"
    logger.info(f"Starting run_agentic_workflow1 for SAR text (length={len(sar_text)})")

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
    logging.info("Results  from agents aggrgated into a single dictionary")

    results = normalize_dict(results)
    logging.info("Results normalized to remove unexpected characters")

    #Write output file for later reuse or verification
    output_file = generate_dynamic_output_file_name(filename="entity_metrics",output_file_type="json",
                                                    output_folder="./data/output")
    
    write_data_to_file(results,output_file)
    
    logger.info("Finished run_agentic_workflow1")
    return results


def run_agentic_workflow2(input:Dict, config_file:str) -> List[Dict[str, Dict[int, Dict[str, Any]]]] :
    
    agent_configs = load_agents_from_single_config(config_file)
    agents = instantiate_agents_for_trxn_generation(agent_configs)
    n_agents = len(agents)
    assert len(agents)==3 , f"The 3 agents required for trxn generation have not been passed. Only {n_agents} agents have been created"
    logger.info(f"Starting run_agentic_workflow2 with input keys={list(input.keys())}")

    logging.info("All agents instantiated successfully")
    sar_agent = agents["SAR_Agent_2"]
    trxn_generation_agent = agents["Transaction_Generation_Agent"]
    trxn_generation_agent_w_tool = agents["Transaction_Generation_Agent_w_Tool"]


    ##########################################################
    # Instantiate Group Chat Manager Agent
    ##########################################################

    group_chat_manager_config = get_agent_config(agent_configs, agent_name = "Group_Chat_Manager")
    try:
        # Extract configuration parameters
        
        llm_config = group_chat_manager_config.get('llm_config')
        summary_method = group_chat_manager_config.get("summary_method")
        summary_prompt = group_chat_manager_config.get("summary_prompt")
        #logging.info("Loaded configuration for Group Chat and Group Chat Manager")
        #groupchat = GroupChat(agents = [trxn_generation_agent_w_tool1,trxn_generation_agent_w_tool2],messages=[],max_round=2,allow_repeat_speaker=False)
        #manager = GroupChatManager(groupchat=groupchat, llm_config = llm_config)
        #logging.info("Instantiated GroupChat and GroupChat Manager")

    except Exception as e:
        logging.error("Failed to instantiate Group Chat Manager")
        raise

    logging.info(f"Input is of type: {type(input)}")
    sub_narratives = split_dictionary_into_subnarratives(input)
    logging.info(f"No of sub-narratives created: {len(sub_narratives)}")

    ### Call the agentic workflow repeatedly for each transaction set and concatenate the results   ###
    trxn_df_list = [] #List of generated trxn dataframes
    for i,sub_narrative in enumerate(sub_narratives): 
        #Convert Dict to string to pass to LLM
        input_text = json.dumps(sub_narrative,indent =2)
        logging.info(f"Loaded configuration for Group Chat and Group Chat Manager for sub narratuve {i}")
        groupchat = GroupChat(agents = [trxn_generation_agent,trxn_generation_agent_w_tool],messages=[],max_round=2,allow_repeat_speaker=False)
        manager = GroupChatManager(groupchat=groupchat, llm_config = llm_config)
        logging.info(f"Instantiated GroupChat and GroupChat Manager for sub narratuve {i}")
        # Use DiskCache as cache
        with Cache.disk() as cache:
            chat_results = sar_agent.initiate_chat(
                        manager,
                        message = input_text,
                        summary_method= summary_method,
                        summary_args = {
                            "summary_prompt":summary_prompt
                        } )
        results = chat_results.summary
        cleaned_results = results.strip("```python\n").strip("```")
        # Convert to dictionary
        results_dict = ast.literal_eval(cleaned_results)
        logging.info(f"Results from  Transaction Generation Agent converted to a dictionary for Sub narrative {i+1}")
        trxn_df = convert_trxn_dict_to_df(i+1,results_dict)
        trxn_df_list.append(trxn_df)

    # Concatenate to get a single dataframe with trxns for all trxns sets
    if trxn_df_list:
        trxns_df_final = pd.concat(trxn_df_list)
        trxns_df_final["Transaction_ID"] = range(1, len(trxns_df_final) + 1)
    else:
        logging.warning("No transaction dataframes were generated. Returning empty dataframe.")
        trxns_df_final = pd.DataFrame()

    #Write output file for later reuse
    output_file = generate_dynamic_output_file_name(filename="trxn_metrics",output_file_type="csv",
                                                    output_folder="./data/output")
    write_data_to_file(trxns_df_final,output_file)

    logger.info("Finished run_agentic_workflow2")
    return  trxns_df_final
