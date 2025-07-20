from autogen import GroupChat, GroupChatManager
from utils import load_agents_from_single_config , get_agent_config, split_dictionary_into_subnarratives,convert_dict_to_df,generate_dynamic_output_file_name , write_data_to_file, normalize_dict
from agents.agents import instantiate_all_base_agents, instantiate_agents_for_trxn_generation
from agents.agent_utils import  route_and_execute
from autogen import Cache
from typing import  Dict, Any, List
import ast
import logging
import json
import pandas as pd
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    logger.info("All agents instantiated successfully")

    #Dictionary to store other relevant config of each agent
    agent_config_dict = {}

    # Iterate over agents and  configure 
    for name, _ in agents.items():
        agent_config_dict[name] = {} # Dictonary to store other relevant configs for each agent
        agent_config = get_agent_config(agent_configs, name) #Get configuratins for the specific agent
        agent_config_dict[name]["summary_prompt"] = agent_config.get("summary_prompt")
        agent_config_dict[name]["summary_method"] = agent_config.get("summary_method")
        agent_config_dict[name]["max_turns"] = agent_config.get("max_turns",1)
        logger.info("  Additional configs for Agent %s read successfully",name)
        
    

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
    logger.info("Results from Entity Extraction Agent converted to a dictionary")
    results_dict1 = ast.literal_eval(cleaned_results1)
    logger.info("Results from Entity Resolution Agent converted to a dictionary")
    results_dict2 = ast.literal_eval(cleaned_results2)
    logger.info("Results from Narrative Extraction Agent converted to a dictionary")

    # Combine results from first agentic workflow
    results = {**results_dict0,**results_dict1,**results_dict2}
    logger.info("Results  from agents aggregated into a single dictionary")

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
    logger.info("All agents instantiated successfully")
    logger.info(f"Input is of type: {type(input)}")
    logger.info(f"Starting run_agentic_workflow2 with input keys={list(input.keys())}")
    sub_narratives = split_dictionary_into_subnarratives(input)
    logger.info(f"No of sub-narratives created: {len(sub_narratives)}")

    ### Call the agentic workflow repeatedly for each transaction set and concatenate the results   ###
    trxn_df_list = [] # List of generated trxn dataframes

    # Helper to process one sub-narrative
    def _process_sub_narrative(i: int, sub_narrative: Dict) -> pd.DataFrame:
        results_dict = route_and_execute(agents, sub_narrative)
        output_file = generate_dynamic_output_file_name(
            filename="trxns_dict",
            output_file_type="json",
            output_folder="./data/output"
        )
        write_data_to_file(results_dict, output_file)
        logger.info(f"Results from chosen Transaction Generation Agent for Sub narrative {i+1} has been generated")
        return convert_dict_to_df(i+1, results_dict)

    # Execute sub-narrative processing asynchronously
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_process_sub_narrative, i, sn): i
            for i, sn in enumerate(sub_narratives)
        }
        for future in as_completed(futures):
            trxn_df_list.append(future.result())

    #Columns that indicate a trxn has been duplicated under the same sub-narrative attributed to different account IDs
    DEDUP_COLS = [
    "Originator_Account_ID",
    "Originator_Name",
    "Beneficiary_Account_ID",
    "Beneficiary_Name",
    "Originator_Customer_ID",
    "Beneficiary_Customer_ID",
    "Trxn_Date",
    "Trxn_Amount",
    "Trxn_Channel",
]

    # Concatenate to get a single dataframe with trxns for all trxns sets
    if trxn_df_list:
        trxns_df_final = pd.concat(trxn_df_list)
        if len(trxn_df_list)>1:
            #Drop duplicate rows as same narrative could be attributes to two account ids (Originator and Beneficary)
            #Do this only if there is more than one trxn set
            before = len(trxns_df_final)
            trxns_df_final = (
                trxns_df_final
                .sort_values(DEDUP_COLS)  # deterministic
                .drop_duplicates(subset=DEDUP_COLS, keep="first")
            )
            removed = before - len(trxns_df_final)
            logger.info("Deduplicated %d rows using subset=%s", removed, DEDUP_COLS)
        trxns_df_final["Transaction_ID"] = range(1, len(trxns_df_final) + 1)
        #Replace missing Originator and Beneficary accunt IDS with None
        trxns_df_final["Originator_Account_ID"] = trxns_df_final["Originator_Account_ID"].fillna("")
        trxns_df_final["Beneficiary_Account_ID"] = trxns_df_final["Beneficiary_Account_ID"].fillna("")
    else:
        logger.warning("No transaction dataframes were generated. Returning empty dataframe.")
        trxns_df_final = pd.DataFrame()

    #Write output file for later reuse
    output_file = generate_dynamic_output_file_name(filename="trxns",output_file_type="csv",
                                                    output_folder="./data/output")
    write_data_to_file(trxns_df_final,output_file)

    logger.info("Finished run_agentic_workflow2")
    return  trxns_df_final
