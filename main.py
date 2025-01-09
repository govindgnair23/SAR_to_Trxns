# main.py
from autogen import ConversableAgent
from utils import load_agents_from_single_config, read_data , get_agent_config, write_dict_to_json_file
from agents.agents import instantiate_all_base_agents
from dotenv import load_dotenv
import os
import json
from autogen import Cache
from typing import  Dict, Any
import ast

def run_agentic_workflow(sar_text: str,config_file:str) -> Dict[str, Any]:
    '''
    Runs the full agentic workflow and returns results as a dictionary
    
    '''
    agent_configs = load_agents_from_single_config(config_file)
    agents = instantiate_all_base_agents(agent_configs)

    # Example usage: Iterate over agents and perform actions
    for name, agent in agents.items():
        print(f"Agent '{name}' instantiated successfully.")
    
    # Retrieve the Entity Extraction Agent's configuration
    ee_agent_config = get_agent_config(agent_configs, "Entity_Extraction_Agent")
    ee_summary_prompt = ee_agent_config.get("summary_prompt")

    # Use DiskCache as cache
   # with Cache.disk() as cache:

    chat_results = agents["SAR_Agent"].initiate_chats(
    [
        {
            "recipient":  agents["Entity_Extraction_Agent"],
            "message": sar_text,
            "max_turns": 1,
            "summary_method": "reflection_with_llm",
            "summary_args": {
            "summary_prompt" :  ee_summary_prompt                                                               
                },
        } ] )

    results = chat_results[0].summary

    
    cleaned_results = results.strip("```python\n").strip("```")
    # Convert to dictionary
    results_dict = ast.literal_eval(cleaned_results)

    
    # # Clean the JSON string if necessary
    # cleaned_results = results.replace('\n', '').strip()

    # # Parse the cleaned JSON string
    # try:
    #     results_dict = json.loads(cleaned_results)
    # except json.JSONDecodeError as e:
    #     print("Failed to decode JSON:", e)
    #     print("Response content:", repr(cleaned_results))


    return results_dict

def main():
    load_dotenv()
    api_key = os.getenv("OPEN_API_KEY")
    config_file = 'configs/agents_config.yaml'  # Update this path as needed
    
    # train_sars = read_data(train=True)
    

    # For now pick one SAR narrative  
    #message = train_sars[0]
    message = """ 
                John deposited $5000 in Cash into Acct #345723 at Bank of America. John sends $3000 to Jill's account at Chase.
                Jill deposited $3000 in Cash into her Acct at Chase Bank.John and Jill own a business Acme Inc that has a Business account, Account #98765. 
                John sends $2000 from Acct #345723 to Account #98765. Jill sends $1000 from her Acct at Chase Bank to Acct #98765.
            """




    results_dict = run_agentic_workflow(message,config_file)
    print(results_dict)
  
    # output_file_path = "./data/output/results0.json"
    # write_dict_to_json_file(results_dict, output_file_path)




if __name__ == "__main__":
    main()