import openai
from utils import get_agent_config
import json
from typing import List
import logging
# Configure logging
logger = logging.getLogger(__name__)

def choose_agent(narrative: str,agent_configs:dict) -> str:
    """
     Router that returns exactly one of:
      - "Simple_Agent"
      - "Tool_Agent"
    based on the complexity of the narrative
    """

    router_config = get_agent_config(agent_configs, agent_name = "Router")
    model= router_config.get('model')
    instructions = router_config.get("system_message")
    resp = openai.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role":"system","content":(
                instructions
            )},
            {"role":"user","content": narrative},
        ],
        max_tokens=5,
    )
    return resp.choices[0].message.content.strip()





def make_router_schema(agents: dict) -> dict:
    """
    Build a function schema whose `agent` property enum
    is exactly the list of agent_names.
    """
    agent_names = list(agents.keys())
    return {
        "name": "choose_agent",
        "description": "Pick exactly one agent from the provided list",
        "parameters": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": agent_names
                }
            },
            "required": ["agent"],
            "additionalProperties": False
        }
    }

def route_and_execute(agents:List,narrative:dict):
    """
    Function to take the narrative to be synthesized, pass it to the router agent, get the recommended agent 
    and execute it to generate transactions
    """

     # Get just the narrative to pass to the router
    narrative_ = narrative["Narratives"]

    #Convert to text to pass to LLM
    narrative_text = json.dumps(narrative_,indent =2)

    #Get recommended agent from router
    message = [{"role":"user","content":narrative_text}]
    router_agent = agents["Router_Agent"]
    chosen_agent_name = router_agent.generate_reply(message)
    logger.info (f"Agent chosen is: {chosen_agent_name}")

    chosen_agent = agents[chosen_agent_name]

    #Convert full narrative back to a string
    full_narrative = json.dumps(narrative,indent =2)
    message = [{"role":"user","content":full_narrative}]

    #Get the required response from the chosen agent
    trxns = chosen_agent.generate_reply(message)

    #If the agent returns a dictionary just return it
    if isinstance(trxns,dict):
        return trxns
    else:
        #If it returns a JSON convert to a dictionary and return it
        try:
            trxns_ = json.loads(trxns)
            return trxns_
        except json.JSONDecodeError:
            print("Not a valid JSON")
