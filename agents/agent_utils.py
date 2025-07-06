import openai
from utils import get_agent_config
import json
from typing import List
import logging
# Configure logging
logger = logging.getLogger(__name__)


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

def route(agents: dict, narrative: dict) -> str:
    narrative_ = narrative["Narratives"]
    narrative_text = json.dumps(narrative_, indent=2)
    message = [{"role": "user", "content": narrative_text}]
    router_agent = agents["Router_Agent"]
    chosen_agent_name = router_agent.generate_reply(message)
    logger.info(f"Agent chosen is: {chosen_agent_name}")
    return chosen_agent_name

def route_and_execute(agents:dict,narrative:dict):
    """
    Function to take the narrative to be synthesized, pass it to the router agent, get the recommended agent 
    and execute it to generate transactions
    """

    # Determine which agent to use
    chosen_agent_name = route(agents, narrative)

    # Prepare and send full narrative to the chosen agent
    full_narrative = json.dumps(narrative, indent=2)
    message = [{"role": "user", "content": full_narrative}]
    chosen_agent = agents[chosen_agent_name]
    trxns = chosen_agent.generate_reply(message)

    # Parse and return the transactions dictionary
    if isinstance(trxns, dict):
        return trxns
    try:
        return json.loads(trxns)
    except json.JSONDecodeError:
        print("Not a valid JSON")
        return {}
