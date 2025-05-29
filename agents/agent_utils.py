import openai
from utils import get_agent_config
import json


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

def generate_transactions_for(narrative: dict, agents: dict):
    narrative_ = json.dumps(narrative["Narratives"],indent =2)
    choice = choose_agent(narrative_)
    print(choice)
    print(choice=="Tool_Agent")
    msgs = [{"role": "user", "content": narrative}]
    if choice == "Simple_Agent":
        # simple agent just replies normally (no function)
        trxn_generation_agent =  agents["Trxn_Generation_Agent"]
        chat = trxn_generation_agent.generate_reply(messages=msgs,sender=None)
        return chat
    elif choice == "Tool_Agent":
        # tool agent calls your Python generate_transactions directly
        chat = trxn_generation_agent_gpt.generate_reply(
                    messages=msgs,
                    sender=None,
                    function_call="generate_transactions" 
        )
        return chat
    else:
        raise RuntimeError(f"Router chose unknown agent: {choice}")


def build_router_prompt(agents:list):
    """
    agents: list of objects with .name and .description attributes
    """
    num_agents = len(agents)

    # 1) Header
    header = (
        f"You are a router who has to choose between {num_agents} agents "
        f"whose skills are described below:\n\n"
    )

    # 2) Agent descriptions
    descs = "\n\n".join(f"{a.name}: {a.description}" for a in agents)

    # 3) Choice list
    choices = "\n".join(f"- {a.name}" for a in agents)

    # 4) Combine
    prompt = (
        header
        + descs
        + "\n\nDepending on which agent is most suitable, return exactly "
          "and only one of:\n"
        + choices
        + "\n\nDo NOT return anything else (no quotes, no extra text)."
    )
    return prompt

def make_router_schema(agent_names: list[str]) -> dict:
    """
    Build a function schema whose `agent` property enum
    is exactly the list of agent_names.
    """
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
