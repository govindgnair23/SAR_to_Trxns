from autogen import ConversableAgent 
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.function_utils import get_function_schema
import logging
import json
import ast
from utils import get_agent_config, get_config_list
from agents.tools import generate_transactions, generate_transactions_schema
from agents.agent_utils import   make_router_schema
import openai
from typing import Any, Callable, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)



class FunctionCallingAgent:
    def __init__(
        self,
        name: str,
        system_message: str,
        llm_config: Dict[str, Any],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        function_map: Optional[Dict[str, Callable[..., Any]]] = None,
        human_input_mode: str = "NEVER",
        code_execution_config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        """
        :param name:              Agent identifier
        :param system_message:    System prompt for the LLM
        :param llm_config:        Dict with keys like "model", "temperature", "max_tokens", "response_format", etc.
        :param function_schemas:  List of JSON-schema dicts describing available functions
        :param function_map:      Map from function name to Python callable
        :param human_input_mode:  UNUSED—machine-only agent
        :param code_execution_config: RESERVED for custom runtimes
        :param description:       Optional description
        """
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config
        self.function_schemas = function_schemas or []
        self.function_map = function_map or {}
        self.human_input_mode = human_input_mode
        self.code_execution_config = code_execution_config
        self.description = description

    def generate_reply(
        self,
        user_message: List[Dict[str,str]]
    ) -> Any:
        """
        1) Sends messages to the OpenAI Chat API, passing llm_config and function_schemas.
        2) If the response contains a function_call, executes the corresponding Python function.
        3) Returns the function’s result or the assistant’s content.
        """
        # Build the message list
        full_messages = [{"role": "system", "content": self.system_message}] + user_message #[{"role": "user", "content": user_message}] 

        # Assemble API parameters
        api_kwargs: Dict[str, Any] = {
            "model": self.llm_config["model"],
            "temperature": self.llm_config.get("temperature", 0),
            "messages": full_messages,
        }
        if "max_tokens" in self.llm_config:
            api_kwargs["max_tokens"] = self.llm_config["max_tokens"]
        if "response_format" in self.llm_config:
            api_kwargs["response_format"] = {
                "type": self.llm_config["response_format"]["type"]
            }
          # 3) If we have function_schemas, force a function_call using self.function_map’s key
        if self.function_schemas:
            api_kwargs["functions"] = self.function_schemas

            # Determine which function name to force:
            if not self.function_map:
                raise RuntimeError("No function_map provided, but function_schemas is non‐empty.")

            # If there’s exactly one function in the map, pick it.
            func_names = list(self.function_map.keys())
            if len(func_names) == 1:
                forced_name = func_names[0]
            else:
                # If you have multiple possible functions in function_map,
                # you might decide your own policy. For now, pick the first:
                forced_name = func_names[0]
                # Or raise an error:
                # raise RuntimeError(f"Expected exactly one function in function_map, but got: {func_names}")

            # Force that single function every time
            api_kwargs["function_call"] = {"name": forced_name}

        # Call the ChatCompletion API
        resp = openai.chat.completions.create(**api_kwargs)
        msg = resp.choices[0].message

        # Handle function_call if present
        if getattr(msg, "function_call", None):
            logger.info("Function call found")
            fname = msg.function_call.name
            args = json.loads(msg.function_call.arguments)
            if fname not in self.function_map:
                raise RuntimeError(f"Unregistered function: {fname}")
            return self.function_map[fname](**args)

        # Otherwise, return the assistant’s text
        logger.info("No valid Function call found")
        return msg.content

class RouterAgent(FunctionCallingAgent):
    def __init__(
        self,
        agents: dict,            # keys: agent‐name (str); values: agent object
        name: str,
        llm_config: dict,
        human_input_mode: str,
        code_execution_config: dict,
        description: str
    ):
        # **1) Set the attribute first, so build_router_prompt and make_router_schema can see it.**
        self.agents = agents

        # **2) Build the JSON schema now that self.agents exists.**
        router_schema = make_router_schema(self.agents)

        # **3) Generate the prompt text based on the dict.**
        prompt_text = self.build_router_prompt()

        # **4) Now call super().__init__ with the correct schema and system_message.**
        super().__init__(
            name=name,
            system_message=prompt_text,
            llm_config=llm_config,
            function_schemas=[router_schema],
            function_map={"choose_agent": lambda agent: agent},
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            description=description
        )

    def choose_agent(self, user_message: List[Dict[str, str]]) -> str:
        # No change here—generate_reply will trigger the function call.
        return self.generate_reply(user_message=user_message, function_call="choose_agent")

    def build_router_prompt(self):
        """
        Use the dict `self.agents` to construct a textual prompt:
          - header saying “You are a router who has to choose between N agents…”
          - list each agent’s .name and .description
          - give the LLM a bullet‐list of exactly which names it must return
        """
        num_agents = len(self.agents)
        header = (
            f"You are a router who has to choose between {num_agents} agents "
            f"whose skills are described below:\n\n"
        )

        # Agent descriptions: iterate over the dict’s values for .name and .description
        descs = "\n\n".join(
            f"{agent.name}: {agent.description}"
            for agent in self.agents.values()
        )

        # Choices: iterate over the dict’s keys (which are the same as agent.name, by design)
        choices = "\n".join(f"- {agent_name}" for agent_name in self.agents.keys())

        prompt = (
            header
            + descs
            + "\n\nDepending on which agent is most suitable, return exactly "
            "and only one of:\n"
            + choices
            + "\n\nDo NOT return anything else (no quotes, no extra text)."
        )
        return prompt

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

        logger.info(f"Loaded configuration for agent '{name}' ")

        if name == "Transaction_Generation_Agent_w_Tool":
            continue  #Need to replace with instantiation of GPTAsistantAgent

        agent = ConversableAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
        )

        logger.info(f"Instantiated '{name}' ")
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
        logger.info(f"Agent '{agent_name}' instantiated successfully.")
        return agent
    except Exception as e:
        logger.error(f"Failed to instantiate agent '{agent_name}': {e}")
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
    logger.info("Results returned by LLM")
    cleaned_results = results.strip("```python\n").strip("```")
    logger.info(f"Results cleaned")
    # Convert to dictionary
    results_dict = ast.literal_eval(cleaned_results)
    assert isinstance(results_dict,dict), "results is not a dictionary"
    logger.info(f"Results converted to a Python dictionary")
    print(results_dict)
    return results_dict





def instantiate_agents_for_trxn_generation(configs):
    '''
    Instantiates agents necessary for trxn generation from a narrative and other inputs. This includes the Simple Trxn generation agent
    and Trxn generation agent which uses a tool as well as the Router Agent.
    '''
    agents = {}

    
    ##########################################################
    # Agent 1: Instantiate Transaction Generation without tool
    ##########################################################

    trxn_generation_agent_config = get_agent_config(configs, agent_name = "Transaction_Generation_Agent")
    try:
        # Extract configuration parameters
        agent_name = trxn_generation_agent_config.get('name', 'Default_Agent_Name')
        system_message = trxn_generation_agent_config.get('system_message')
        llm_config = trxn_generation_agent_config.get('llm_config')
        human_input_mode = trxn_generation_agent_config.get('human_input_mode',"NEVER")
        code_execution_config = trxn_generation_agent_config.get("trxn_generation_agent_config", False)
        description = trxn_generation_agent_config.get("description","")
        # summary_method = trxn_generation_agent_config.get('summary_method')
        # summary_prompt = trxn_generation_agent_config.get('summary_prompt')

        logger.info(f"Loaded configuration for Trxn Generation Agent")

        # Instantiate the agent
        agent1 = ConversableAgent(
            name=agent_name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
            code_execution_config =code_execution_config,
            description= description
        )
        logger.info("Transaction_Generation_Agent instantiated successfully.")

        agents[agent_name] = agent1
    except Exception as e:
        logger.error("Failed to instantiate Transaction_Generation_Agent")
        raise


    

    ###########################################################
    # Agent 2: Instantiate Transaction Generation with tool use
    ###########################################################

    trxn_generation_agent_w_tool_config = get_agent_config(configs, agent_name = "Transaction_Generation_Agent_w_Tool")
    try:
        # Extract configuration parameters
        agent_name = trxn_generation_agent_w_tool_config.get('name', 'Default_Agent_Name')
        instructions = trxn_generation_agent_w_tool_config.get('instructions')
        llm_config = trxn_generation_agent_w_tool_config.get('llm_config')
        description =  trxn_generation_agent_w_tool_config.get("description","")
        

        logger.info(f"Loaded configuration for Trxn Generation Agent with Tool")

        # Instantiate the agent
        agent2 = FunctionCallingAgent(
            name=agent_name,
            system_message=instructions,
            llm_config=llm_config,
            description=description,
            function_schemas= [generate_transactions_schema],
            function_map={"generate_transactions": generate_transactions}
        )


        logger.info("Transaction_Generation_Agent with tool instantiated successfully.")
        agents[agent_name] = agent2
    except Exception as e:
        logger.error("Failed to instantiate Transaction_Generation_Agent with tool")
        raise

    ###########################################################
    # Agent 3: Router Agent for routing between agents
    ###########################################################
    router_agent_config = get_agent_config(configs, agent_name = "Router_Agent")
    try:
        # Extract configuration parameters
        agent_name = router_agent_config.get('name', 'Default_Agent_Name')
        llm_config = router_agent_config.get('llm_config')
        human_input_mode = router_agent_config.get('human_input_mode',"NEVER")
        description =  router_agent_config.get("description","")
        
        logger.info(f"Loaded configuration for  Router Agent")

        # Instantiate the agent
        agent3 = RouterAgent(
                 agents = agents, 
                 name = agent_name, 
                 llm_config = llm_config, 
                 human_input_mode= human_input_mode, 
                 code_execution_config = code_execution_config, 
                 description = description)


        logger.info("Router Agent instantiated successfully.")
        agents[agent_name] = agent3
    except Exception as e:
        logger.error("Failed to instantiate Router Agent")
        raise



    return agents

