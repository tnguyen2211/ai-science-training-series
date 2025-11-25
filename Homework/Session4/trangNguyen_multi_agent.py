from typing import TypedDict, Annotated

from langgraph.graph import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from inference_auth_token import get_access_token

#from tools import molecule_name_to_smiles, smiles_to_coordinate_file, run_mace_calculation

# Import my custom tools
from trangNguyen_tools import (
    molecule_name_to_smiles, 
    smiles_to_coordinate_file, 
    run_mace_calculation,
    calculate_molecular_weight
)
# ============================================================
# 1. State definition
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 2. Routing logic
# ============================================================
def route_tools(state: State):
    """Route to the 'tools' node if the last message has tool calls; otherwise, route to 'done'.

    Parameters
    ----------
    state : State
        The current state containing messages and remaining steps

    Returns
    -------
    str
        Either 'tools' or 'done' based on the state conditions
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "done"


# ============================================================
# 3. LLM node: the "agent"
# ============================================================
# Agent 1: Simulation Specialist (Uses MACE tools)
def simulation_agent(state: State, llm: ChatOpenAI, tools: list):
    system_prompt = (
        "You are a computational chemist. "
        "Step 1: Identify the molecule SMILES. "
        "Step 2: Generate the XYZ coordinate file. "
        "Step 3: Run MACE optimization. "
        "Once MACE is finished, stop using tools."
    )
    messages = [{"role": "system", "content": system_prompt}] + state['messages']
    llm_with_tools = llm.bind_tools(tools)
    return {"messages": [llm_with_tools.invoke(messages)]}

# --- CHANGE 2: Upgrade Agent 2 to use the new tool ---
def analysis_agent(state: State, llm: ChatOpenAI, tools: list):
    system_prompt = (
        "You are a chemical data analyst. "
        "CRITICAL INSTRUCTION: You DO NOT know the molecular weight. "
        "You MUST call the 'calculate_molecular_weight' tool first. "
        "Stop generating text immediately after calling the tool. "
        "Wait for the tool output. "
        "ONLY after you receive the tool output, generate the final JSON."
        )
    messages = [{"role": "system", "content": system_prompt}] + state['messages']
    llm_with_tools = llm.bind_tools(tools)
    return {"messages": [llm_with_tools.invoke(messages)]}

# ============================================================
# 4. LLM / tools setup
# ============================================================
# Get token for your ALCF inference endpoint
access_token = get_access_token()

# Initialize the model hosted on the ALCF endpoint
llm = ChatOpenAI(
    model_name="openai/gpt-oss-20b",
    # model_name="Qwen/Qwen3-32B",
    api_key=access_token,
    base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
    temperature=0,
)

# Tool list that the LLM can call
# Define Tool Sets
sim_tools = [molecule_name_to_smiles, smiles_to_coordinate_file, run_mace_calculation]
analysis_tools = [calculate_molecular_weight]

# ============================================================
# 5. Build the graph
# ============================================================
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("simulation_agent", lambda state: simulation_agent(state, llm, sim_tools))
graph_builder.add_node("sim_tools_node", ToolNode(sim_tools))

graph_builder.add_node("analysis_agent", lambda state: analysis_agent(state, llm, analysis_tools))
graph_builder.add_node("analysis_tools_node", ToolNode(analysis_tools)) # --- CHANGE 3: Dedicated tool node

# Edges: Start -> Simulation
graph_builder.add_edge(START, "simulation_agent")

# Logic: Simulation Agent -> Sim Tools OR Analysis Agent
graph_builder.add_conditional_edges(
    "simulation_agent",
    route_tools,
    {"tools": "sim_tools_node", "done": "analysis_agent"}
)
# Sim Tools -> Back to Simulation Agent
graph_builder.add_edge("sim_tools_node", "simulation_agent")

# Logic: Analysis Agent -> Analysis Tools OR End
graph_builder.add_conditional_edges(
    "analysis_agent",
    route_tools,
    {"tools": "analysis_tools_node", "done": END}
)
# Analysis Tools -> Back to Analysis Agent
graph_builder.add_edge("analysis_tools_node", "analysis_agent")

graph = graph_builder.compile()

# ============================================================
# 6. Run / stream the graph
# ============================================================
if __name__ == "__main__":
    prompt = "Optimize Acetone using MACE. Calculate its molecular weight. Return results in JSON."
    print(f"--- User Input: {prompt} ---")
    
    for chunk in graph.stream({"messages": prompt}, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
