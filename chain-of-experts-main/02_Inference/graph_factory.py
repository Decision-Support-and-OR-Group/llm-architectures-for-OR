from typing import List, Literal, Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph

import agents as agents

MAX_RETRIES = 2

class AgentState(TypedDict, total=False):
    problem_description: str
    problem_model: str
    code: str
    execution_result: str
    issues_after_review: str
    status: Literal["success", "error"]
    messages: Annotated[List[BaseMessage], add_messages]
    number_of_revisions: int


def router_3_agent(state):
    if state["status"] == "success":
        return "__end__"

    if state.get("number_of_revisions", 0) > MAX_RETRIES:
        print(f"--- [Router] Reached max level of retries {MAX_RETRIES} for given problem. Interrupting.. ---")
        return "__end__"

    return "coder"


def router_2_agent(state):
    if state["status"] == "success":
        return "__end__"

    if state.get("number_of_revisions", 0) > MAX_RETRIES:
        print(f"--- [Router] Reached max level of retries {MAX_RETRIES} for given problem. Interrupting.. ---")
        return "__end__"

    return "validator"


def create_architecture(architecture_name, llm, tool_executor):
    print(f"--- [Factory]: Building {architecture_name} architecture")

    workflow = StateGraph(AgentState)

    if architecture_name == "1-agent":
        monolith_node = agents.create_monolith_agent(llm, tool_executor)

        workflow.add_node("monolith", monolith_node)
        workflow.set_entry_point("monolith")

        workflow.add_conditional_edges(
            "monolith",
            lambda x: "monolith" if x["status"] == "error" and x.get("number_of_revisions", 0) <= 3 else "__end__"
        )

    if architecture_name == "2-agent":
        modeler_agent = agents.create_modeler_agent(llm)
        validator_agent = agents.create_executor_validator_agent(llm, tool_executor)

        workflow.add_node("modeler", modeler_agent)
        workflow.add_node("validator", validator_agent)

        workflow.set_entry_point("modeler")
        workflow.add_edge("modeler", "validator")
        workflow.add_conditional_edges("validator", router_2_agent)

    elif architecture_name == "3-agent":
        modeler_node = agents.create_modeler_agent(llm)
        coder_node = agents.create_coder_agent(llm)
        reviewer_node = agents.create_reviewer_agent(llm, tool_executor)

        workflow.add_node("modeler", modeler_node)
        workflow.add_node("coder", coder_node)
        workflow.add_node("reviewer", reviewer_node)

        workflow.set_entry_point("modeler")
        workflow.add_edge("modeler", "coder")
        workflow.add_edge("coder", "reviewer")
        workflow.add_conditional_edges("reviewer", router_3_agent)

    elif architecture_name == "4-agent":
        planner_node = agents.create_planner_agent(llm)
        modeler_node = agents.create_modeler_agent(llm)
        coder_node = agents.create_coder_agent(llm)
        reviewer_node = agents.create_reviewer_agent(llm, tool_executor)

        workflow.add_node("planner", planner_node)
        workflow.add_node("modeler", modeler_node)
        workflow.add_node("coder", coder_node)
        workflow.add_node("reviewer", reviewer_node)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "modeler")
        workflow.add_edge("modeler", "coder")
        workflow.add_edge("coder", "reviewer")
        workflow.add_conditional_edges("reviewer", router_3_agent)

    return workflow.compile()
