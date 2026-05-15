from langchain_core.messages import SystemMessage, HumanMessage


def create_modeler_agent(llm):
    system_prompt = """
    You are an expert in modelling optimisation problems.
    Your task is to analyse a textual problem and transform it
    into a formal mathematical or algorithmic model.
    Clearly describe the objective, variables, constraints, and objective function.
    Suggest specific Python libraries (e.g., pulp, scipy.optimize, ortools)
    suitable for solving this problem.
    """

    def modeler_agent(state):
        print("--- [Agent: Modeler] ---")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["problem_description"]),
        ]

        response = llm.invoke(messages)
        response.name = "Modeler"
        return {
            "problem_model": response.content,
            "messages": [response],
        }

    return modeler_agent


def create_coder_agent(llm):
    system_prompt = """
    You are an expert Python programmer specialising in optimisation problems.
    You will receive a formal model of the problem (or code with an error to correct).
    Your task is to write a COMPLETE, RUN-READY Python script 
    that solves this problem.
    Important:
    1. The script must be self-sufficient.
    2. Import all required libraries.
    3. Initialise data DIRECTLY in code.
    4. Solve the problem and get the objective value.
    5. CRITICAL: At the very end, print the final objective value strictly in this format:
       print(f"FINAL_OBJ: {value}")
       (Do not add any other text in that line).
    6. Respond ONLY with Python code.
    """

    def coder_agent(state):
        print("--- [Agent: Coder] ---")

        if state.get("issues_after_review"):
            prompt = (
                f"The previous code was incorrect. Here are the comments: "
                f"{state['issues_after_review']}\n\n"
                f"Original model: {state['problem_model']}\n\n"
                f"Write the corrected code:"
            )
        else:
            prompt = state["problem_model"]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        response.name = "Coder"
        cleaned_code = (
            response.content.replace("```python", "").replace("```", "").strip()
        )

        return {
            "code": cleaned_code,
            "messages": [response],
        }

    return coder_agent


def create_reviewer_agent(llm, tool_executor):
    def reviewer_agent(state):
        print("--- [Agent: Reviewer] ---")
        code = state["code"]

        execution_result = tool_executor.invoke({"code": code})

        if "Execution failed:" in execution_result or "Traceback" in execution_result:
            critique_prompt = (
                "Code does not work. Error:\n"
                f"{execution_result}\n\n"
                "Tell the coder agent what needs to be fixed."
            )
            critique_msg = llm.invoke(critique_prompt)
            critique_msg.name = "Reviewer"
            return {
                "execution_result": execution_result,
                "issues_after_review": critique_msg.content,
                "status": "error",
                "messages": [critique_msg],
            }
        else:
            return {
                "execution_result": execution_result,
                "issues_after_review": None,
                "status": "success",
            }

    return reviewer_agent


def create_planner_agent(llm):
    system_prompt = """
    You are a Planner Agent. You will receive a complex textual optimisation problem.
    Your task is to create a strategic plan for modelling it.
    This plan will be forwarded directly to the Modeller as their main instruction. Analyse the problem and return text that will structure the Modeller's work.
    Your plan should include:
    1. Key optimisation goal: (e.g. Goal: Minimise total transport cost).
    2. Suggested model type: (e.g., Model: Linear Programming (LP)).
    3. List of main decision variables: (e.g., Variables: Amount of goods shipped from factory A to warehouse B).
    4. A list of key constraints to consider: (e.g., Constraints: Supply at factories, Demand at warehouses, Prohibited routes).
    Do not write a formal LP model (e.g., x + y <= 10). Focus only on the strategy and components.
    """

    def planner_agent(state):
        print("--- [Agent: Planner] ---")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["problem_description"]),
        ]

        plan = llm.invoke(messages)
        plan.name = "Planner"

        new_description = f"""
--- STRATEGIC PLAN FROM PLANNER ---
{plan.content}

--- ORIGINAL PROBLEM TEXT ---
{state['problem_description']}
"""

        return {
            "problem_description": new_description,
            "messages": [plan],
        }

    return planner_agent


def create_executor_validator_agent(llm, tool_executor):
    system_prompt = """
    You are an expert `Programmer-Validator` specialist.
    You will receive a problem specification OR (if a loop) an error from your previous attempt.

    Your task is to write a COMPLETE, RUN-READY Python script.

    Important:
    1. The script must be self-sufficient.
    2. Import all required libraries.
    3. Initialise data DIRECTLY in code.
    4. Solve the problem and get the objective value.
    5. CRITICAL: At the very end, print the final objective value strictly in this format:
       print(f"FINAL_OBJ: {value}")
       (Do not add any other text in that line).
    6. Respond ONLY with Python code.
    """

    def programmer_validator_agent(state):
        print("--- [Agent: Programmer-Validator] ---")

        previous_error = state.get("execution_result")
        if previous_error and ("Error" in previous_error or "Błąd" in previous_error):
            prompt_content = f"""
            Your previous script attempt failed with this error:
            {previous_error}
            
            Here is the original problem specification (for context):
            {state["problem_model"]}
            
            Please write a new, corrected, complete Python script.
            """
        else:
            prompt_content = state["problem_model"]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt_content),
        ]

        response = llm.invoke(messages)
        response.name = "Programmer-Validator"
        code = response.content.replace("```python", "").replace("```", "").strip()

        print("...Wykonywanie wygenerowanego kodu...")
        execution_result = tool_executor.invoke({"code": code})

        status = "success"
        if "Error" in execution_result or "Błąd" in execution_result or "Execution failed:" in execution_result:
            status = "error"

        return {
            "code": code,
            "execution_result": execution_result,
            "status": status,
            "messages": [response],
        }

    return programmer_validator_agent


def create_monolith_agent(llm, tool_executor):
    system_prompt = """
    You are an expert Operations Research Assistant.
    You will receive a natural language description of an optimization problem.

    Your task is to:
    1. Analyze the problem internally.
    2. Write a COMPLETE, RUN-READY Python script to solve it.

    Important Rules:
    1. Import all required libraries (e.g., `pulp`, `scipy`).
    2. Initialize data DIRECTLY in the code based on the text.
    3. CRITICAL: At the very end, print the final objective value strictly in this format:
       print(f"FINAL_OBJ: {value}")
    4. Respond ONLY with Python code.
    """

    def monolith_agent(state):
        print("--- [Agent: Monolith] ---")

        previous_error = state.get("execution_result")

        original_problem = state.get("problem_description") or state.get("problem_model", "")

        if not original_problem:
            raise KeyError(
                "Missing problem description in state. Expected 'problem_description' (1-agent) "
                "or 'problem_model' (multi-agent)."
            )

        if previous_error and any(
            marker in previous_error
            for marker in ("Error", "Błąd", "Traceback", "Execution failed:")
        ):
            prompt_content = f"""
    Your previous script attempt failed with this error:
    {previous_error}

    Here is the original problem description:
    {original_problem}

    Please write a new, corrected Python script.
    """
        else:
            prompt_content = original_problem

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt_content),
        ]

        response = llm.invoke(messages)
        response.name = "Monolith"

        code = (
            response.content
            .replace("```python", "")
            .replace("```", "")
            .strip()
        )

        execution_result = tool_executor.invoke({"code": code})

        status = "success"
        if any(marker in execution_result for marker in ("Error", "Błąd", "Traceback", "Execution failed:")):
            status = "error"

        return {
            "code": code,
            "execution_result": execution_result,
            "status": status,
            "number_of_revisions": state.get("number_of_revisions", 0) + (1 if status == "error" else 0),
            "messages": [response],
        }

    return monolith_agent