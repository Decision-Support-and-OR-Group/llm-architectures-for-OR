import pandas as pd
import time, datetime
import os
import re
from langchain_openai import ChatOpenAI
from tools import safe_python_executor
from adapters.nl4opt_adapter import NL4OptAdapter
from graph_factory import create_architecture
from langchain_community.callbacks import get_openai_callback

TESTED_ARCHITECTURES = ["1-agent", "2-agent", "3-agent", "4-agent"]

TESTED_MODELS = {
    "gpt-5.1": ChatOpenAI(model="gpt-5.1-2025-11-13", temperature=0.1),
    #"gpt-5-mini": ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0.1),
}

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_output_dir = f"results_{timestamp}"
os.makedirs(base_output_dir, exist_ok=True)

# To można zmienić np na len, start od wybranego zadania i end
START = 0
END = 200

adapter = NL4OptAdapter()
results = []

def extract_objective_value(stdout):
    if not stdout:
        return None

    # Wersja obsługująca liczby typu: 123, 123.45, -0.5, 1e-3, -2.5E+10
    pattern = r"FINAL_OBJ:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    match = re.search(pattern, stdout)

    if not match:
        # Nic nie znaleziono – możesz zwrócić None albo rzucić wyjątek
        return None

    value_str = match.group(1).strip()
    if not value_str:
        # Na wszelki wypadek, gdyby regex jednak złapał pusty string
        raise ValueError(f"Objective value found but empty in stdout: {stdout!r}")

    try:
        return float(value_str)
    except ValueError:
        # Debug, gdyby format liczby był inny niż oczekiwany
        raise ValueError(f"Could not convert objective value {value_str!r} to float. Full stdout:\n{stdout}")


def format_history(messages_list):
    if not messages_list:
        return ""

    formatted_log = []
    for msg in messages_list:
        role = getattr(msg, "name", msg.type)
        content = msg.content.strip()
        formatted_log.append(f"[{role.upper()}]:\n{content}")

    return "\n\n---\n\n".join(formatted_log)


def save_log_to_file(base_dir, architecture, model_name, problem_id, content):
    # Ścieżka: results/.../logs/2-agent/gpt-4o/
    log_dir = os.path.join(base_dir, "logs", architecture, model_name)
    os.makedirs(log_dir, exist_ok=True)

    file_path = os.path.join(log_dir, f"{problem_id}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return file_path


for architecture in TESTED_ARCHITECTURES:
    for model_name, model_instance in TESTED_MODELS.items():
        print(f"\n[Test] Architecture: {architecture}, Model: {model_name}")
        agent_graph = create_architecture(architecture, model_instance, tool_executor=safe_python_executor)
        for i in range(START, END):
            problem = adapter[i]
            agent_obj_value = None

            problem_id = f"nl4opt_test_{i}"
            problem_text = problem["en_question"]
            ground_truth = problem["en_answer"]

            print(f"  Solving: {problem_id}...")
            initial_state = {
                "problem_description": problem_text,
                "number_of_revisions": 0
            }

            final_state = {}
            latency_sec = 0
            token_cost = 0
            total_cost_usd = 0

            with get_openai_callback() as callback:
                start_time = time.perf_counter()

                final_state = agent_graph.invoke(
                    initial_state,
                    {"recursion_limit": 10}
                )

                end_time = time.perf_counter()

                latency_sec = end_time - start_time
                prompt_tokens = callback.prompt_tokens
                completion_tokens = callback.completion_tokens
                total_tokens = callback.total_tokens
                total_cost_usd = callback.total_cost
                execution_output = final_state.get("execution_result", "")
                agent_obj_value = extract_objective_value(execution_output)

            raw_history = final_state.get("messages", [])
            formatted_history_str = format_history(raw_history)

            log_path = save_log_to_file(base_output_dir, architecture, model_name, problem_id, formatted_history_str)
            results.append({
                "architecture": architecture,
                "model": model_name,
                "problem_id": problem_id,
                "status": final_state.get("status", "unknown"),
                "final_code": final_state.get("code"),
                "execution_result": final_state.get("execution_result"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost_usd,
                "latency_sec": latency_sec,
                "agent_objective_value": agent_obj_value,
                "ground_truth_answer": ground_truth,
            })


excel_path = os.path.join(base_output_dir, "experiment_summary.xlsx")
df_results = pd.DataFrame(results)
df_results.to_excel(excel_path, index=False)
print(df_results.head())