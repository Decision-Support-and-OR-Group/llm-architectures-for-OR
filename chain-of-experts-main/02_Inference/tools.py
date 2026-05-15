import docker
from langchain_core.tools import tool


@tool
def safe_python_executor(code: str) -> str:
    """
    Execute untrusted Python code in a custom Docker image
    that already has optimization libraries installed.
    """
    print("--- [Tool] Executing code in Docker ---")

    try:
        client = docker.from_env()
        image = "pbad-python:latest"

        command = ["python", "-c", code]

        container_output = client.containers.run(
            image=image,
            command=command,
            remove=True,
            detach=False,
            stdout=True,
            stderr=True,
            network_disabled=True,
        )

        stdout = container_output.decode("utf-8")
        print(stdout)
        return f"Execution result: {stdout}"

    except Exception as e:
        print(f"--- [Tool] Exception in executing code in Docker: {e} ---")
        return f"Execution failed: {e!r}"


all_tools = [safe_python_executor]
