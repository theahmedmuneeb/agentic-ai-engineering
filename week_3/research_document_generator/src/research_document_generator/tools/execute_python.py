from daytona import (
    Daytona,
    Image,
    CreateSandboxFromImageParams,
)
from pathlib import Path
from crewai.tools import tool


@tool("Execute Python Code")
async def execute_python_code(code: str, output_file: str | None = None) -> str:
    """
    Execute the given Python code.

    If output_file is provided, the generated code must save the final file
    using the exact filename specified by output_file in the current working
    directory. Do not change the filename or save it elsewhere.

    Args:
        code: The Python code to execute.
        output_file: Expected output filename (e.g. "output.pdf")

    Returns:
        The execution output. If a file is generated, returns the local path
        to the downloaded file.
    """

    daytona = Daytona()

    print("Executing Python code in a sandbox...")

    image = (
        Image.debian_slim("3.12")
        .pip_install(["reportlab", "python-pptx", "matplotlib", "Pillow", "requests"])
        .workdir("/workspace")
    )

    sandbox = daytona.create(CreateSandboxFromImageParams(image=image))
    try:
        result = sandbox.process.code_run(code)

        if output_file:
            local_dir = Path("output")
            local_dir.mkdir(exist_ok=True)

            local_path = local_dir / output_file

            try:
                sandbox.fs.download_file(output_file, str(local_path))
                return str(local_path)
            except Exception:
                return str(result)

        return result
    finally:
        sandbox.delete()
