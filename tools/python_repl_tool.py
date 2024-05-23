from langchain_experimental.tools import PythonAstREPLTool
from langchain_experimental.tools.python.tool import PythonInputs


def get_python_repl_tool(locals):
    return PythonAstREPLTool(
        locals=locals,
        name="python_repl",
        description="Runs code and returns the output of the final line in a dataframe object",
        args_schema=PythonInputs,
    )
