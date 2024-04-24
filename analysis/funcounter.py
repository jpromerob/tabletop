import inspect
import re

def list_functions(script_path):
    with open(script_path, 'r') as file:
        script_content = file.read()

    function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    functions = re.findall(function_pattern, script_content)

    return functions

# Example usage:
script_path = 'syn_analyzer.py'  # Replace with the path to your Python script
functions = list_functions(script_path)
print("Functions defined in the script:")
for func in functions:
    print(func)
