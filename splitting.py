import ast


globalFuncs = []
def parse_functions(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        source_code = file.read()

    tree = ast.parse(source_code)
    functions = []
    function_map = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_map[node.name] = node

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            end_line = node.end_lineno
            function_name = node.name
            function_code = '\n'.join(source_code.splitlines()[start_line:end_line])
            callees = {}
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                    callee_name = sub_node.func.id
                    if callee_name in function_map:
                        callee_node = function_map[callee_name]
                        callee_start_line = callee_node.lineno - 1
                        callee_end_line = callee_node.end_lineno
                        callee_code = '\n'.join(source_code.splitlines()[callee_start_line:callee_end_line])
                        callees[callee_name] = callee_code
            functions.append((function_name, start_line, end_line, function_code, callees))

    return functions, source_code.splitlines()

def insert_comment(filename, comment, insert_after_line):
    with open(filename, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line.rstrip())
        if i == insert_after_line - 1:
            new_lines.append(comment)

    with open(filename, 'w') as file:
        file.write('\n'.join(new_lines) + '\n')



