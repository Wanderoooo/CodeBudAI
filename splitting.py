import ast

def parse_functions(filename):
    with open(filename, 'r') as file:
        source_code = file.read()

    tree = ast.parse(source_code)
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            end_line = node.end_lineno
            functions.append((node.name, start_line, end_line))

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

# Usage
insert_comment('example.py', '# Custom text inserted', 5)
