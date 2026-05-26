import ast
import json
import re


ALLOWED_TORCH_FUNCS = {
    "empty", "empty_like", "empty_strided",
    "zeros", "zeros_like",
    "ones", "ones_like",
    "full", "full_like",
    "tensor", "arange", "linspace", "as_tensor",
}

ALLOWED_TENSOR_METHODS = {
    "size", "shape", "stride", "numel", "dtype", "device", "dim",
    "is_contiguous", "data_ptr", "element_size", "storage_offset",
    "contiguous", "to", "view", "view_as", "reshape",
    "permute", "transpose", "expand", "expand_as",
    "flatten", "unflatten", "unsqueeze", "squeeze",
    "narrow", "clone", "detach", "t",
    "type", "float", "half", "bfloat16", "int", "long", "bool", "double",
    "cpu", "npu", "cuda", "item", "tolist",
    "requires_grad_", "zero_", "index_select", "is_npu", "is_cuda",
}

ALLOWED_BUILTIN_FUNCS = {
    "min", "max", "abs", "len", "range", "int", "float", "bool",
    "list", "tuple", "str", "type", "isinstance", "print",
    "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "hasattr", "getattr", "setattr",
}

FORBIDDEN_TENSOR_METHODS = {
    "sum", "mean", "max", "min", "prod", "cumsum", "cumprod",
    "argmax", "argmin", "var", "std",
    "matmul", "mm", "bmm", "addmm",
    "add", "sub", "mul", "div", "fmod", "remainder",
    "add_", "sub_", "mul_", "div_",
    "relu", "sigmoid", "tanh", "gelu", "silu", "elu", "leaky_relu",
    "relu_", "sigmoid_", "tanh_",
    "exp", "log", "log2", "log10", "sqrt", "pow", "abs",
    "sin", "cos", "clamp", "clamp_", "ceil", "floor", "round",
    "reciprocal", "neg", "sign",
    "softmax", "log_softmax",
    "norm", "layer_norm", "batch_norm", "group_norm",
    "conv1d", "conv2d", "conv3d", "conv_transpose2d", "linear",
    "dropout", "softplus", "hardtanh", "hardswish",
    "eq", "ne", "lt", "gt", "le", "ge", "where",
}

FORBIDDEN_NN_MODULES = {
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
    "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "Dropout", "Dropout1d", "Dropout2d", "Dropout3d",
    "ELU", "GELU", "GLU", "Hardtanh", "Hardsigmoid", "Hardswish",
    "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
    "LayerNorm", "LeakyReLU", "Linear", "LocalResponseNorm",
    "LogSigmoid", "LogSoftmax",
    "MaxPool1d", "MaxPool2d", "MaxPool3d",
    "PReLU", "ReLU", "ReLU6", "RMSNorm", "SELU", "SiLU",
    "Sigmoid", "Softmax", "Softmin", "Softplus", "Softshrink",
    "Tanh", "Tanhshrink", "Threshold",
}

PLACEHOLDER_IMPORT_NAMES = {"TORCH_EXTENSION_NAME"}

KERNEL_EXT_PATTERNS = [
    re.compile(r"_\w+_ext$"),
    re.compile(r"\w+_ext$"),
    re.compile(r"\w+_ascendc\w*$"),
    re.compile(r"benchmark_ops$"),
    re.compile(r"custom_ops_lib$"),
    re.compile(r"cann_bench$"),
    re.compile(r"_C$"),
]


def detect_python_kernel_cheating(generated_code):
    code = _extract_model_python_code(generated_code)
    if code is None:
        return False, None

    validation = _validate_kernel_python_code(code)
    if validation["valid"]:
        return False, validation
    return True, validation


def _extract_model_python_code(generated_code):
    stripped = generated_code.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json|python)?\s*(.*?)```", stripped, re.DOTALL)
        if match:
            stripped = match.group(1).strip()

    model_src = _extract_string_assignment(stripped, "model_src")
    if model_src is not None:
        return model_src

    try:
        spec = json.loads(stripped)
    except json.JSONDecodeError:
        spec = None

    if isinstance(spec, dict) and isinstance(spec.get("sources"), list):
        for source in spec["sources"]:
            path = source.get("path", "")
            if path.endswith("ModelNew.py") or path.endswith("model_new.py"):
                return source.get("content", "")
        for source in spec["sources"]:
            path = source.get("path", "")
            if path.endswith(".py"):
                return source.get("content", "")
        return None

    try:
        ast.parse(stripped)
    except SyntaxError:
        return None
    return stripped


def _extract_string_assignment(code, name):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue

        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                return value.value
    return None


def _resolve_call_name(node):
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return func.value.id, func.attr
        if isinstance(func.value, ast.Attribute) and isinstance(func.value.value, ast.Name):
            return f"{func.value.value.id}.{func.value.attr}", func.attr
    if isinstance(func, ast.Name):
        return None, func.id
    return None


def _is_ext_module_name(name):
    if name in PLACEHOLDER_IMPORT_NAMES:
        return False
    return any(pattern.match(name) for pattern in KERNEL_EXT_PATTERNS)


def _find_extension_imports(tree):
    extensions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                actual = alias.name
                used = alias.asname or alias.name
                is_placeholder = actual in PLACEHOLDER_IMPORT_NAMES
                if is_placeholder or _is_ext_module_name(actual):
                    extensions[used] = {
                        "name": actual,
                        "line": node.lineno,
                        "is_placeholder": is_placeholder,
                    }
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                actual = alias.name
                used = alias.asname or alias.name
                is_placeholder = actual in PLACEHOLDER_IMPORT_NAMES
                if is_placeholder or _is_ext_module_name(actual):
                    extensions[used] = {
                        "name": actual,
                        "line": node.lineno,
                        "is_placeholder": is_placeholder,
                    }
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
                continue
            resolved = _resolve_call_name(node.value)
            if resolved in {
                ("importlib", "import_module"),
                ("importlib.util", "module_from_spec"),
            }:
                extensions[target.id] = {
                    "name": target.id,
                    "line": node.lineno,
                    "is_placeholder": False,
                }
    return extensions


def _find_model_forward(tree):
    fallback = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in {"ModelNew", "Model"}:
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
                if node.name == "ModelNew":
                    return item, "ModelNew"
                fallback = (item, "Model")
    return fallback or (None, None)


def _find_class_defs(tree):
    return {
        node.name: node
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.ClassDef)
    }


def _class_methods(class_node):
    if class_node is None:
        return {}
    return {
        item.name: item
        for item in class_node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _is_forbidden_nn_module_call(call_node):
    resolved = _resolve_call_name(call_node)
    if resolved is None:
        return False
    qual, attr = resolved
    return qual in {"nn", "torch.nn"} and attr in FORBIDDEN_NN_MODULES


def _custom_class_constructed(call_node, class_defs):
    resolved = _resolve_call_name(call_node)
    if resolved is None:
        return None
    qual, attr = resolved
    if qual is None and attr in class_defs:
        return attr
    return None


def _self_module_attrs(class_node, class_defs):
    custom_attrs = {}
    forbidden_attrs = {}
    for method in _class_methods(class_node).values():
        for node in ast.walk(method):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            for target in node.targets:
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    continue
                custom_class = _custom_class_constructed(node.value, class_defs)
                if custom_class is not None:
                    custom_attrs[target.attr] = custom_class
                elif _is_forbidden_nn_module_call(node.value):
                    forbidden_attrs[target.attr] = _call_display_name(node.value)
    return custom_attrs, forbidden_attrs


def _find_module_functions(tree):
    return {
        node.name: node
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _call_display_name(call_node):
    resolved = _resolve_call_name(call_node)
    if resolved is None:
        return "<unknown>"
    qual, attr = resolved
    return f"{qual}.{attr}" if qual else attr


def _reachable_functions_from_forward(forward_node, root_class_name, module_functions, class_defs):
    if forward_node is None:
        return []

    reachable = []
    seen = set()
    stack = [("forward", forward_node, root_class_name)]
    while stack:
        name, node, class_name = stack.pop()
        key = (name, class_name)
        if key in seen:
            continue
        seen.add(key)
        reachable.append((name, node, class_name))

        class_node = class_defs.get(class_name)
        class_methods = _class_methods(class_node)
        custom_attrs, _ = _self_module_attrs(class_node, class_defs)

        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            resolved = _resolve_call_name(child)
            if resolved is None:
                continue
            qual, attr = resolved
            if qual is None and attr in module_functions:
                stack.append((attr, module_functions[attr], None))
            elif qual == "self" and attr in class_methods:
                stack.append((f"{class_name}.{attr}", class_methods[attr], class_name))
            elif qual == "self" and attr in custom_attrs:
                submodule_class = custom_attrs[attr]
                submodule_forward = _class_methods(class_defs.get(submodule_class)).get("forward")
                if submodule_forward is not None:
                    stack.append((f"{class_name}.{attr}.forward", submodule_forward, submodule_class))
    return reachable


def _forbidden_torch_ops(callables, ext_names, module_function_names=None, class_defs=None):
    violations = []
    if not callables:
        return violations

    module_function_names = module_function_names or set()
    class_defs = class_defs or {}
    arithmetic_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod, ast.MatMult)
    for function_name, function_node, class_name in callables:
        class_node = class_defs.get(class_name)
        class_methods = _class_methods(class_node)
        custom_attrs, forbidden_attrs = _self_module_attrs(class_node, class_defs)

        for node in ast.walk(function_node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, arithmetic_ops):
                violations.append({
                    "function": function_name,
                    "line": node.lineno,
                    "call": type(node.op).__name__,
                    "reason": "Tensor arithmetic in reachable model code must be implemented by a custom kernel",
                })
                continue
            if not isinstance(node, ast.Call):
                continue
            resolved = _resolve_call_name(node)
            if resolved is None:
                continue
            qual, attr = resolved
            if qual in ext_names:
                continue
            if qual == "torch" and attr not in ALLOWED_TORCH_FUNCS:
                violations.append({
                    "function": function_name,
                    "line": node.lineno,
                    "call": f"torch.{attr}",
                    "reason": "PyTorch compute op used in reachable model code",
                })
            elif qual in {"F", "functional", "torch.nn.functional", "nn.functional"}:
                violations.append({
                    "function": function_name,
                    "line": node.lineno,
                    "call": f"{qual}.{attr}",
                    "reason": "PyTorch functional compute op used in reachable model code",
                })
            elif qual is None and attr in ALLOWED_BUILTIN_FUNCS:
                continue
            elif qual is None and attr in module_function_names:
                continue
            elif qual == "self" and attr in class_methods:
                continue
            elif qual == "self" and attr in custom_attrs:
                continue
            elif qual == "self" and attr in forbidden_attrs:
                violations.append({
                    "function": function_name,
                    "line": node.lineno,
                    "call": f"self.{attr}(...)",
                    "reason": f"PyTorch nn.Module compute op used in reachable model code: {forbidden_attrs[attr]}",
                })
            elif attr in FORBIDDEN_TENSOR_METHODS and qual not in {"torch", "F", "functional", "torch.nn.functional", "nn.functional"}:
                violations.append({
                    "function": function_name,
                    "line": node.lineno,
                    "call": f"{qual}.{attr}()" if qual else f"{attr}()",
                    "reason": "Tensor compute method used in reachable model code",
                })
    return violations


def _scalar_for_loops(callables):
    violations = []
    if not callables:
        return violations
    for function_name, function_node, _ in callables:
        for node in ast.walk(function_node):
            if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
                continue
            if _resolve_call_name(node.iter) != (None, "range"):
                continue
            loop_var = node.target.id if isinstance(node.target, ast.Name) else ""
            if _loop_has_tensor_indexing(node, loop_var) and _loop_has_computation(node):
                violations.append({
                    "function": function_name,
                    "line": node.lineno,
                    "loop_var": loop_var,
                    "reason": "Python range loop mixes tensor indexing and computation",
                })
    return violations


def _loop_has_tensor_indexing(for_node, loop_var):
    if not loop_var:
        return False
    for child in ast.walk(for_node):
        if isinstance(child, ast.Subscript):
            for sub_node in ast.walk(child.slice):
                if isinstance(sub_node, ast.Name) and sub_node.id == loop_var:
                    return True
    return False


def _loop_has_computation(for_node):
    arithmetic_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod, ast.MatMult)
    binop_count = 0
    for child in ast.walk(for_node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, arithmetic_ops):
            binop_count += 1
        if isinstance(child, ast.Call):
            resolved = _resolve_call_name(child)
            if resolved is None:
                continue
            qual, attr = resolved
            if qual is None and attr in ALLOWED_BUILTIN_FUNCS:
                continue
            if qual == "torch" and attr not in ALLOWED_TORCH_FUNCS:
                return True
            if qual in {"F", "functional", "torch.nn.functional", "nn.functional"}:
                return True
            if attr in FORBIDDEN_TENSOR_METHODS:
                return True
    return binop_count >= 2


def _validate_kernel_python_code(code):
    result = {
        "valid": False,
        "regression_type": None,
        "suggestion": "",
        "checks": {
            "no_forbidden_torch_ops": {"passed": False, "violations": [], "error": None},
            "no_scalar_for_loops": {"passed": False, "violations": [], "error": None},
        },
    }
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        result["regression_type"] = 1
        result["checks"]["no_forbidden_torch_ops"]["error"] = f"SyntaxError: {error}"
        result["suggestion"] = "Generated Python code cannot be parsed for cheating detection."
        return result

    extensions = _find_extension_imports(tree)
    valid_ext_names = {name for name, details in extensions.items() if not details["is_placeholder"]}

    forward_node, class_name = _find_model_forward(tree)
    if forward_node is None:
        result["valid"] = True
        return result

    module_functions = _find_module_functions(tree)
    class_defs = _find_class_defs(tree)
    reachable_callables = _reachable_functions_from_forward(forward_node, class_name, module_functions, class_defs)
    violations = _forbidden_torch_ops(
        reachable_callables,
        valid_ext_names,
        set(module_functions),
        class_defs,
    )
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations
    if violations:
        result["regression_type"] = 3
        result["checks"]["no_forbidden_torch_ops"]["error"] = (
            f"Found {len(violations)} forbidden PyTorch/tensor compute operations in reachable model code"
        )
        result["suggestion"] = "Move all core computation into the custom kernel; forward() should only allocate/reshape/call kernels."
        return result
    result["checks"]["no_forbidden_torch_ops"]["passed"] = True

    loop_violations = _scalar_for_loops(reachable_callables)
    result["checks"]["no_scalar_for_loops"]["violations"] = loop_violations
    if loop_violations:
        result["regression_type"] = 4
        result["checks"]["no_scalar_for_loops"]["error"] = (
            f"Found {len(loop_violations)} scalar Python loops over tensors in forward()"
        )
        result["suggestion"] = "Replace scalar Python tensor loops with vectorized custom kernel work."
        return result
    result["checks"]["no_scalar_for_loops"]["passed"] = True

    result["valid"] = True
    return result
