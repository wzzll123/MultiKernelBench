import os
from config import project_root_path
from dataset import dataset
from utils.utils import read_file, underscore_to_pascalcase

template_statement="""You write custom {} kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom {} kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
template_instruction="""
Optimize the architecture named Model with custom {} operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

template_example_intro='''
Here's an example to show you the syntax of inline embedding custom {} operators in torch: The example given architecture is:
'''

template_new_arch_intro='''
The example new arch with custom {} kernels looks like this:
'''

ASCENDC_PROBLEM_STATEMENT = 'You are an expert in writing custom AscendC kernels to optimize PyTorch architectures by replacing specific operators for performance gains.\n'
ASCENDC_PROBLEM_INSTRUCTION='''
Your task: Replace relevant PyTorch operators in the architecture named Model with custom AscendC kernels. Generate an optimized version named ModelNew, including the six Python strings listed above. Just output the code, no other text, and NO testing code!\n
'''

def read_relavant_files(language, op, example):
    category = dataset[op]['category']
    example_arch_path = os.path.join(
        project_root_path, f"prompts/cuda_model_{example}.py"
    )
    example_new_arch_path = os.path.join(
        project_root_path, f"prompts/{language}_new_model_{example}.py"
    )
    new_arch_path = os.path.join(
        project_root_path, f"reference/{category}/{op}.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )
    if not os.path.exists(new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {new_arch_path}"
        )
    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)
    arch = read_file(new_arch_path)
    return arch, example_arch, example_new_arch

def generate_template(arc_src, example_arch_src, example_new_arch_src, language):
    prompt = template_statement.format(language, language)

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        {template_example_intro.format(language)} \n
        ``` \n
        {example_arch_src}
        ``` \n
        {template_new_arch_intro.format(language)} 
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """
    prompt += template_instruction.format(language)
    return prompt    

def ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, example_op):
        # add custom to name to prevent conficts with existing operators
        op = op + '_custom'
        example_op = example_op + '_custom'
        prompt = ASCENDC_PROBLEM_STATEMENT

        if example_arch_src != "" and example_new_arch_src != "":
            prompt += f"""
    Here is an example to illustrate the expected transformation using custom AscendC operators. **Original architecture with kernel name `{example_op}`:**\n
    ```python \n
    {example_arch_src}
    ``` \n
    Transformed version using custom AscendC kernels:
    This transformation includes six embedded Python strings: `project_json_src`, `host_tiling_src`, `host_operator_src`, `kernel_src`, `python_bind_src` and `model_src`.
    The kernel function name in `kernel_src` must exactly match the provided kernel name. The operator definition in `project_json_src` and `host_operator_src` should also correspond to the kernel name, but follow PascalCase naming: 
    ```python
    {example_new_arch_src}
    ``` \n
    """

        prompt += f"""
    Now, you are given the following architecture with kernel name {op}(PascalCase: {underscore_to_pascalcase(op)}): \n
    ```python
    {arc_src}
    ```
        """
        prompt += ASCENDC_PROBLEM_INSTRUCTION
        return prompt


ASCENDC_DIRECT_LAUNCH_PROBLEM_STATEMENT = """
You are an expert in writing Ascend C direct-launch kernels and PyTorch pybind modules.
Your task is to optimize the provided PyTorch architecture named Model by returning a single JSON submission.
The JSON must embed all code directly in `sources[].content`.
"""

ASCENDC_DIRECT_LAUNCH_INSTRUCTION = """
Return exactly one JSON object and no other text. Do not wrap it in Markdown.

Required JSON shape:
- `name`: solution name.
- `author`: author or model name.
- `problem`: the benchmark problem name.
- `format_version`: use "0.1".
- `description`: short description.
- `entry.model`: must be "ModelNew.py::ModelNew".
- `build.type`: must be "ascendc_direct_launch".
- `sources`: a list of files. Each item has `path` and `content`.

Source layout guidance:
- Include `ModelNew.py`, defining class `ModelNew(nn.Module)` with the same constructor and forward signature as Model. It may import `benchmark_ops` from `kernel/build`.
- Include one or more Ascend C kernel source/header files under `kernel/`.
- Include a pybind source, normally `kernel/pybind11.cpp`, that exposes the generated operators through `PYBIND11_MODULE(benchmark_ops, m)`.
- You may split helper headers, launch wrappers, tiling helpers, and multiple kernels into additional files under `kernel/` as needed.
- By default the backend compiles all `kernel/*.cpp` except `kernel/pybind11.cpp` as Ascend C kernel sources and uses `kernel/pybind11.cpp` as the binding source. If you need a different layout, specify it in `build.kernel_sources`, `build.binding_sources`, and `build.include_dirs`.

Constraints:
- The optimized model must be named `ModelNew`.
- Keep `forward` inputs and outputs compatible with the original Model.
- Do not include tests, explanations, Markdown fences, or extra files outside the JSON.
- Use NPU tensors and torch_npu stream APIs in the pybind layer.
- The pybind module name must be `benchmark_ops`.
"""


def read_direct_launch_relevant_files(op, example):
    category = dataset[op]['category']
    example_arch_path = os.path.join(
        project_root_path, f"prompts/cuda_model_{example}.py"
    )
    example_json_path = os.path.join(
        project_root_path, f"prompts/ascendc_direct_launch_model_{example}.json"
    )
    arch_path = os.path.join(
        project_root_path, f"reference/{category}/{op}.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(f"Example architecture file not found: {example_arch_path}")
    if not os.path.exists(example_json_path):
        raise FileNotFoundError(f"Example JSON file not found: {example_json_path}")
    if not os.path.exists(arch_path):
        raise FileNotFoundError(f"Architecture file not found: {arch_path}")

    return read_file(arch_path), read_file(example_arch_path), read_file(example_json_path)


def ascendc_direct_launch_template(arc_src, example_arch_src, example_json_src, op, example_op):
    prompt = ASCENDC_DIRECT_LAUNCH_PROBLEM_STATEMENT
    prompt += f"""

Here is a one-shot example for problem `{example_op}`.

Original architecture:
```python
{example_arch_src}
```

Expected JSON submission:
```json
{example_json_src}
```

Now optimize the following architecture for problem `{op}`:
```python
{arc_src}
```

{ASCENDC_DIRECT_LAUNCH_INSTRUCTION}
"""
    return prompt
