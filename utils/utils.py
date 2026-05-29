import os
import config
import re
from openai import OpenAI
from dataset import dataset
import torch

def get_client(model):
    #client 
    if model.startswith('deepseek'):
        DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
        client = OpenAI(
            api_key=DEEPSEEK_KEY,
            base_url="https://api.deepseek.com",
            timeout=10000000,
            max_retries=3,
        )
    elif model.startswith('qwen'):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=10000000,
            max_retries=3,
        ) 
    else:
        api_key = os.environ.get("OPEN_ROUNTER_KEY")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return client

def get_ref_src_path(op):
    return os.path.join(config.ref_impl_base_path, dataset[op]['category'], f'{op}.py')


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code_block = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code_block.startswith(code_type):
                code = code_block[len(code_type) :].strip()

        return code, f'```{code_block}```'

def underscore_to_pascalcase(underscore_str):
    """
    Convert underscore-separated string to PascalCase.
    
    Args:
        underscore_str (str): Input string with underscores (e.g., "vector_add")
        
    Returns:
        str: PascalCase version (e.g., "VectorAdd")
    """
    if not underscore_str:  # Handle empty string
        return ""
    
    parts = underscore_str.split('_')
    # Capitalize the first letter of each part and join
    return ''.join(word.capitalize() for word in parts if word)
