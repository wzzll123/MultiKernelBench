# MultiKernelBench

A benchmark for evaluating LLMs' ability to generate kernels for various platform. Now supporting CUDA and triton kernels for GPUs, Ascendc and TileLang kernels for NPUs, pallas kernels for TPUs and SYCL kernels for Intel GPUs.

## Latest News
- **08/10/2025** – Added **TileLang-Ascend backend** support for **Ascend NPUs**.  
- **12/08/2025** – Added **SYCL backend** support for **Intel GPUs** – thanks to **NinaWie** for the contribution!  
- **18/07/2025** – 🎉 Announced the open-source release of **MultiKernelBench**, a **multi-platform benchmark for kernel generation**, now publicly available!


## Quick start

### Set up
```bash
conda create --name multi-kernel-bench python=3.10
conda activate multi-kernel-bench
pip install -r requirements.txt

# For NPU users:
pip install torch-npu==2.1.0.post12

# For Intel GPU (torch xpu) users:
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/xpu
```
You can rent GPU or NPU resources from online platforms such as [autodl](https://www.autodl.com/home). For TPU resources, you can use services like [Google Colab](https://colab.research.google.com/)

### Config
Set configurations in config.py, including temperature and top_p for LLM. For CUDA-based evaluation, set arch_list. For Ascendc evaluation, set ascendc_device = ai_core-<soc_version>.

### Set API keys for LLM
```
export DEEPSEEK_API_KEY=<your deepseek api key>
export DASHSCOPE_API_KEY=<your aliyun api key>
export OPEN_ROUNTER_KEY=<your openrouter api key>
```

### Generate kernels using LLM and write them to output
```bash
python generate_and_write.py --model deepseek-chat --language ascendc --strategy add_shot --categories activation
```
Generated code is saved in ```output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}```.

#### Available Arguments

- `--runs`: Number of runs (default: `1`)
- `--model`: Model name (default: `deepseek-chat`)
- `--language`: Language used (default: `cuda`)
- `--strategy`: Prompt strategy type (default: `add_shot`)
- `--categories`: Space-separated list of categories (default: `activation`)  
  Use `all` to include all categories.

### Evalutation
```bash
python evaluation.py --model deepseek-chat --language ascendc --strategy add_shot --categories activation
```
Evaluation result is saved in ```output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}/result_{category}.json```.

## Adding a Prompting Strategy for a New or Existing Language

To add a custom prompting strategy, follow these steps:
1. **Create a Python file:**  
   Add a new file under `prompt_generators/` named as:  
   `prompt_generators/{language}_{strategy_name}.py`  

2. **Create a New Strategy Class**

   - Inherit from `BasePromptStrategy`.
   - Implement the `generate(self, op)` method.

2. **Register the Strategy**

   Use the `@register_prompt(language, strategy_name)` decorator with the desired language and strategy name.
## Adding a New Backend

To integrate a new backend, follow these steps:

1. **Create a New Python File**

   Add a new file under `backends/` named as:  
   `backends/{backend_name}.py`

2. **Create a Backend Class**

   - Inherit from `Backend`.
   - Implement all required methods:
     - `get_device()`
     - `get_hardware_name()`
     - `compile(generated_code, op)`
     - `correctness_execution(ref_src)`
     - `time_execution()`
     - `cleanup()` (optional)

3. **Register the Backend**

   Use the `@register_backend(name)` decorator with your backend's unique name.
## Credits

This project uses code from [KernelBench](https://github.com/ScalingIntelligence/KernelBench), licensed under the MIT License.

