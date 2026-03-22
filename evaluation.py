import os
import json
from dataset import dataset, category2exampleop
import subprocess
import tempfile
from config import temperature, top_p
import argparse
def eval_all(out_dir, language, categories, op_tested=dataset.keys()):
    result = {}
    if categories == ['all']:
        output_file = os.path.join(out_dir,'result.json')
    else:
        output_file = os.path.join(out_dir, f'result_{"_".join(categories)}.json')
    if os.path.exists(output_file):
        print(f"[INFO] Already evaluated, please see {output_file}")
        return
    for op in op_tested:
        print(f"[INFO] Evaluating op {op}")
        with open(os.path.join(out_dir, f'{op}.txt'), 'r') as saved_log:
            response_txt = saved_log.read()
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tf_input, \
            tempfile.NamedTemporaryFile(mode='r', delete=True) as tf_output:

            tf_input.write(response_txt)
            tf_input.flush()
            try:
                captured_text = subprocess.run(
                    ['python3', 'eval_single_runner.py', tf_input.name, op, language, tf_output.name],
                    check=True,
                    capture_output=True,     # capture stdout and stderr
                    text=True,               # decode bytes to str
                    timeout=180
                )
                result_item = json.load(tf_output)
                if not result_item['compiled']:
                    detailed_compiler_error = (
                        '\n[STDOUT]\n' + captured_text.stdout + '\n[STDERR]\n' + captured_text.stderr
                    )
                    result_item['compile_info'] += detailed_compiler_error

            except subprocess.CalledProcessError as e:
                if 'FileNotFoundError' in e.stderr:
                    print("[FAIL] FileNotFoundError - Possibly due to incorrect 'project_root_path' setting in config.py")
                    break
                elif e.returncode == -11:
                    print("[FAIL] Segmentation fault" )
                    seg_result = {'compiled': True, 'correctness': False, 'performance': None, 'correctness_info': 'Segmentation fault'} 
                    result[op] = seg_result
                    continue
                else:
                    print("[FAIL] unknown error, please report or fix bug")
                    print('[STDOUT]')
                    print(e.stdout)
                    print('[STDERR]')
                    print(e.stderr)
                    unknown_result = {'compiled': True, 'correctness': False, 'performance': None, 'correctness_info': 'Unknown fault'} 
                    result[op] = unknown_result
                    continue
            except subprocess.TimeoutExpired as e:
                print("[FAIL] run timeout")
                print('[STDOUT]')
                print(e.stdout)
                print('[STDERR]')
                print(e.stderr)
                time_result = {'compiled': True, 'correctness': False, 'performance': None, 'correctness_info': 'Timeout fault'} 
                result[op] = time_result
                continue
            result[op] = result_item
            print(f'[INFO] {result_item}')
        
    with open(output_file, 'w') as f:
        print(f"[INFO] Evaluated succesfully, write into {output_file}") 
        json.dump(result, f, indent=2)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--model', type=str, default='deepseek-chat', help='Model name')
    parser.add_argument('--language', type=str, default='cuda', help='Programming language')
    parser.add_argument('--strategy', type=str, default='add_shot', help='Strategy type.')
    parser.add_argument('--categories', nargs='+', default=['activation'], help='List of categories.')

    args = parser.parse_args()

    runs = args.runs
    model = args.model
    language = args.language
    strategy = args.strategy
    categories = args.categories

    print(f"Runs: {runs}")
    print(f"Model: {model}")
    print(f"Language: {language}")
    print(f"Strategy: {strategy}")
    print(f"Categories: {categories}")

    op_tested = list(dataset.keys())
    if categories != ['all']:
        op_tested = [op for op in op_tested if dataset[op]['category'] in categories]

    if '/' in model:
        # processing openrouter model
        model_name = model.split('/')[1]
    else:
        model_name = model

    for run in range(runs):
        out_dir = f'output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}'
        eval_all(out_dir, language, categories, op_tested)  
