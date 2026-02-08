import os
import subprocess
import shutil
from config import op_engineer_dir, deploy_path, ascendc_device, project_root_path
from utils.utils import underscore_to_pascalcase


def _inject_kernel_include_paths(target_directory, include_paths):
    if not include_paths:
        return

    cmake_path = os.path.join(target_directory, "op_kernel", "CMakeLists.txt")
    if not os.path.exists(cmake_path):
        return

    with open(cmake_path, "r") as f:
        cmake_src = f.read()

    include_lines = []
    for include_path in include_paths:
        if not include_path:
            continue
        include_line = f"add_ops_compile_options(ALL OPTIONS -I{include_path})"
        if include_line not in cmake_src:
            include_lines.append(include_line)

    if not include_lines:
        return

    injected = "\n".join(include_lines)
    if "add_kernels_compile()" in cmake_src:
        cmake_src = cmake_src.replace("add_kernels_compile()", f"{injected}\nadd_kernels_compile()", 1)
    else:
        cmake_src = f"{cmake_src.rstrip()}\n{injected}\n"

    with open(cmake_path, "w") as f:
        f.write(cmake_src)



def ascend_compile(generated_code, op, context, extra_kernel_include_paths=None):
    op = op + '_custom'
    op_capital=underscore_to_pascalcase(op)
    target_directory=os.path.join(op_engineer_dir, op_capital)
    
    try:
        compile(generated_code, "<string>", "exec")
        exec(generated_code, context)  # For Python, use exec() (be careful with untrusted code)
    except Exception as e:
        raise Exception(f'Error in generated code {e}')
    
    # create ascendc project
    if os.path.exists(os.path.join(op_engineer_dir, op_capital)):
        print("[INFO] Operator project already exists, deleted")
        shutil.rmtree(os.path.join(op_engineer_dir, op_capital))
    with open(os.path.join(op_engineer_dir, f'{op}.json'), 'w') as f:
        f.write(context.get('project_json_src'))
    try:
        print("[INFO] Begin create operator project")
        os.chdir(op_engineer_dir)
        result = subprocess.run(["msopgen", 'gen', '-i', f'{op}.json', '-c', ascendc_device, '-lan', 'cpp', '-out', op_capital], check=True, capture_output=True, text=True)
        print("[INFO] Create operator project succeeded")
    except subprocess.CalledProcessError as e:
        print("[INFO] Create operator project failed!")
        # print("Exit Code:", e.returncode)
        print("Error Output:\n", e.stdout)
        print("Error Output:\n", e.stderr)
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}'
        raise Exception(feedback) 

    # write code to specific location
    with open(os.path.join(target_directory, 'op_host', f'{op}_tiling.h'), 'w') as f:
        f.write(context.get('host_tiling_src'))

    with open(os.path.join(target_directory, 'op_host', f'{op}.cpp'), 'w') as f:
        f.write(context.get('host_operator_src'))

    _inject_kernel_include_paths(target_directory, extra_kernel_include_paths)

    with open(os.path.join(target_directory, 'op_kernel', f'{op}.cpp'), 'w') as f:
        f.write(context.get('kernel_src'))

    with open(os.path.join(op_engineer_dir, 'CppExtension', 'csrc', f'op.cpp'), 'w') as f:
        f.write(context.get('python_bind_src'))

    try:
        environ_varible = 'ASCEND_CUSTOM_OPP_PATH' # this varible will purturb build if set
        os.environ.pop(environ_varible, None)
        print("[INFO] Begin build")
        os.chdir(target_directory)
        result = subprocess.run(["./build.sh"], check=True, capture_output=True, text=True)
        print("[INFO] Build succeeded")
    except subprocess.CalledProcessError as e:
        print("[INFO] Build failed!")
        error_output = ''
        for line in e.stdout.split('\n'):
            if '[ERROR]' in line or 'error:' in line:
                print(line)
                error_output += line
                error_output += '\n'
        for line in e.stderr.split('\n'):
            if '[ERROR]' in line or 'error:' in line:
                print(line)
                error_output += line
                error_output += '\n'
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{error_output}'
        raise Exception(feedback)



    try:
        print("[INFO] Begin deploy")
        os.chdir(os.path.join(target_directory, 'build_out'))
        # result = subprocess.run(["./custom_opp_ubuntu_aarch64.run", f'--install-path={deploy_path}'], check=True, capture_output=True, text=True)
        result = subprocess.run(["./custom_opp_ubuntu_aarch64.run"], check=True, capture_output=True, text=True)
        print("[INFO] Deploy succeeded")
    except subprocess.CalledProcessError as e:
        print("[INFO] Deploy failed!")
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}'
        raise Exception(feedback)



    try:
        print("[INFO] Begin pybind")
        os.chdir(os.path.join(op_engineer_dir, 'CppExtension'))
        result = subprocess.run(['bash', "build_and_run.sh"], check=True, capture_output=True, text=True)
        print("[INFO] Pybind succeeded\n")
    except subprocess.CalledProcessError as e:
        # Print error if build.sh fails
        print("[INFO] Pybind failed!")
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}'
        raise Exception(feedback)

    # Update ASCEND_CUSTOM_OPP_PATH
    custom_opp_path = f"{project_root_path}/ascend_op_projects/opp/vendors/customize"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = custom_opp_path

    # Update LD_LIBRARY_PATH
    if 'ascend_op_projects' not in os.environ["LD_LIBRARY_PATH"]:
        custom_lib_path = f"{project_root_path}/ascend_op_projects/opp/vendors/customize/op_api/lib/"
        existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{custom_lib_path}:{existing_ld_path}"
    
    try:
        compile(context['model_src'], "<string>", "exec")
        exec(context['model_src'], context)  # For Python, use exec() (be careful with untrusted code)
    except Exception as e:
        raise Exception(f'Error in generated code {e}')

    os.chdir(project_root_path)



if __name__ == '__main__':
    import torch
    import torch_npu
    import custom_ops_lib
    op = 'relu'
    generated_method = getattr(custom_ops_lib, op)
