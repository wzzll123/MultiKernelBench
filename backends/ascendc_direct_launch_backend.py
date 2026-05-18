import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch_npu

from backends.backend_registry import Backend, register_backend
from config import ascendc_device, op_engineer_dir
from utils.correctness import execute_template
from utils.performance import time_execution_event_template


def _strip_json_fence(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()
    if stripped.startswith("json\n"):
        return stripped[5:].strip()
    return stripped


def _detect_ascend_path():
    for env_name in ("ASCEND_INSTALL_PATH", "ASCEND_HOME_PATH"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser().resolve()

    candidates = [
        Path.home() / "Ascend" / "ascend-toolkit" / "latest",
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[-1]


def _safe_rel_path(path_text):
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe source path: {path_text}")
    return path


def _format_cmake_list(items, indent=4):
    prefix = " " * indent
    return "\n".join(f"{prefix}{item}" for item in items)


def _sanitize_name(name):
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return sanitized or "task"


def _create_work_dir(op):
    root = Path(op_engineer_dir) / "ascendc_direct_launch"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    work_dir = root / f"{_sanitize_name(op)}_{timestamp}_{os.getpid()}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    return work_dir


def _find_extension_path(build_dir, module_name):
    matches = sorted(build_dir.glob(f"{module_name}*.so"))
    if not matches:
        raise FileNotFoundError(f"Cannot find built extension {module_name}*.so in {build_dir}")
    return matches[0]


def _unique_paths(paths):
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _looks_like_binding_source(path):
    try:
        return "PYBIND11_MODULE" in path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False


def _infer_pybind_module_name(paths):
    names = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in re.finditer(r"PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", text):
            name = match.group(1)
            if name != "TORCH_EXTENSION_NAME":
                names.append(name)
    names = _unique_paths(names)
    if len(names) > 1:
        raise ValueError(f"Multiple pybind module names found: {', '.join(names)}")
    return names[0] if names else None


def _format_subprocess_error(stage, cmd, cwd, error):
    return (
        f"{stage} failed\n"
        f"Command: {' '.join(cmd)}\n"
        f"Working directory: {cwd}\n"
        f"Exit code: {error.returncode}\n"
        f"[STDOUT]\n{error.stdout or ''}\n"
        f"[STDERR]\n{error.stderr or ''}"
    )


def _run_checked(stage, cmd, cwd, env):
    print(f"[ascendc_direct_launch] {stage}: {' '.join(cmd)}")
    print(f"[ascendc_direct_launch] cwd: {cwd}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip())
        return result
    except subprocess.CalledProcessError as error:
        raise RuntimeError(_format_subprocess_error(stage, cmd, cwd, error)) from error


def _generate_cmakelists(staging_dir, build_dir, module_name, kernel_sources, binding_sources, include_dirs):
    source_lines = [str(staging_dir / src) for src in kernel_sources]
    binding_lines = [str(staging_dir / src) for src in binding_sources]
    include_lines = [str(staging_dir / inc) for inc in include_dirs]

    return f"""cmake_minimum_required(VERSION 3.16.0)
project(MultiKernelBenchAscendCDirectLaunch)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "${{SOC_VERSION}}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "${{ASCEND_CANN_PACKAGE_PATH}}" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "{build_dir}")

if(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the CANN package is installed.")
endif()

include(${{ASCENDC_CMAKE_DIR}}/ascendc.cmake)

message(STATUS "MultiKernelBench AscendC direct launch build")
message(STATUS "  module name: {module_name}")
message(STATUS "  staging dir: {staging_dir}")
message(STATUS "  build dir: {build_dir}")

set(ASCEND_INCLUDE_DIRS
    ${{ASCEND_CANN_PACKAGE_PATH}}/include
    ${{ASCEND_CANN_PACKAGE_PATH}}/include/experiment/runtime
    ${{ASCEND_CANN_PACKAGE_PATH}}/include/experiment/msprof
    ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/include
    ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/ascendc/include/basic_api/impl
    ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/ascendc/include/basic_api/interface
    ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/ascendc/include/highlevel_api/impl
    ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/ascendc/include/highlevel_api/tiling
    ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/ascendc/impl/aicore/basic_api
)

execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATH
)
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH
)
execute_process(COMMAND python3 -m pybind11 --includes
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${{PYBIND11_INC}})

message(STATUS "  torch path: ${{TORCH_PATH}}")
message(STATUS "  torch_npu path: ${{TORCH_NPU_PATH}}")
message(STATUS "  kernel sources: {source_lines}")
message(STATUS "  binding sources: {binding_lines}")

set(COMMON_INCLUDE_DIRS
{_format_cmake_list(include_lines)}
    ${{ASCEND_INCLUDE_DIRS}}
    ${{TORCH_NPU_PATH}}/include
    ${{TORCH_PATH}}/include
    ${{TORCH_PATH}}/include/torch/csrc/api/include
)

ascendc_library(kernels STATIC
{_format_cmake_list(source_lines)}
)
ascendc_include_directories(kernels PRIVATE
  ${{COMMON_INCLUDE_DIRS}}
)

add_library(pybind11_lib SHARED
{_format_cmake_list(binding_lines)}
)
target_link_libraries(pybind11_lib PRIVATE
  kernels
  torch_npu
  m
  dl
)
target_compile_options(pybind11_lib PRIVATE
  ${{PYBIND11_INC}}
  -D_GLIBCXX_USE_CXX11_ABI=1
)
target_compile_definitions(pybind11_lib PRIVATE
  TORCH_EXTENSION_NAME={module_name}
)
target_include_directories(pybind11_lib PRIVATE ${{COMMON_INCLUDE_DIRS}})
target_link_directories(pybind11_lib PRIVATE
  ${{TORCH_PATH}}/lib
  ${{TORCH_NPU_PATH}}/lib
)
set_target_properties(pybind11_lib PROPERTIES
  BUILD_RPATH "${{TORCH_PATH}}/lib;${{TORCH_NPU_PATH}}/lib"
)

execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
)
set_target_properties(pybind11_lib PROPERTIES
  OUTPUT_NAME {module_name}
  PREFIX ""
  SUFFIX "${{PYTHON_EXTENSION_SUFFIX}}"
)
"""


@register_backend("ascendc_direct_launch")
class AscendCDirectLaunchBackend(Backend):
    def __init__(self):
        self.context = {}
        self.device = self.get_device()
        self.work_dir = None

    def get_device(self):
        return torch.device("npu:0")

    def get_hardware_name(self):
        return ascendc_device

    def compile(self, generated_code, op):
        try:
            spec = json.loads(_strip_json_fence(generated_code))
            self.work_dir = _create_work_dir(op)
            print(f"[ascendc_direct_launch] staging dir: {self.work_dir}")
            self._write_sources(spec)
            self._build_extension(spec)
            self._load_model(spec)
            return True, None
        except Exception as e:
            return False, str(e)

    def correctness_execution(self, ref_src):
        synchronize = torch_npu.npu.synchronize
        try:
            exec(ref_src, self.context)
        except Exception as e:
            raise RuntimeError(f"Failed to compile reference model: {str(e)}")
        return execute_template(synchronize, self.device, self.context)

    def time_execution(self, eval_target="ModelNew"):
        event_class = torch_npu.npu.Event
        synchronize = torch_npu.npu.synchronize
        return time_execution_event_template(self.context, self.device, synchronize, event_class, eval_target)

    def cleanup(self):
        del self.context
        self.context = {}
        if self.work_dir is not None:
            print(f"[ascendc_direct_launch] keeping build directory: {self.work_dir}")
            self.work_dir = None
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize(device=self.device)

    def _write_sources(self, spec):
        seen = set()
        for source in spec.get("sources", []):
            rel_path = _safe_rel_path(source["path"])
            if rel_path in seen:
                raise ValueError(f"Duplicate source path: {rel_path}")
            seen.add(rel_path)
            target = self.work_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.get("content", ""), encoding="utf-8")
            print(f"[ascendc_direct_launch] wrote source: {rel_path}")

    def _build_extension(self, spec):
        build_spec = spec.get("build", {})
        kernel_sources = [_safe_rel_path(path) for path in build_spec.get("kernel_sources", [])]
        binding_sources = [_safe_rel_path(path) for path in build_spec.get("binding_sources", [])]
        include_dirs = [_safe_rel_path(path) for path in build_spec.get("include_dirs", [])]
        source_paths = [
            _safe_rel_path(source["path"])
            for source in spec.get("sources", [])
            if source.get("path")
        ]
        cpp_sources = sorted(path for path in source_paths if path.suffix in (".cpp", ".cc", ".cxx"))

        if not binding_sources:
            binding_sources = [
                path for path in cpp_sources
                if path.name == "pybind11.cpp" or _looks_like_binding_source(self.work_dir / path)
            ]
        if not binding_sources:
            raise FileNotFoundError(
                "Cannot infer pybind source. Add a .cpp source containing PYBIND11_MODULE "
                "or set build.binding_sources."
            )
        module_name = (
            build_spec.get("module_name")
            or _infer_pybind_module_name([self.work_dir / path for path in binding_sources])
            or "benchmark_ops"
        )

        if not kernel_sources:
            binding_set = set(binding_sources)
            kernel_sources = [path for path in cpp_sources if path not in binding_set]
        if not kernel_sources:
            raise FileNotFoundError(
                "Cannot infer Ascend C kernel sources. Add non-binding .cpp files "
                "or set build.kernel_sources."
            )

        if not include_dirs:
            include_dirs = _unique_paths(
                [Path(".")]
                + [path.parent for path in source_paths if path.parent != Path(".")]
            )

        build_dir = (
            self.work_dir / _safe_rel_path(build_spec["build_dir"])
            if build_spec.get("build_dir")
            else self.work_dir / binding_sources[0].parent / "build"
        )

        print(f"[ascendc_direct_launch] module: {module_name}")
        print(f"[ascendc_direct_launch] kernel sources: {', '.join(map(str, kernel_sources))}")
        print(f"[ascendc_direct_launch] binding sources: {', '.join(map(str, binding_sources))}")
        print(f"[ascendc_direct_launch] include dirs: {', '.join(map(str, include_dirs))}")
        print(f"[ascendc_direct_launch] build dir: {build_dir}")

        for rel_path in kernel_sources + binding_sources:
            if not (self.work_dir / rel_path).is_file():
                raise FileNotFoundError(f"Missing source file: {rel_path}")

        cmake_dir = build_dir / "_autogen_cmake"
        cmake_dir.mkdir(parents=True, exist_ok=True)
        ascend_path = _detect_ascend_path()
        (cmake_dir / "CMakeLists.txt").write_text(
            _generate_cmakelists(
                staging_dir=self.work_dir,
                build_dir=build_dir,
                module_name=module_name,
                kernel_sources=kernel_sources,
                binding_sources=binding_sources,
                include_dirs=include_dirs,
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["ASCEND_HOME_PATH"] = str(ascend_path)
        configure_cmd = [
            "cmake",
            "-S",
            str(cmake_dir),
            "-B",
            str(build_dir),
            "-DSOC_VERSION=Ascend910B2",
            f"-DASCEND_CANN_PACKAGE_PATH={ascend_path}",
            "-DCMAKE_BUILD_TYPE=Debug",
        ]
        build_cmd = ["cmake", "--build", str(build_dir), "-j"]

        _run_checked("CMake configure", configure_cmd, cwd=self.work_dir, env=env)
        _run_checked("CMake build", build_cmd, cwd=self.work_dir, env=env)

        extension_path = _find_extension_path(build_dir, module_name)
        print(f"[ascendc_direct_launch] built extension: {extension_path}")
        self.context["_ascendc_direct_launch_extension"] = str(extension_path)

    def _load_model(self, spec):
        entry = spec.get("entry", {}).get("model", "ModelNew.py::ModelNew")
        if "::" not in entry:
            raise ValueError(f"Invalid model entry: {entry}")
        model_path_text, model_name = entry.split("::", 1)
        model_path = self.work_dir / _safe_rel_path(model_path_text)
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model entry file: {model_path_text}")

        print(f"[ascendc_direct_launch] loading model entry: {entry}")
        model_src = model_path.read_text(encoding="utf-8")
        model_globals = self.context
        old_file = model_globals.get("__file__")
        model_globals["__file__"] = str(model_path)
        try:
            exec(compile(model_src, str(model_path), "exec"), model_globals)
        finally:
            if old_file is None:
                model_globals.pop("__file__", None)
            else:
                model_globals["__file__"] = old_file

        if model_name not in self.context:
            raise AttributeError(f"Model entry {model_name} not found in {model_path_text}")
