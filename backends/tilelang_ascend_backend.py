import torch_npu
import torch
from backends.backend_registry import register_backend, Backend
from utils.ascend_compile_pipeline import ascend_compile
from utils.correctness import execute_template
from utils.performance import time_execution_event_template
from config import project_root_path,ascendc_device
import tilelang
import tempfile
import importlib
import importlib.util
import tempfile
import os



@register_backend('tilelang_ascend')
class TilelangAscendBackend(Backend):
    def __init__(self):
        self.context = {}
        self.device = self.get_device()
        tilelang.cache.clear_cache()
        
    def get_device(self):
        return torch.device('npu:0')

    def get_hardware_name(self):
        return ascendc_device  # torch_npu.npu.get_device_name(device) causes crash

    def compile(self, generated_code, op):
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
                tmp_file.write(generated_code.encode("utf-8"))
                tmp_path = tmp_file.name

            # Import the module from that file so code objects get co_filename = tmp_path
            module_name = "temp_module"
            spec = importlib.util.spec_from_file_location(module_name, tmp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            self.context['ModelNew'] = getattr(module, "ModelNew")
            return True, None
        except Exception as e:
            # propagate compile‑time details to caller
            return False, str(e)


    def correctness_execution(self, ref_src):
        synchronize = torch_npu.npu.synchronize
        try:
            exec(ref_src, self.context)
        except Exception as e:
            raise RuntimeError(f"Failed to compile reference model: {str(e)}")
        return execute_template(synchronize, self.device, self.context)

    def time_execution(self, eval_target='ModelNew'):
        event_class = torch_npu.npu.Event
        synchronize = torch_npu.npu.synchronize
        return time_execution_event_template(self.context, self.device, synchronize, event_class, eval_target)

    def cleanup(self):
        del self.context
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize(device=self.device)
        
