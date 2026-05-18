import torch_npu
import torch
from backends.backend_registry import register_backend, Backend
from utils.cheating_detection import detect_python_kernel_cheating
from utils.ascend_compile_pipeline import ascend_compile
from utils.correctness import execute_template
from utils.performance import time_execution_event_template
from config import project_root_path,ascendc_device
import os

@register_backend('ascendc')
class AscendBackend(Backend):
    def __init__(self):
        self.context = {}
        self.device = self.get_device()
    def get_device(self):
        return torch.device('npu:0')

    def get_hardware_name(self):
        return ascendc_device  # torch_npu.npu.get_device_name(device) causes crash

    def compile(self, generated_code, op):
        try:
            ascend_compile(generated_code, op, self.context)
            return True, None
        except Exception as e:
            os.chdir(project_root_path)
            return False, str(e)

    def detect_cheating(self, generated_code):
        return detect_python_kernel_cheating(generated_code)

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
