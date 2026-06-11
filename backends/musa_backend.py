import torch
import torch_musa  # noqa: F401 - importing registers torch.musa

from backends.backend_registry import Backend, register_backend
from utils.correctness import execute_template
from utils.performance import time_execution_event_template


@register_backend("musa")
class MusaBackend(Backend):
    def __init__(self):
        self.context = {}
        self.device = self.get_device()

    def get_device(self):
        return torch.device("musa:0")

    def get_hardware_name(self):
        return torch.musa.get_device_name(device=self.device)

    def compile(self, generated_code, op):
        try:
            compile(generated_code, "<string>", "exec")
            exec(generated_code, self.context)
            return True, None
        except Exception as e:
            return False, str(e)

    def correctness_execution(self, ref_src):
        synchronize = torch.musa.synchronize
        try:
            exec(ref_src, self.context)
        except Exception as e:
            raise RuntimeError(f"Failed to compile reference model: {str(e)}")
        return execute_template(synchronize, self.device, self.context)

    def time_execution(self, eval_target="ModelNew"):
        synchronize = torch.musa.synchronize
        event_class = torch.musa.Event
        return time_execution_event_template(
            self.context, self.device, synchronize, event_class, eval_target
        )

    def cleanup(self):
        del self.context
        with torch.musa.device(self.device):
            torch.musa.empty_cache()
            if hasattr(torch.musa, "reset_peak_memory_stats"):
                torch.musa.reset_peak_memory_stats(device=self.device)
            torch.musa.synchronize(device=self.device)
