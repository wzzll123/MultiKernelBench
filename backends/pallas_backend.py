import torch
from backends.backend_registry import register_backend, Backend
import os
import gc
import torch_xla.core.xla_model as xm
from utils.correctness import execute_template
from config import num_perf_trials, num_warmup
import torch_xla
import torch_xla.debug.metrics as met

@register_backend('pallas')
class PallasBackend(Backend):
    def __init__(self):
        self.context = {}
        self.device = self.get_device()

    def get_device(self):
        return xm.xla_device()

    def get_hardware_name(self):
        return 'v2-8'

    def compile(self, generated_code, op):
        try:
            compile(generated_code, "<string>", "exec")
            exec(generated_code, self.context)
            return True, None
        except Exception as e:
            return False, str(e)

    def correctness_execution(self, ref_src):
        exec(ref_src, self.context)
        def synchronize(device=None):
            torch_xla.sync(wait=True)

        return execute_template(synchronize, self.device, self.context)

    def time_execution(self, eval_target='ModelNew'):
        get_inputs = self.context['get_inputs']
        get_init_inputs = self.context['get_init_inputs']
        ModelNew = self.context[eval_target]
        init_inputs = get_init_inputs()
        init_inputs = [
            x.to(device=self.device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]
        with torch.no_grad():
            custom_model = ModelNew(*init_inputs).to(self.device)
        ExecuteTime_list = self.profile_op(get_inputs, custom_model)
        return ExecuteTime_list
    
    def profile_op(self, get_inputs, method):
        met.clear_counters()
        device = xm.xla_device()
        # warmup
        for _ in range(10):
            inputs = get_inputs()
            inputs = [
                x.to(device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            output = method(*inputs)
            xm.mark_step()  # 确保所有操作完成
        res_time = []

        inputs = get_inputs()
        inputs = [
            x.to(device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        met.clear_counters()
        met.clear_all()
        for _ in range(num_perf_trials + 1):
            met.clear_counters()
            met.clear_all()
            # 同步设备，确保计算完成
            output = method(*inputs)
            torch_xla.sync(wait=True)
            xm.mark_step()
            xm.wait_device_ops()
            # 获取ExecuteTime指标的数据
            # execute_time_ns = met.metric_data('ExecuteTime')[1]
            execute_time_ns = met.metric_data('ExecuteTime')[2][-1][1]
            execute_time_sec = execute_time_ns / 1e9
            res_time.append(execute_time_sec)
        return res_time[1:]
    
    def cleanup(self):
        del self.context
        gc.collect()
        xm.mark_step()
        xm.wait_device_ops()
