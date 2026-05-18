BACKEND_REGISTRY = {}

def register_backend(name):
    def decorator(cls):
        BACKEND_REGISTRY[name] = cls()
        return cls
    return decorator

class Backend:
    def get_device(self):
        raise NotImplementedError

    def get_hardware_name(self):
        raise NotImplementedError

    def compile(self, generated_code, op):
        raise NotImplementedError

    def correctness_execution(self, ref_src):
        raise NotImplementedError

    def time_execution(self, eval_target='ModelNew'):
        # Support both modelNew and baseline eval
        raise NotImplementedError

    def detect_cheating(self, generated_code):
        """Return (is_cheating, info) for LLM outputs that bypass custom kernels.

        Backends opt in because custom-kernel entry patterns differ across
        CUDA/Triton/Pallas/AscendC.
        """
        return False, None

    def cleanup(self):
        pass
