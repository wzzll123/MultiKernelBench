import torch
from backends.backend_registry import register_backend, Backend
from backends.ascendc_backend import AscendBackend
import hashlib
import linecache
import triton

@register_backend('triton_ascend')
class TritonBackend(AscendBackend):
    def compile(self, generated_code, op):
        try:
            # ----- 1. stash source in linecache ------------------------------
            fake_fname = f"<triton_{hashlib.md5(generated_code.encode()).hexdigest()}>"
            linecache.cache[fake_fname] = (
                len(generated_code),          # length
                None,                         # mtime  (unused)
                generated_code.splitlines(True),
                fake_fname,
            )

            # ----- 2. compile + exec ----------------------------------------
            py_obj = compile(generated_code, fake_fname, "exec")
            exec(py_obj, self.context)

            # ----- 3. eager‑compile kernels ---------------------------------
            for obj in self.context.values():
                if hasattr(obj, "warm_cache"):   # Triton ≥2.1 JITFunction
                    try:
                        obj.warm_cache()         # compile signature‑only
                    except TypeError:
                        pass                     # needs runtime tensors – skip

            return True, None

        except Exception as e:
            # propagate compile‑time details to caller
            return False, str(e)
