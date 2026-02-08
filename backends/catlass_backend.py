import os

from backends.backend_registry import register_backend
from backends.ascendc_backend import AscendBackend
from config import catlass_include_path, project_root_path
from utils.ascend_compile_pipeline import ascend_compile


@register_backend("catlass")
class CatlassBackend(AscendBackend):
    def compile(self, generated_code, op):
        include_path = catlass_include_path

        try:
            ascend_compile(
                generated_code,
                op,
                self.context,
                extra_kernel_include_paths=[include_path],
            )
            return True, None
        except Exception as e:
            os.chdir(project_root_path)
            return False, str(e)
