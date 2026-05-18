from prompt_generators.prompt_registry import BasePromptStrategy, register_prompt
from prompt_generators.prompt_utils import (
    ascendc_direct_launch_template,
    read_direct_launch_relevant_files,
)


@register_prompt("ascendc_direct_launch", "add_shot")
class AscendCDirectLaunchAddShotPromptStrategy(BasePromptStrategy):
    def generate(self, op) -> str:
        arc_src, example_arch_src, example_json_src = read_direct_launch_relevant_files(
            op, "add"
        )
        return ascendc_direct_launch_template(
            arc_src, example_arch_src, example_json_src, op, "add"
        )
