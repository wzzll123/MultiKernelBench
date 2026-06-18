from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from prompt_generators.prompt_utils import generate_template, read_relavant_files
from dataset import category2exampleop, dataset

@register_prompt("musa", "selected_shot")
class MusaSelectPromptStrategy(BasePromptStrategy):
    def generate(self, op) -> str:
        category = dataset[op]['category']
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('musa', op, category2exampleop[category])
        return generate_template(arc_src, example_arch_src, example_new_arch_src, 'MUSA')
