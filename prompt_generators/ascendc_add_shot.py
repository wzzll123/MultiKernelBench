from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from prompt_generators.prompt_utils import read_relavant_files, ascendc_template



@register_prompt("ascendc", "add_shot")
class AscendcDefaultPromptStrategy(BasePromptStrategy):        
    def generate(self, op):
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('ascendc', op, 'add')
        return ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, 'add')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print the AscendC add-shot prompt for an op.")
    parser.add_argument("op", nargs="?", default="relu", help="Benchmark op name, default: relu")
    args = parser.parse_args()

    print(AscendcDefaultPromptStrategy().generate(args.op))
