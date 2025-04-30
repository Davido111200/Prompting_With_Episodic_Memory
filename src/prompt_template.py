class ZeroshotActionPromptTemplate:
    """
    Zeroshot template for input prompts of action LLM
    """
    def __init__(self, template):
        self.template = template

    def fill(self, action_instruction, input_prompt):
        return self.template.replace("[instruction]", action_instruction).replace("[query]", input_prompt)
