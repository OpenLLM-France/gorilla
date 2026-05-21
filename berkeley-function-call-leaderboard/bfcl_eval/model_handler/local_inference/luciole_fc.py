from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.utils import convert_to_tool
from overrides import override


class LucioleFCHandler(QwenFCHandler):
    """Handler for Luciole checkpoints with a Qwen3-like Hermes chat template.

    Tool-call parsing is inherited from QwenFCHandler. _format_prompt renders
    via the tokenizer's own chat_template and reshapes BFCL tools into the
    OpenAI-style {"type": "function", "function": {...}} JSON Schema form —
    the exact shape the model was trained on.
    """

    @override
    def _format_prompt(self, messages, function):
        tools = (
            convert_to_tool(function, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
            if function
            else None
        )
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
