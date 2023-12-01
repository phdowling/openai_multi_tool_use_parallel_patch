import functools
import json
import logging
import openai

from collections import defaultdict
from typing import List, Optional, Dict

# Patches this issue: https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653


def fix_tool_calls(tool_calls: Optional[List[openai.types.chat.ChatCompletionMessageToolCall]]):
    if tool_calls is None:
        return

    replacements: Dict[int, List[openai.types.chat.ChatCompletionMessageToolCall]] = defaultdict(list)
    for i, tool_call in enumerate(tool_calls):
        current_function = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        if current_function in ('parallel', "multi_tool_use.parallel"):
            logging.debug("OpenAI did a weird pseudo-multi-tool-use call, fixing call structure..")
            for _fake_i, _fake_tool_use in enumerate(function_args['tool_uses']):
                _function_args = _fake_tool_use['parameters']
                _current_function = _fake_tool_use['recipient_name']
                if _current_function.startswith("functions."):
                    _current_function = _current_function[len("functions."):]

                fixed_tc = openai.types.chat.ChatCompletionMessageToolCall(
                    id=f'{tool_call.id}_{_fake_i}',
                    type='function',
                    function=openai.types.chat.chat_completion_message_tool_call.Function(
                        name=_current_function,
                        arguments=json.dumps(_function_args)
                    )
                )
                replacements[i].append(fixed_tc)

    shift = 0
    for i, replacement in replacements.items():
        tool_calls[:] = tool_calls[: i + shift] + replacement + tool_calls[i + shift + 1:]
        shift += len(replacement)


def _patch_tool_calling(f, is_async):
    if is_async:
        @functools.wraps(f)
        async def wrapped(*args, **kwargs):
            response = await f(*args, **kwargs)
            for choice in response.choices:
                fix_tool_calls(choice.message.tool_calls)
            return response
    else:
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            response = f(*args, **kwargs)
            for choice in response.choices:
                fix_tool_calls(choice.message.tool_calls)
            return response
    return wrapped


def _wrap_init(init):
    @functools.wraps(init)
    def new_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.chat.completions.create = _patch_tool_calling(
            self.chat.completions.create, is_async=isinstance(self, openai.AsyncOpenAI)
        )
    return new_init


openai.AsyncOpenAI.__init__ = _wrap_init(openai.AsyncOpenAI.__init__)
openai.OpenAI.__init__ = _wrap_init(openai.OpenAI.__init__)
