import json

from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

import openai_multi_tool_use_parallel_patch as patch


def test_patch():
    tool_calls = [ChatCompletionMessageToolCall(
        id="1",
        type="function",
        function=Function(
            name="multi_tool_use.parallel",
            arguments=json.dumps({
                "tool_uses": [{
                    "recipient_name": "real_func",
                    "parameters": {"arg": "hello_world_1"}
                },{
                    "recipient_name": "real_func",
                    "parameters": {"arg": "hello_world_2"}
                }]
            })
        )
    )]
    patch.fix_tool_calls(tool_calls)
    assert tool_calls == [ChatCompletionMessageToolCall(
        id="1_0",
        type="function",
        function=Function(
            name="real_func",
            arguments='{"arg": "hello_world_1"}'
        )
    ), ChatCompletionMessageToolCall(
        id="1_1",
        type="function",
        function=Function(
            name="real_func",
            arguments='{"arg": "hello_world_2"}'
        )
    )]