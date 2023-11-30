# Patching `multi_tool_use.parallel` in OpenAI
Quick monkey-patching fix for OpenAI's hallucinated multi_tool_use.parallel issue.

This is a simple workaround for [this issue](https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653) where the GPT ChatCompletions API sometimes hallucinates a strange alternative way of calling multiple tools at once.
In my experience, these calls at least always follow a predictable structure and can be hotfixed to look like normal, legal tool calls. This patch does that, and also rewrites the message accordingly (which, in my experience, makes the model perform calls work in the normal expected way.)

This fix will be redundant soon, but it looks like I am not the only one who occasionally runs into this error, so I thought I would share this workaround.

## Installation and usage
`pip install openai-multi-tool-use-parallel-patch`

In your code:
```python
import openai_multi_tool_use_parallel_patch  # import applies the patch
import openai

client = openai.AsyncOpenAI(...)  # sync client will be patched too

...

response = await client.chat.completions.create(...)  # no changes to the call signature or response vs vanilla OpenAI client
```

That's it - theoretically this should work even if you `import openai` first, but at the least you will need to make sure not to create your `openai.OpenAI` or `openai.AsyncOpenAI` instances before importing this patch.
