import re
from litellm import batch_completion
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
)
from typing import Callable
from vllm import LLM, SamplingParams


def find_label(res: str):
    # Regex instead of JSON because of poor finetuning job...
    match = re.search(r"\"SCORE\":\s*(\w+)", res)
    if match:
        score_value = match.group(1)
        return score_value
    # A bandaid solution
    final_search = res.split('"SCORE":')[-1]
    if "PASS" in final_search:
        return "PASS"
    if "FAIL" in final_search:
        return "FAIL"
    return None


def setup_target_llm(model_name: str):

    tokenizer, llm, sample_params = None, None, None

    # Hacky but it is what it is
    if "Llama" in model_name:
        assert model_name in [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct",
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm = LLM(model=model_name)
        sample_params = SamplingParams(max_tokens=8000)

    return tokenizer, llm, sample_params


def get_inference_funcs(
    messages: list[dict],
    llm: LLM,
    tokenizer: LlamaTokenizer,
    model_name: str,
    sample_params: SamplingParams,
):
    if "Llama-" in model_name:
        inputs = tokenizer.apply_chat_template(messages, tokenize=False)
        generate: Callable = lambda x: llm.generate(x, sample_params)
        parse_response: Callable = lambda responses: [
            resp.outputs[0].text for resp in responses
        ]
    else:
        inputs = messages
        generate: Callable = lambda x: batch_completion(model=model_name, messages=x, temperature=0)
        if "gpt" in model_name:
            parse_response: Callable = lambda responses: [
                completion.choices[0].message.content for completion in responses
            ]
        elif "claude" in model_name:
            parse_response: Callable = lambda responses: [
                completion.choices[0].message.content for completion in responses
            ]

    return inputs, generate, parse_response
