import asyncio
from enum import Enum
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tqdm import tqdm
from transformers import LlamaTokenizer
from typing import Callable
from vllm import LLM, SamplingParams
from hallu_detection_prompt import PROMPT
from utils import find_label, get_inference_funcs


token_to_id = {
    "PASS": 51935,  # The actual token is " PASS"
    "FAIL": 34207,  # The actual token is " FAIL"
}
flip_label: Callable = lambda x: "FAIL" if x == "PASS" else "PASS"


class YesNo(Enum):
    YES = "Yes"
    NO = "No"


class SimilarityResponse(BaseModel):
    similar: YesNo


async def mutate(prompt, client: AsyncOpenAI, model):
    mutate_prompt = "Your job is to slightly perturb prompts while preserving their original meaning. You can feel free to modify syntactic and grammatical structure, and feel free to use reasonably similar synonyms or phrases to replace existing words. Do not, however, change the original intent of the question."

    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": mutate_prompt,
            },
            {"role": "user", "content": f"Perturb the following prompt: {prompt}"},
        ],
        temperature=0.85,
    )
    return completion.choices[0].message.content


async def similar(prompt, client: AsyncOpenAI, model, og_question):
    # Job is to see if questions are semantically similar enough
    simlar_prompt = "Your job is to compare two sentences and determine whether or not they are asking the same underlying question."

    try:
        completion: SimilarityResponse = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": simlar_prompt,
                },
                {
                    "role": "user",
                    "content": f"Consider Sentence A: {og_question}.\nAnd now consider Sentence B: {prompt}.\nDo the two Sentences ask very precisely the same question? Answer either Yes or No.",
                },
            ],
            max_retries=5,
            response_model=SimilarityResponse,
        )
        return completion.similar
    except ValidationError as e:
        print(e)
        return None


def batch_gpt_call(
    prompts: list[str], op: Callable, client: AsyncOpenAI, model="gpt-3.5-turbo"
):
    async def handler(prompts):
        tasks = [op(prompt, client, model) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        return responses

    return asyncio.run(handler(prompts))


def remove_indices(lst, indices):
    # Sort the indices in descending order to avoid index shifting
    for index in sorted(indices, reverse=True):
        if 0 <= index < len(lst):
            lst.pop(index)
    return lst


def get_labels(
    target_model: LLM,
    model_name: str,
    tokenizer: LlamaTokenizer,
    questions: list[str],
    answer: str,
    context: str,
    og_label: str,
    # TODO: temp closer to 0
    sample_params: SamplingParams = SamplingParams(
        max_tokens=8000, logprobs=20, temperature=0
    ),
    batch_size: int = 100,
    use_logprobs: bool = False,
) -> tuple[list, list, list]:
    messages = [
        [
            {
                "role": "user",
                "content": PROMPT.format(
                    question=question, context=context, answer=answer
                ),
            }
        ]
        for question in questions
    ]
    messages, generate, parse_responses = get_inference_funcs(
        messages, target_model, tokenizer, model_name, sample_params
    )
    labels, all_responses, scores = [], [], []

    for i in tqdm(range(0, len(messages), batch_size), desc="Processing batches"):
        responses = generate(messages[i : i + batch_size])
        text_responses = parse_responses(responses)
        idx_to_remove = []
        for i, resp in enumerate(responses):
            try:
                if use_logprobs:
                    scores.append(
                        resp.outputs[0]
                        # 3rd to last token is FAIL or PASS
                        .logprobs[-3][token_to_id[flip_label(og_label)]].logprob
                    )
                else:
                    # This devolves to random search.
                    # TODO: use logprobs from OpenAI API.
                    # Harder in general to know what position the PASS or FAIL token is except for models trained on the same output format.
                    scores.append(10)
            except Exception as e:
                print(e)
                idx_to_remove.append(i)
                continue

        text_responses = remove_indices(text_responses, idx_to_remove)
        labels.extend(map(find_label, text_responses))
        all_responses.extend(text_responses)

    return labels, all_responses, scores
