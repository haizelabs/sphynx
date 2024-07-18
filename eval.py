# If you think we're BSing, verify our results yourself

import argparse
import json
from tqdm import tqdm
from transformers import (
    LlamaTokenizer,
)
from typing import Literal
from vllm import LLM, SamplingParams
from hallu_detection_prompt import PROMPT
from utils import find_label, get_inference_funcs, setup_target_llm


def run_inference(
    model_name: str,
    haize_question_variants: list[str],
    messages: list[dict],
    tokenizer: LlamaTokenizer = None,
    sample_params: SamplingParams = SamplingParams(max_tokens=8000, temperature=0),
    llm: LLM = None,
    batch_size: int = 8,
    og_label=None,
) -> list[dict]:

    if not messages:
        return []

    inputs, generate, parse_responses = get_inference_funcs(
        messages, llm, tokenizer, model_name, sample_params
    )

    results = []
    for i in tqdm(range(0, len(inputs), batch_size), desc="Processing batches"):
        batch_inputs = inputs[i : i + batch_size]
        responses = generate(batch_inputs)
        text_responses = parse_responses(responses)
        labels = list(map(find_label, text_responses))
        results += [
            {"haize_variant": s, "variant_response": r, "variant_label": l}
            for s, r, l in zip(haize_question_variants, text_responses, labels)
        ]

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct",
        help="The hallucination detection model to be haized.",
        choices=[
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct",
        ],
    )
    parser.add_argument(
        "--benchmark-file",
        default="sphynx_hallu_induction.json",
    )
    args = parser.parse_args()
    model_name = args.model

    tokenizer, llm, sample_params = setup_target_llm(model_name)
    input_examples = json.load(open(args.benchmark_file, "r"))
    all_results = []
    for ex in input_examples:

        d, q, c, a, ol = map(
            lambda x: ex[x], ["data-id", "question", "context", "answer", "og_label"]
        )
        haize_question_variants = [x["haize_variant"] for x in ex["haize_set"]]
        formatted_messages = [
            [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        **dict(zip(["question", "context", "answer"], [screw, c, a]))
                    ),
                }
            ]
            for screw in haize_question_variants
        ]

        haized_results = run_inference(
            model_name,
            haize_question_variants,
            formatted_messages,
            tokenizer=tokenizer,
            sample_params=sample_params,
            llm=llm,
            batch_size=36,
            og_label=ol,
        )
        ex["haize_set"] = haized_results
        all_results.append(ex)
        save_name = model_name.split("/")[-1]
        json.dump(all_results, open(f"results/{save_name}_being_silly.json", "w"), indent=4)
