"""
A trivial beam search pertubation method that confuses the heck out of leading hallucination detection models.
We perturb the Question component of (Context, Question, Answer).
This algorithm is literally almost as dumb as random search.
Yes it still finds plenty of scenarios that decimate "SOTA" hallucination detection models...
"""

import argparse
import instructor
import json
import random
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from transformers import LlamaTokenizer
from typing import Union
from vllm import LLM
from beam import beam_search
from utils import setup_target_llm

load_dotenv()
console = Console()


def sphynx_break(
    client: AsyncOpenAI,
    instructor_client: AsyncOpenAI,
    tokenizer: LlamaTokenizer,
    llm_under_haize: Union[LLM, AsyncOpenAI],
    model_name: str,
    example: dict,
) -> dict:
    id, question, answer, context, label = (
        example["id"],
        example["question"],
        example["answer"],
        example["passage"],
        example["label"],
    )

    console.print(
        "------ Original Question, Answer, Context, Label ------", style="green"
    )
    print("\n<<QUESTION>>")
    print(question)
    print("\n<<ANSWER>>")
    print(answer)
    print("\n<<CONTEXT>>")
    print(context)
    print("\n<<LABEL>>")
    print(label)
    console.print(
        "\n-------------------------------------------------------\n\n", style="green"
    )

    # Search for ways to rewrite the original question in ways that confuse hallucination detection models
    screwups = beam_search(
        question,
        answer,
        context,
        label,
        client,
        instructor_client,
        llm_under_haize,
        model_name,
        tokenizer,
        beam_size=15,
        explore_size=20,
        max_iters=15,
        # Capping the number of desired screwups so we don't embarass these SOTA detection models too much...
        desired_screwups=5,
    )
    final = {
        "data-id": id,
        "question": question,
        "context": context,
        "answer": answer,
        "og_label": label,
        "lynx-screwups": screwups,
    }

    return final


def main():

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
        "--num-examples",
        default=100,
        help="How many (question, context, answer) triples to haize. Can increase num-examples to however large. We just didn't want to waste $ so we set it to 100.",
    )
    args = parser.parse_args()

    dataset = load_dataset("PatronusAI/HaluBench")["test"]
    examples = random.choices(dataset, k=args.num_examples)

    client = AsyncOpenAI()
    instructor_client = instructor.patch(AsyncOpenAI())
    tokenizer, llm_under_haize, _ = setup_target_llm(args.model)

    results = []
    for ex in examples:
        result = sphynx_break(
            client, instructor_client, tokenizer, llm_under_haize, args.model, ex
        )
        result["data-id"] = ex["id"]
        results.append(result)
        save_name = args.model.split("/")[-1]
        json.dump(results, open(f"{save_name}_being_silly.json", "w"), indent=4)


if __name__ == "__main__":
    main()
