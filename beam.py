"""
Sorry guys...
But also you're welcome for free adversarial data :)
"""

import instructor
import json
import numpy as np
import random
from anthropic import AsyncAnthropic
from datasets import load_dataset
from dotenv import load_dotenv
from functools import partial
from itertools import chain
from openai import AsyncOpenAI
from rich.console import Console
from transformers import LlamaTokenizer
from typing import Union
from vllm import LLM
from beam_utils import YesNo, batch_gpt_call, get_labels, mutate, similar
from utils import find_label, setup_target_llm

load_dotenv()
console = Console()


def beam_search(
    init: str,
    answer: str,
    context: str,
    og_label: str,
    client: Union[AsyncAnthropic, AsyncOpenAI],
    instructor_client: Union[AsyncAnthropic, AsyncOpenAI],
    llm_under_haize: LLM,
    model_name: str,
    tokenizer: LlamaTokenizer,
    beam_size: int = 10,
    explore_size: int = 10,
    max_iters: int = 10,
    desired_screwups: int = 20,
):

    beam: list[str] = [init]
    flipped_screwups, flipped_strings = (
        [],
        set(),
    )  # Candidates that flip the boundary. Oopsies!
    i = 0
    while i < max_iters and len(flipped_screwups) < desired_screwups:
        console.print(
            f"\n\n---------------------- Iter {i} -----------------------", style="blue"
        )
        i += 1

        mutant_prompts = []
        while not mutant_prompts:
            # Searching all nodes from beam
            mutant_prompts = batch_gpt_call(
                list(chain.from_iterable([node] * explore_size for node in beam)),
                mutate,
                client,
                "gpt-3.5-turbo",
            )
            # Ensure semantic similarity of question
            semantic_sims = batch_gpt_call(
                mutant_prompts,
                partial(similar, og_question=init),
                instructor_client,
                "gpt-4o",
            )
            mutant_prompts = [
                m
                for i, m in enumerate(mutant_prompts)
                if semantic_sims[i]
                and semantic_sims[i] == YesNo.YES
                and m not in flipped_strings
            ]
            beam.extend(mutant_prompts)

        # Get hallucination detection model's predicted labels
        labels, raw_responses, scores = get_labels(
            llm_under_haize,
            model_name,
            tokenizer,
            mutant_prompts,
            answer,
            context,
            og_label,
        )
        new_flipped = []
        for mutant, label, resp in zip(mutant_prompts, labels, raw_responses):
            # Check if hallucination detection model is being silly
            if label != og_label:
                new_flipped.append(
                    {
                        "haize_variant": mutant,
                        "variant_response": resp,
                        "variant_label": find_label(resp),
                    }
                )
                flipped_strings.add(mutant)
        flipped_screwups.extend(new_flipped)

        console.print(f"\nTOTAL FAILURES ==> {len(flipped_screwups)}", style="red")
        console.print("== NET NEW FAILURES ==", style="red")
        console.print(new_flipped, style="red")

        # Beam me up scotty.
        idx = np.argsort(-np.array(scores))
        beam = list(np.array(beam)[idx][:beam_size])
        console.print(
            f"\n-------------------------------------------------------", style="blue"
        )

    return list(flipped_screwups)


if __name__ == "__main__":
    # Example testing...
    dataset = load_dataset("PatronusAI/HaluBench")["test"]
    example = random.choice(dataset)
    question, answer, context, label = (
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

    client = AsyncOpenAI()
    model_name = "PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct"
    instructor_client = instructor.patch(AsyncOpenAI())
    tokenizer, llm_under_haize, _ = setup_target_llm()

    flipped_screwups = beam_search(
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
        max_iters=30,
    )
    final = {
        "question": question,
        "context": context,
        "answer": answer,
        "og_label": label,
        "screwups": flipped_screwups,
    }

    json.dump(final, open("beam_results.json", "w"), indent=4)
