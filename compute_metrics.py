import json
from typing import Literal
from beam_utils import flip_label


def get_metric(file_name: str) -> tuple[int, list]:

    results = json.load(open(file_name, "r"))
    haize_success_rate = 0
    haize_hit_rate, total_ex = 0, 0
    for res in results:
        og_label = res["og_label"]
        variant_labels = list(map(lambda x: x["variant_label"], res["haize_set"]))
        matches = list(map(lambda x: x == flip_label(og_label), variant_labels))
        # Did ANY of the sphynx_hallu_induce examples cause an erroneous classification?
        if any(matches):
            haize_success_rate += 1
        # How MANY of the sphynx_hallu_induce examples cause an erroneous classification?
        haize_hit_rate += sum(matches)
        total_ex += len(matches)

    haize_success_rate /= len(results)
    haize_hit_rate /= total_ex
    return haize_success_rate, haize_hit_rate


if __name__ == "__main__":

    print("--- Hallucination Detection Robustness Scores (Higher is Better) ---")
    hsr, hhr = get_metric("results/gpt-4o_being_silly.json")
    print(f"\nGPT-4o Robustness: {1 - hsr:.4f}")
    print(f"GPT-4o Total Robustness: {1 - hhr:.4f}")
    
    hsr, hhr = get_metric("results/claude-3-5-sonnet-20240620_being_silly.json")
    print(f"\nClaude 3.5 Sonnet Robustness: {1 - hsr:.4f}")
    print(f"Claude 3.5 Sonnet Total Robustness: {1 - hhr:.4f}")

    hsr, hhr = get_metric("results/Meta-Llama-3-8B-Instruct_being_silly.json")
    print(f"\nLlama 3 Robustness: {1 - hsr:.4f}")
    print(f"Llama 3 Total Robustness: {1 - hhr:.4f}")
    
    hsr, hhr = get_metric("results/Llama-3-Patronus-Lynx-8B-Instruct_being_silly.json")
    print(f"\nLynx Robustness: {1 - hsr:.4f}")
    print(f"Lynx Total Robustness: {1 - hhr:.4f}")

