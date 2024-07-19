# Sphynx: Dynamically Testing and Perplexing Leading Hallucination Detection Models

## Table of Contents
1. [Introduction](#introduction)
2. [Searching for Similar, Perplexing Questions](#searching-for-similar-perplexing-questions)
3. [A Stupidly Simple Haizing Algorithm](#a-stupidly-simple-haizing-algorithm)
4. [Perplexing Leading Hallucination Detection Models](#perplexing-sota-hallucination-detection-models)
5. [Haizing Leads to Robustness](#haizing-leads-to-robustness)

## Introduction

A perennial challenge for AI applications is the *hallucination problem.* Everybody knows this: nobody wants LLMs that act up and spit out incorrect or undesired behaviors. 

Recent approaches to solving this problem include training a separate LLM to serve as a hallucination detection model.

Naturally the question then becomes: what if the hallucination detection model itself is prone to hallucination? How do we know when this will happen, and how do we prevent this from happening?

Well, it's probably time to *[haize](https://haizelabs.com/)* these models to figure out where they fail :)

![Sphynx-Flow](sphynx.png)

## Searching for Similar, Perplexing Questions

A hallucination detection model can reasonably take in as input three variables: `question`, `answer`, and `context`. It spits out one of two outputs: `PASS` or `FAIL`. `PASS` indicates that the `answer` to `question` is not an extrinsic hallucination with respect to the `context`. `FAIL` means that `answer` was a hallucination.

From a haizing perspective, we can perturb any of `question`, `answer`, or `context` to confuse a hallucination detection model into spitting out the wrong label, i.e. `FAIL` instead of `PASS` or vice versa. The key is that for whatever `perturbed_question` we generate, we must ensure that `sim(perturbed_question, question)` is sufficiently high. Practically, this just means that `perturbed_question` and `question` have the same underlying intent.

For example, consider the following question (`halueval-9787`):

> *Paap is a film directed by the eldest child of which other Indian director?*

The ground truth answer is `Mahesh Bhatt`.

Now consider this very subtle rephrasing:

> *What Indian director's eldest child directed the film Paap?*

Obviously the answer to both questions -- `Mahesh Bhatt` -- is the same. But *the slight shift in question phrasing* caused a "leading" hallucination detector to think that Mahesh Bhatt is a valid answer to the original question, but a *hallucination* with respect to the second question.

That seems not very desirable...we call that a `gotcha`! 

## A Stupidly Simple Haizing Algorithm

So how do we search for such gotchas? Well, it turns out an extremely simple beam search (almost random search) method can wreak havoc on these hallucination detection models. Essentially, the idea is to fuzz-test the hallucination detector by mining for "rephrasings" of the original question that inch closer to the `PASS`/`FAIL` decision boundary. 

Crossing the boundary -- i.e. flipping the label -- while preserving the original question's intent means we've elicited a failure case of the hallucination detector. Canonically, such a manifestation is called an adversarial attack.

Here's some light pseudocode that explains what's happening:

```
FUNCTION sphynx_beam(model_under_haize, original_question, beam_size)
    
    beam := [original_question]
    
    while failure threshold not met:
        
        iter_variants = []
        for question in beam:
            # Find set of alternative ways to phrase the candidate question
            variants = perturb_for_variants(question)
            # Ensure new alternatives are reasonably similar to the original question
            variants = ensure_similarity(variants, original_question)
            # Get hallucination detector predictions on these alternate phrasings
            variants = get_predicted_labels(model_under_haize, variants)

            iter_variants.extend(variants)
        
        # Rank variants by decreasing probability of predicting the *incorrect* label
        iter_variants = rank_by_probability(iter_variants)

        beam = iter_variants[:beam_size]
        
END FUNCTION
```

For full details, check out `beam.py`.

## Perplexing Leading Hallucination Detection Models

As you can tell, nothing crazy is happening on the algorithmic side! And yet, this simple procedure is able to elicit a wide range of failure modes, showcasing their (lack of) robustness.

Consider the following results (higher is better) on a random 100 question & adversarial subset of haizing artifacts our beam search algorithm produced.

| Metric     |     GPT-4o |  Claude-3.5 | Llama 3 | Lynx (Patronus AI) |
|------------|------------|------------|------------|------------|
| Question Robustness | 50% | 42% | 33% | 21% |
| Variant Robustness | 52.63% | 48.34% | 50.28% | 31.99% |

These scores measure how robust a hallucination detector is with respect to the "gotchas" as described above. The higher the robustness score, the better. Question Robustness measures if the hallucination detector prevents *all* of the adversarial attacks with respect to a question; Variant Robustness measures if the detector prevents a *particular* adversarial attack with respect to a question.

The original questions are randomly sampled from the wonderful [HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench). Our adversarial attacks can be found in `sphynx_halluce_induce.json`. 

Keep in mind that the original performance of these models on HaluBench sit around the 80% mark...

To reproduce these results (e.g. against Llama 3), run the following:

```
python eval.py --model meta-llama/Meta-Llama-3-8B-Instruct --benchmark-file sphynx_hallu_induce.json
python compute_metrics.py
```

To haize a model (i.e. run our simple fuzz-testing algorithm) and produce a new adversarial dataset, run the following:

```
python sphynx.py --model PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct --num-examples 200
```

## Haizing Leads to Robustness

So what's the point of all this?

As we roll out AI in high-stakes use cases, it is critical that we be a bit more thoughtful in dealing with the brittleness and stochasticity of these systems.

As we enter this new era, it's pretty evident that static dataset approaches to building and testing models are probably not going to cut it. As is clear in this particular setting, leading hallucination detection models can perform *great* (80%+ robustness) on *static* datasets, but are pretty much decimated by minimally sophisticated *dynamic* testing approaches like Sphynx.

Rather than wait for your application (e.g. hallucination detector) to fail out in the wild, haizing *forces* these failures to happen in development. Haizing can efficiently generate many samples with minor variations corresponding to different user’s queries to stress-test your system.

The haizing mission is inspired by precisely such observations and ideals. The mission is to bring the art of dynamic and powerful fuzz-testing to address the wonkiness and brittleness of LLM-land. Only by rigorously, scalably, and automatically testing your models to understand all of their corner cases and weaknesses can you even begin to remediate such weaknesses.

Only by haizing can you achieve truly reliable AI systems.