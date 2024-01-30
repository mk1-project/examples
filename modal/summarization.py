## # Batch Document Summarization on Modal
##
## Summarization is the task of generating concise summaries of given data chunks or documents, usually
## processed as a large batch of inference requests. Typical examples of text summarization include generating
## a summary of latest news articles or emails or slack messages for the day. Summarization is also a key building
## block of complex LLM applications like RAG.
##
## > With Flywheel, you can easily **process up to 2x summarization tasks per unit GPU, in the same amount
## of time** compared to leading inference solutions. This directly transaltes to cost savings
## and faster turnaround times for your summarization tasks. Furthermore, the larger the batchsize, the higher the inference
## throughput you can achieve with Flywheel. Combine this with seamless autoscaling on Modal to churn through your inference
## tasks in no time.
##
## In this example, we show how to use a Llama-2-13B-chat model for text summarization on an A100-40GB on Modal using MK1 Flywheel.
## We'll be summarizing news articles from the [`cnn_dailymail`](https://huggingface.co/datasets/cnn_dailymail) dataset from HuggingFace.
## This mimics the scenario where you'd want to summarize chunks of documents or articles.
##
## ## Setup
## We start by importing the necessary pacakges for this example, such as `datasets`. We will also use a pre-loaded volume with the Llama-2-13B-chat model,
## to take full advantage of the faster cold starts using Modal's internal filesystem. You can find more information
## on "Bring-Your-Own-Model" (BYOM) in our [docs](https://docs.mk1.ai/modal/byom.html)
## which describes how to setup volumes to bootstrap your inference with Flywheel for your own models.
##
## While the Flywheel inference runtime can readily autoscale on the Modal platform, we will constrain this example to a single GPU instance
## with [`concurrency_limit`](https://modal.com/docs/guide/scale#limiting-concurrency).

import os
import pickle
from tqdm import tqdm
from datasets import load_dataset

import modal


# Volume with pre-loaded Llama-2-13B-chat model.
volume = modal.Volume.lookup("models")

# Instance the Flywheel inference runtime and bind the volume containing the model weights.
Model = modal.Cls.lookup(
    "mk1-flywheel-latest", "Model", workspace="mk1"
).with_options(
    gpu=modal.gpu.A100(size="40GB"),
    volumes={"/models": volume},
    concurrency_limit=1,
)

model_path = "Llama-2-13b-chat-hf"
model = Model(model_path=os.path.join("/models", model_path))

# The container coldstarts at the first request, so we'll send a sample request to warm it up.
response = model.generate.remote("What is text summarization?", max_tokens=800, eos_token_ids=[1,2], temperature=0.8, top_p=0.95)
print(response['prompt'], response['responses'][0]['text'], "\n\n")

## ## Run the Model
## In this example, we'll batch process the entire for a list of inputs. You can run this application with `python3 summarization.py`.
##
## First, we'll define the prompt template and system prompt for the summarization generation request.
PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST] """

SUMMARIZATION_SYSTEM_PROMPT = """Provide a concise summary of the given text.
The summary should cover all the key points and main ideas presented in the original text,
while also condensing the information into a concise and easy-to-understand format.

Summarize the following: """

## Then, load a set of decently sized (chunks limited to 4k characters) news articles from the `cnn_dailymail` dataset
## and decorate it with the prompt template for summarization.
## We control the size of the workload with `NUM_PROMPTS`. You can get higher throughput with a larger batch of prompts.
## As a reference, 2000 articles may take a little over 6 minutes to process.
NUM_PROMPTS = 64
dataset = load_dataset("cnn_dailymail", '1.0.0', split="test")
articles = list(filter(lambda x: len(x) < 4096, [x for x in dataset['article']]))
articles = articles[:NUM_PROMPTS]

prompts = [
    PROMPT_TEMPLATE.format(system=SUMMARIZATION_SYSTEM_PROMPT, user=text) for text in articles
]

## Batch process all the prompts, while accumulating the responses. Note that in this case,
## the generation requests are processed asynchronously using
## [`map`](https://modal.com/docs/guide/streaming-endpoints#streaming-responses-with-map-and-starmap)
## and the responses will appear out of order with `order_outputs=False`.
responses = []
prompt_tokens = 0
generated_tokens = 0

for response in tqdm(model.generate.map(prompts, order_outputs=False, kwargs={"max_tokens": 500, "eos_token_ids": [1,2]}), total=len(prompts)):
    prompt_tokens += response['prompt_tokens']
    generated_tokens += response['responses'][0]['generated_tokens']
    responses.append(response)

## Lastly, save the responses and print a summary.
# Print the first few JSON responses
for i, response in enumerate(responses[:8]):
    print(f"\n\n[{i}/{len(responses)}] PROMPT: {response['prompt']}\nRESPONSE: {response['responses'][0]['text']}")

# Save the responses
with open(f'summarization_results.pkl', 'wb') as file:
    pickle.dump(responses, file)

# Summary
print("\n\nSummary:")
print(f"Number of generation requests: {len(responses)}")
print(f"Prompt tokens: {prompt_tokens}")
print(f"Generated tokens: {generated_tokens}")

## ## Performance
## For the summarization task, we can compare the performance of Flywheel with vLLM by measuring how long it takes
## to process the entire batch of prompts. Simply put, with Flywheel we can process the entire batch in half the time
## on the same GPU. Practically, this halves the cost per request.
##
## Concretely for this example, it takes 416 seconds to process the 2048 articles with Flywheel, while it takes 898 seconds
## with vLLM. This is a flat **2.16x speedup**.
##
## > As of 2/5/2024, considering the [pricing of Flywheel on Modal](https://modal.com/pricing), processing 2048 news articles costs you \$0.48 with Flywheel,
## while it is \$0.93 with vLLM (bare metal cost on Modal). **This is over 45% savings in inference cost for the same task**.
##
## ```{image} ./summarization_n2048_performance.png
## :class: bg-primary
## :width: 75%
## :align: center
## ```
## \
## An additional evaluation a small batch of 64 prompts also shows the same trend. Flywheel processes the batch in 17 seconds,
## compared to 27 seconds with vLLM for a 1.6x speedup. As indicated before, the throughput is greater with
## larger workloads. For a task with a large batch of prompts, Flywheel optimally schedules the tasks for the best overall latency.
## The idea is that once you get the Flywheel spinning at full throttle 🏎️, it shreds through inference tasks.
##
## ```{image} ./summarization_n64_performance.png
## :class: bg-primary
## :width: 75%
## :align: center
## ```
## \
## Lastly, a look at the instantaneous **tokens per second** (rolling window average) for both systems cements the fact that Flywheel has a higher throughput.
## The following plot shows a rolling window measurement of the throuhput for generated tokens for the 2048 batch of tasks.
##
## ```{image} ./summarization_n2048_throughput.png
## :class: bg-primary
## :width: 75%
## :align: center
## ```
##