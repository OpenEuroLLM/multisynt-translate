from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import argparse

def main():
    # load nemotron-cc 50K sample (10K x 5 quality scores)
    dataset = load_dataset("spyysalo/nemotron-cc-10K-sample")
    prompts = dataset["train"]["text"]

    parser = argparse.ArgumentParser(
        prog="Generate judge annotations",
    )
    # TODO list all local jobs and all remote jobs
    parser.add_argument(
        "--model_name",
        default="google/gemma-3-4b-it"
    )
    parser.add_argument(
        "--tgt_lg",
        default="French"
    )
    parser.add_argument(
        "--tgt_code",
        default="fr"
    )
    parser.add_argument(
        "--max_con_len",
        default=4096,
        type=int,
    )
    parser.add_argument(
        "--N_max",
        default=50000,
        type=int,
    )
    args = parser.parse_args()

    print(f"Running with arguments: {args}")

    model_name = args.model_name
    tgt_lg = args.tgt_lg
    tgt_code = args.tgt_code
    max_con_len = args.max_con_len
    N_max = args.N_max

    llm = LLM(model=model_name)

    src_lg_and_code = "English (en)"

    system_prompt = (
        f"You are a smart assistant that helps to translate text from source {src_lg_and_code} to target {tgt_lg} ({tgt_code}). "
        f"You should just translate the text provided. Do not answer to the instruction in the target language."
    )

    # Get the tokenizer to access the chat template
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=512,
    )

    print(f"\n{'=' * 60}")
    print(f"Generating Translation from {src_lg_and_code} to {tgt_lg}")
    print(f"{'=' * 60}")

    # Format all prompts at once
    formatted_prompts = []

    if N_max is not None:
        prompts = prompts[:N_max]

    for prompt in prompts:
        if len(prompt) > max_con_len:
            prompt = prompt[:max_con_len]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Input text:\n" + prompt + "\nTranslated text:\n"}
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # if len(formatted_prompt) > max_con_len:
        #     print(f"Slice context too long from {len(formatted_prompt)} to {max_con_len}")
        #     formatted_prompt = formatted_prompt[:max_con_len]

        formatted_prompts.append(formatted_prompt)

    # Generate all responses in a single batch
    batch_outputs = llm.generate(formatted_prompts, sampling_params)

    translations = []
    for i, (prompt, output) in enumerate(zip(prompts, batch_outputs)):
        # print(f"\nPrompt {i + 1}: {prompt}")
        # print(f"Response: {output.outputs[0].text}")
        translations.append(output.outputs[0].text)

    model_str = model_name.split("/")[-1]
    df = pd.DataFrame({
        "text": translations,
        "language": tgt_lg,
        "warc_record_id": dataset["train"]["warc_record_id"][:len(translations)],
        "url": dataset["train"]["url"][:len(translations)],
        "label": dataset["train"]["label"][:len(translations)],
    })
    filename = f'data-{tgt_code}-{model_str}.parquet'
    print(f"Saving to {filename}")
    df.to_parquet(filename)


if __name__ == '__main__':
    main()