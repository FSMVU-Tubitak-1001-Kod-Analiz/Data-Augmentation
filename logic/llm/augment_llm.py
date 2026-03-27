import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def llama_create():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct").to("cuda:0")
    model.eval()

    smell_df = pd.read_csv("unique_data_setV3.csv")

    set_seed(42)

    results = []
    base_prompt = """Produce 4 distinct refactorings of the following code. Don't introduce comments. Don't use markdown. No need for any notes or explanations. Separate your refactorings with a <sep> symbol.\n{}"""
    batch_texts = []
    batch_metadata = []
    batch_size = 1
    batch_index = 0

    for idx, item in tqdm(smell_df.iterrows(), total=len(smell_df)):
        # Prepare message with chat template
        messages = [{"role": "user", "content": base_prompt.format(item["function"])}]

        # Apply chat template to get formatted text
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False  # Get text first, tokenize in batch
        )

        batch_texts.append(text)
        batch_metadata.append((item["smellKey"], idx))

        # Process batch when full or at end
        if len(batch_texts) == batch_size or idx == smell_df.index[-1]:
            # Tokenize the batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=False
            ).to(model.device)

            # Generate with fixed max_new_tokens
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1000,  # Fixed value, not dynamic
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,  # Deterministic generation
                    temperature=None,  # Disable if do_sample=False
                )

            # Decode each output
            for i, (output, (smell_key, original_idx)) in enumerate(zip(outputs, batch_metadata)):
                # Find where the input ends (skip padding)
                input_length = len(inputs["input_ids"][i])

                # Decode only the generated part
                generated_text = tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True
                )

                results.append((generated_text, smell_key, original_idx))

            batch_index += 1
            result_df = pd.DataFrame(results, columns=["function", "smellKey", "original_function_idx"])
            result_df.to_csv(f"results2/{batch_index}.csv")

            # Clear batch for next iteration
            results = []
            batch_texts = []
            batch_metadata = []

            del outputs
            del inputs
            torch.cuda.empty_cache()


if __name__ == '__main__':
    llama_create()
