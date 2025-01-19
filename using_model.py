from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
  "SeaLLMs/SeaLLMs-v3-7B-Chat", # can change to "SeaLLMs/SeaLLMs-v3-1.5B-Chat" if your resource is limited
  torch_dtype=torch.bfloat16, 
  device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("SeaLLMs/SeaLLMs-v3-7B-Chat")

# prepare messages to model
prompt = "Hiii How are you?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
print(f"Formatted text:\n {text}")
print(f"Model input:\n {model_inputs}")

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True, eos_token_id=tokenizer.eos_token_id)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(f"Response:\n {response[0]}")
