from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import transformers
import torch
import pandas as pd

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

model_name = 'mistralai/Mistral-7B-Instruct-v0.1'

tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         load_in_4bit=True,
#         quantization_config=bnb_config,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#     )

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        offload_folder="offload"
    )

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer = tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


data = pd.read_csv('data1.csv', sep=",")


for index, row in data.iterrows():
  text = row['tags']

  prompt = f"""Return a list of named entities in the text.
          Text: ```{text}```
          Named entities:
          Dont generate other useless information.
"""
  sequences = pipe(
      prompt,
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      max_length=200,
  )

for seq in sequences:
    print(f"Result: {seq['generated_text']}")