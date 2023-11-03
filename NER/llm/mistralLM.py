# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# import transformers
# import torch
# import pandas as pd
#
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )
#
# model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
#
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         max_memory={0:'20GIB'},
#         trust_remote_code=True,
#     )
#
# # model = AutoModelForCausalLM.from_pretrained(
# #         model_name,
# #         torch_dtype=torch.bfloat16,
# #         device_map="auto",
# #         max_memory={0:'20GIB'},
# #         trust_remote_code=True,
# #     )
#
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer = tokenizer,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     max_memory={1:'20GIB'},
# )
#
#
# data = pd.read_csv('data1.csv', sep=",")
#
#
# for index, row in data.iterrows():
#   text = row['tags']
#
#   prompt = f"""Return a only list of named entities in the text. No need of additional information. Provide only
#           Text: ```{text}```
#           Named entities:
#
# """
#   sequences = pipe(
#       prompt,
#       do_sample=True,
#       top_k=10,
#       num_return_sequences=1,
#       eos_token_id=tokenizer.eos_token_id,
#       max_length=200,
#   )
#
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import transformers
import torch
import pandas as pd

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_name = 'mistralai/Mistral-7B-Instruct-v0.1'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "20GIB", 1: "20GIB"},
    trust_remote_code=True,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

data = pd.read_csv('text-sent.csv', sep="\t")

for index, row in data.iterrows():

    doc_id = row['doc_id']
    text = row['sentences']
    # prompt = f"""
    # Think you are a historian and you are supposed to find named entities and relationships in holocaust text.
    # - First identify the named entities with their named entity tags of the given text delimited by ```
    # - Print the named entities only
    # holocaust text : ```{text}```"""

    prompt = f"""Return a list of named entities in the given text. 
          Text: ```{text}```
          Named entities:
          
"""
    sequences = pipe(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=500,
    )

    for i, seq in enumerate(sequences):
        print(sequences)
        # result_text = seq['generated_text']
        # # Define the output filename based on doc_id
        # output_filename = f'results_{doc_id}_{i}.txt'
        #
        # # Save the result to the output file
        # with open(output_filename, 'w') as output_file:
        #     output_file.write(result_text)
