from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments, pipeline

import math
import pandas as pd

# df1 = pd.read_csv('data/clean-ushmm.csv')
df2 = pd.read_csv('data/wiener_novermberdown_2.csv')
df3 = pd.read_csv('data/yale-good.csv')

half_length1 = len(df2) // 2
half_length2 = len(df3) // 2

# Take training half of the data from each DataFrame
half_df1 = df2.iloc[:half_length1]
half_df2 = df3.iloc[:half_length2]
# train_df = pd.concat([half_df1['text'], half_df2['text'], df1['text']], axis=0, ignore_index=True)
train_df = pd.concat([half_df1['text'], half_df2['text']] , axis=0, ignore_index=True)

header = 'text'
combined_series = train_df.rename(header)
train_set = pd.DataFrame(combined_series)
# train_set = train_set.dropna()
print(train_set)

# testing half
# second_half_df = df2.iloc[half_length1:]
# third_half_df = df3.iloc[half_length2:]
# test_df = pd.concat([second_half_df['text'], third_half_df['text']], axis=0, ignore_index=True)
# header = 'text'
# combined_series = test_df.rename(header)
# test_set = pd.DataFrame(combined_series)

# pretraining the model
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

# def preprocess_function(examples):
#     return tokenizer(examples, padding=True, truncation=True)
# tokenised_data = train_set.apply(preprocess_function, axis=1)

# Tokenize the text from the DataFrame column individually
tokenized_train_data = []
for example in train_set['text']:
    tokens = tokenizer(example, padding=True, truncation=True)
    tokenized_train_data.append(tokens)
#
# tokenized_test_data = []
# for example in test_set['text']:
#     tokens = tokenizer(example, padding=True, truncation=True)
#     tokenized_test_data.append(tokens)

input_ids_train = tokenized_train_data
# input_ids_test = tokenized_test_data

# print(tokenized_data)

# block_size = 128
# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = [item for sublist in examples for item in sublist]
#     total_length = len(concatenated_examples)
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#     # customize this part to your needs.
#     if isinstance(total_length, int) and total_length >= block_size:
#         total_length = (total_length // block_size) * block_size
#     # Split by chunks of block_size.
#     result = [concatenated_examples[i : i + block_size] for i in range(0, total_length, block_size)]
#     return result
#
# processed_data = []
#
# # Apply 'group_texts' to each text example in the list
# for text_example in tokenized_data:
#     processed_text = group_texts(text_example)
#     processed_data.append(processed_text)
# #
# print(processed_data)
#

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")

training_args = TrainingArguments(
    output_dir='./mlm_model',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,  # Pass DataCollatorForLanguageModeling
    train_dataset=input_ids_train,  # Pass the tokenized text data directly
    # eval_dataset=input_ids_test,
)

trainer.train()
#
#
# eval_results = trainer.evaluate()
# print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# Use the trained model to predict masked words
fill_mask = pipeline(task='fill-mask', model=model, tokenizer=tokenizer)
results = fill_mask("The [MASK] brown fox jumps over the lazy dog.")

for result in results:
    print(f"Predicted word: {result['token_str']} (Score: {result['score']:.4f})")