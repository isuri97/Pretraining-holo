from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments, pipeline
import torch
from torch.utils.data import DataLoader, Dataset
import math
import pandas as pd
import argparse


parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
arguments = parser.parse_args()

df1 = pd.read_csv('data/clean-ushmm.csv')
df2 = pd.read_csv('data/wiener_novermberdown_2.csv')
df3 = pd.read_csv('data/yale-good.csv')

half_length1 = len(df2) // 2
half_length2 = len(df3) // 2

# Take training half of the data from each DataFrame
half_df1 = df2.iloc[:half_length1]
half_df2 = df3.iloc[:half_length2]
train_df = pd.concat([half_df1['text'], half_df2['text'], df1['text']], axis=0, ignore_index=True)
# train_df = pd.concat([half_df1['text'], half_df2['text']] , axis=0, ignore_index=True)

header = 'text'
combined_series = train_df.rename(header)
train_set = pd.DataFrame(combined_series)
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
# tokenized_train_data = []
# for example in train_set['text']:
#     tokens = tokenizer(example, padding=True, truncation=True)
#     tokenized_train_data.append(tokens)
# #
# tokenized_test_data = []
# for example in test_set['text']:
#     tokens = tokenizer(example, padding=True, truncation=True)
#     tokenized_test_data.append(tokens)

# input_ids_train = tokenized_train_data
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
def split_text_into_chunks(text, max_chunk_length):
    if isinstance(text, str):  # Check if the value is a string and not NaN/None
        chunks = []
        for i in range(0, len(text), max_chunk_length):
            chunks.append(text[i:i + max_chunk_length])
        return chunks
    else:
        return []

# Initialize a tokenizer and model (you can replace 'bert-base-uncased' with any other model)
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')


# Tokenize and process each text chunk in the DataFrame
def tokenize_and_process_text(text):
    text_chunks = split_text_into_chunks(text, max_sequence_length)
    tokenized_chunks = []

    for text_chunk in text_chunks:
        tokenized_chunk = tokenizer.encode(text_chunk, add_special_tokens=True, max_length=max_sequence_length,
                                           truncation=True)
        tokenized_chunks.append(tokenized_chunk)

    return tokenized_chunks


# Define the maximum sequence length for the model
max_sequence_length = 512

# Apply tokenization and processing to the 'text' column
train_set['tokenized_text'] = train_set['text'].apply(tokenize_and_process_text)

# Flatten the 'tokenized_text' column into a single list of tokens
tokenized_text = [token for chunk in train_set['tokenized_text'] if chunk for token in chunk]

# Create a custom dataset for MLM training
class CustomDataset(Dataset):
    def __init__(self, tokenized_text, tokenizer, max_sequence_length):
        self.inputs = tokenized_text
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),  # All tokens are attended to
        }

# Create a DataLoader for the custom dataset
dataset = CustomDataset(tokenized_text, tokenizer, max_sequence_length)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator)

cuda_device = int(arguments.cuda_device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./mlm_model",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust as needed
    per_device_train_batch_size=4,  # Adjust as needed
    save_steps=10,  # Adjust as needed
    save_total_limit=2,  # Adjust as needed
    logging_dir="./logs",
    logging_steps=10,  # Adjust as needed
)

# Create a DataLoader for training
train_dataloader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator)

# Initialize a Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    cuda_device=cuda_device,
    data_collator=data_collator,
    train_dataset=train_dataloader  # Use the DataLoader for training
)

trainer.train()

# Use the trained model to predict masked words
fill_mask = pipeline(task='fill-mask', model=model, tokenizer=tokenizer)
results = fill_mask("The [MASK] brown fox jumps over the lazy dog.")

for result in results:
    print(f"Predicted word: {result['token_str']} (Score: {result['score']:.4f})")


# tokenized_train_data = []
# for example in train_set['text']:
#     tokens = tokenizer(example, padding=True, truncation=True)
#     tokenized_train_data.append(tokens)
