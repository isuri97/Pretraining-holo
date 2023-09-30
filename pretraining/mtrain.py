# Define model parameters to train BERT model from scratch
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, LineByLineTextDataset
from transformers import AutoTokenizer
import pandas as pd
import torch
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM

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



# Read the vocabulary file
# vocab_file_dir = 'phone_review-vocab.txt'

# custom_tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)
custom_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# sentence = 'Motorola V860 is a good phone'
# encoded_input = custom_tokenizer.tokenize(sentence)
# print(encoded_input)
#

# Convert input text data to tokens for custom bert model

# Define a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, block_size=128):
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        text = self.data[i]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.block_size,
            return_tensors='pt'  # Return PyTorch tensors
        )
        return {
            "input_ids": encoding.input_ids.squeeze(),  # Remove the extra dimension
            "attention_mask": encoding.attention_mask.squeeze(),  # Remove the extra dimension
        }


# Create an instance of your custom dataset
dataset = CustomDataset(tokenizer=custom_tokenizer, data=train_set['text'].tolist(), block_size=128)

# Print the number of examples
print('No. of examples: ', len(dataset))

model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
print('No of parameters: ', model.num_parameters())

data_collator = DataCollatorForLanguageModeling(
    tokenizer=custom_tokenizer, mlm=True, mlm_probability=0.15
)

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

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
trainer.save_model('custom_bert_output/')

# sentence = 'Motorola V860 is a good phone'
# encoded_input = custom_tokenizer.tokenize(sentence)
# print(encoded_input)
#


# config = BertConfig(
#     vocab_size=30522,
#     hidden_size=768,
#     num_hidden_layers=6,
#     num_attention_heads=12,
#     max_position_embeddings=512
# )
#
# model = BertForMaskedLM(config)
# print('No of parameters: ', model.num_parameters())
#
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=custom_tokenizer, mlm=True, mlm_probability=0.15
# )
# training_args = TrainingArguments(
#     output_dir='custom_bert_output/',
#     overwrite_output_dir=True,
#     num_train_epochs=30,
#     per_device_train_batch_size=16,
#     save_steps=10_000,
#     save_total_limit=2,
#     prediction_loss_only=True,
#     report_to="none"
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=dataset
# )
#
# trainer.train()
# trainer.save_model('custom_bert_output/')