from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
import math
import pandas as pd

df1 = pd.read_csv('clean-ushmm.csv')
df2 = pd.read_csv('wiener_novermberdown_2.csv')
df3 = pd.read_csv('yale-good.csv')


half_length1 = len(df2) // 2
half_length2 = len(df3) // 2

# Take training half of the data from each DataFrame
half_df1 = df1.iloc[:half_length1]
half_df2 = df2.iloc[:half_length2]
train_df = pd.concat([half_df1['text'], half_df2['text'], df1['text']], axis=0, ignore_index=True)
header = 'text'
combined_series = train_df.rename(header)
train_set = pd.DataFrame(combined_series)
train_set

# testing half
second_half_df = df2.iloc[half_length1:]
third_half_df = df3.iloc[half_length2:]
test_df = pd.concat([second_half_df['text'], third_half_df['text']], axis=0, ignore_index=True)
header = 'text'
combined_series = test_df.rename(header)
test_set = pd.DataFrame(combined_series)



# pretraining the model
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

def preprocess_function(examples):
  return tokenizer([" ".join(x) for x in train_set['text']])

tokenised_data = train_set.apply(preprocess_function, axis=1)

block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

lm_dataset = tokenised_data.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")

training_args = TrainingArguments(
    output_dir="my_awesome_eli5_mlm_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()


eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")