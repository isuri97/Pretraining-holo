import pandas as pd

import re

df1 = pd.read_csv('../pretraining/data/clean-ushmm.csv')
df2 = pd.read_csv('../pretraining/data/wiener_novermberdown_2.csv')
df3 = pd.read_csv('../pretraining/data/yale-good.csv')

half_length1 = len(df2) // 2
half_length2 = len(df3) // 2

# testset
# second_half_df = df2.iloc[half_length1:]
# third_half_df = df3.iloc[half_length2:]
# test_df = pd.concat([second_half_df['text'], third_half_df['text']], axis=0, ignore_index=True)
# header = 'text'
# combined_series = test_df.rename(header)
# test_df = pd.DataFrame(test_df)
# # Define a regular expression pattern to match and remove the pattern \n\n[00:00:02.88]
# pattern = r'\n\n\[.*?\]'
# pattern1 = r'SUBJECT:|INTERVIEWER:|INTERVIEWER 1:|SUBJECT 1:'
# # Use str.replace to remove content inside square brackets
# test_df['text'] = test_df['text'].str.replace(pattern, '', regex=True)
# test_df['text'] = test_df['text'].str.replace(pattern1, '', regex=True)
# # Print the DataFrame with content inside brackets removed
# test_df.to_csv('test_set.csv')

# training set
half_df1 = df2.iloc[:half_length1]
half_df2 = df3.iloc[:half_length2]
train_df = pd.concat([half_df1['text'], half_df2['text'], df1['text']], axis=0, ignore_index=True)
train_df = pd.concat([half_df1['text'], half_df2['text']] , axis=0, ignore_index=True)
header = 'text'
combined_series = train_df.rename(header)
train_set = pd.DataFrame(combined_series)

pattern = r'\n\n\[.*?\]'
pattern1 = r'SUBJECT:|INTERVIEWER:|INTERVIEWER 1:|SUBJECT 1:'
# Use str.replace to remove content inside square brackets
train_set['text'] = train_set['text'].str.replace(pattern, '', regex=True)
train_set['text'] = train_set['text'].str.replace(pattern1, '', regex=True)
print(train_set)

train_set.to_csv('train-set.csv')