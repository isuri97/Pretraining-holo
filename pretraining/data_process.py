import pandas as pd

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

train_df.to_csv('train-set.txt',index=False, header=False)

second_half_df = df2.iloc[half_length1:]
third_half_df = df3.iloc[half_length2:]
test_df = pd.concat([second_half_df['text'], third_half_df['text']], axis=0, ignore_index=True)
test_df.to_csv('test-set.txt', index=False, header=False)
