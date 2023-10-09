import pandas as pd

# train_hipe = pd.read_csv('data/new-hipe-train.csv', sep='\t')
# test_hipe = pd.read_csv('data/new_test.csv', sep='\t')
# dev_hipe = pd.read_csv('data/new_dev.csv', sep='\t')
#
# df_data = pd.concat([train_hipe, test_hipe, dev_hipe], ignore_index=True)
# print(df_data)
#
# df_data.to_csv('new_data.csv', index=False, sep='\t', columns=['TOKEN','NE-COARSE-LIT'])
#
# new_data = pd.read_csv('new_data.csv', sep='\t')
# print(new_data)
# # new_data
# # #
# # #
# sentence_id_seq = 0
#
# dropping_sentences = []
# sentence_id_list = []
#
# for word in new_data['word'].tolist():
#     if word == "." or word == "?" or word == "!":
#         sentence_id_list.append(sentence_id_seq)
#         sentence_id_seq += 1
#         word_count = 0
#     else:
#         sentence_id_list.append(sentence_id_seq)
#
# new_data['sentence_id'] = sentence_id_list
# new_data.to_csv('data/new-hipe-data.csv', index=False, sep='\t')

from sklearn.model_selection import train_test_split

df = pd.read_csv('data/new-hipe-data.csv', sep='\t')
# print(df)

train_ratio = 0.7
test_ratio = 0.15
dev_ratio = 0.15

sentence_id_list = df['sentence_id']
sentence_ids = list(set(sentence_id_list))
train_data, temp_data = train_test_split(sentence_ids, test_size=test_ratio + dev_ratio)

df_train_filtered = df[df["sentence_id"].isin(train_data)]
df_val = df[df["sentence_id"].isin(temp_data)]


sentence_id_list_dev = df_val['sentence_id']
sentence_ids_dev = list(set(sentence_id_list_dev))
test_data, dev_data = train_test_split(sentence_ids_dev, test_size=dev_ratio / (test_ratio + dev_ratio))

df_test = df[df["sentence_id"].isin(test_data)]
df_val = df[df["sentence_id"].isin(dev_data)]

print(df_test)

df_train_filtered.to_csv('new/train_df_hipe', sep='\t')
df_test.to_csv('new/test_df_hipe.csv', sep='\t')
df_val.to_csv('new/val_df_hipe.csv', sep='\t')
