import pandas as pd

df1 = pd.read_csv('../data/wiener.csv')

df1 = pd.DataFrame({'document_id': df1['document_id'], 'words': df1['words'], 'labels': df1['labels']})

sentence_id_list = []

sentence_id_seq = 1342138

dropping_sentences = []

for word in df1['words'].tolist():
    if word == "." or word == "?" or word == "!":
        sentence_id_list.append(sentence_id_seq)
        sentence_id_seq += 1
        word_count = 0
    else:
        sentence_id_list.append(sentence_id_seq)

df1['sentence_id'] = sentence_id_list
df1.to_csv('wiener.csv', sep='\t')
#

import pandas as pd

# Read the text file as a single column dataframe
# Read the text file line by line
# with open('../data/ushmm.txt', 'r') as file:
#     lines = file.readlines()
#
# # Process each line to split it into columns
# data = []
# for line in lines:
#     fields = line.strip().split(',')
#     if len(fields) == 3:
#         data.append(fields)
#
# # Create a DataFrame from the processed data
# df = pd.DataFrame(data, columns=['document_id', 'word', 'label'])
#
# # Write the data to a CSV file
# df.to_csv('us.csv', index=False)