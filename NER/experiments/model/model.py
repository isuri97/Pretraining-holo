import pandas as pd
import torch.cuda
from sklearn import metrics
from sklearn.model_selection import train_test_split
import argparse
import csv
# from collections import Counter
#
from simpletransformers.ner import NERModel, NERArgs

from contextlib import redirect_stdout

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert-base-cased")
parser.add_argument('--model_type', required=False, help='model type', default="bert")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=3)
parser.add_argument('--train', required=False, help='train file', default='data/sample.txt')

arguments = parser.parse_args()


us_data = pd.read_csv('../../data/ushmm.csv', sep='\t', quoting=csv.QUOTE_ALL, encoding='utf-8')
wiener_data = pd.read_csv('../../data/wiener.csv', sep='\t', quoting=csv.QUOTE_ALL, encoding='utf-8')

df_train, df_test = [x for _, x in wiener_data.groupby(wiener_data['document_id'] >= 200)]
df_train = pd.concat([df_train, us_data], ignore_index=True)

df_train =  df_train[df_train['document_id'] < 2]
# document_ids = filtered_df['Document_ID'].tolist()
df_test = df_test[df_test['document_id'] < 201]

df_train = df_train.dropna(subset=['sentence_id'])
df_train = df_train.dropna(subset=['words'])
df_train = df_train.dropna(subset=['labels'])

df_test = df_test.dropna(subset=['sentence_id'])
df_test = df_test.dropna(subset=['words'])
df_test = df_test.dropna(subset=['labels'])

print(df_train.shape)
print(df_test.shape)

# reduce 'o' label from the test set
grouped_df = df_test.groupby('sentence_id').agg({'words': ' '.join, 'labels': ' '.join}).reset_index()
df_with_tags = grouped_df[grouped_df['labels'].str.contains('B')]
df_without_tags = grouped_df[~grouped_df['labels'].str.contains('B')]
# combine 10% of all 'o' labels
sampled_rows = df_without_tags.sample(frac=0.1, random_state=42)
df_test = pd.concat([df_with_tags, sampled_rows], ignore_index=True)

print(len(df_without_tags))
print(len(df_with_tags))

df_test['words'] = df_test['words'].str.split()
df_test['labels'] = df_test['labels'].str.split()
df_test = df_test.explode(['words', 'labels'], ignore_index=True)

print(df_test)

train_df, val_df = train_test_split(df_train, test_size=0.3)

train_df.to_csv('train_df.csv', sep='\t', index=False)
val_df.to_csv('val_df.csv', sep ='\t', index=False)
df_test.to_csv('test_df.csv', sep='\t', index=False)

print(f'training set size {len(df_train)}')
print(f'test set size {len(df_test)}')

print(train_df)
#
model_args = NERArgs()
model_args.train_batch_size = 64
model_args.eval_batch_size = 64
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 3
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.classification_report = True
model_args.evaluate_during_training=True
model_args.wandb_project="holo-ner"
model_args.labels_list = ['O', 'B-DATE', 'B-PERSON', 'B-GPE', 'B-ORG', 'I-ORG', 'B-CARDINAL', 'B-LANGUAGE',
                          'B-EVENT', 'I-DATE', 'B-NORP', 'B-TIME', 'I-TIME', 'I-GPE', 'B-ORDINAL', 'I-PERSON',
                          'B-MILITARY',
                          'I-MILITARY', 'I-NORP', 'B-CAMP', 'I-EVENT', 'I-CARDINAL', 'B-LAW', 'I-LAW', 'B-QUANTITY',
                          'B-RIVER',
                          'I-RIVER', 'B-PERCENT', 'I-PERCENT', 'B-WORK_OF_ART', 'I-QUANTITY', 'B-FAC', 'I-FAC',
                          'I-WORK_OF_ART',
                          'B-MONEY', 'I-MONEY', 'B-STREET', 'I-STREET', 'B-LOC', 'B-GHETTO', 'B-SEA-OCEAN',
                          'I-SEA-OCEAN',
                          'B-PRODUCT', 'I-CAMP', 'I-LOC', 'I-PRODUCT', 'I-GHETTO', 'B-SPOUSAL', 'I-SPOUSAL', 'B-SHIP',
                          'I-SHIP',
                          'B-FOREST', 'I-FOREST', 'B-GROUP', 'I-GROUP', 'B-MOUNTAIN', 'I-MOUNTAIN']

MODEL_NAME = arguments.model_name
MODEL_TYPE = arguments.model_type
cuda_device = int(arguments.cuda_device)
# MODEL_TYPE, MODEL_NAME,
model = NERModel(
    MODEL_TYPE, MODEL_NAME,
    use_cuda=torch.cuda.is_available(),
    cuda_device=cuda_device,
    args=model_args,
)

model.train_model(train_df,eval_df= val_df)
model.save_model()
print(len(df_test))

results, outputs, preds_list, truths, preds = model.eval_model(df_test)
print(results)
preds_list = [tag for s in preds_list for tag in s]
ll = []
key_list = []

print(truths)
print(preds)
df_test['labels'] = truths
df_test['predicted_set'] = preds

# take the label and count is it match with
labels = ['B-SHIP', 'I-SHIP','B-GHETTO', 'I-GHETTO', 'B-STREET', 'I-STREET', 'B-MILITARY', 'I-MILITARY', 'B-DATE', 'I-DATE', 'B-PERSON', 'I-PERSON',
          'B-GPE', 'I-GPE', 'B-TIME', 'I-TIME', 'B-EVENT', 'I-EVENT', 'B-ORG', 'I-ORG', 'B-TIME', 'I-TIME']

print(truths)
print(preds)


classification_report_str = metrics.classification_report(truths,preds,digits=4)

with open('output.txt', 'w') as output_file:
    output_file.write(classification_report_str)

