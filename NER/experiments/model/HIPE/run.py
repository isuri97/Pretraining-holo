import argparse

import pandas as pd
from sklearn import metrics
import torch
from simpletransformers.config.model_args import NERArgs
from simpletransformers.ner import NERModel

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert-base-cased")
parser.add_argument('--model_type', required=False, help='model type', default="bert")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=3)
parser.add_argument('--train', required=False, help='train file', default='data/sample.txt')

arguments = parser.parse_args()

test_df = pd.read_csv('new/test_df_hipe.csv', sep='\t', usecols=['words','labels','sentence_id'])
train_df = pd.read_csv('new/train_df_hipe', sep='\t', usecols=['words','labels','sentence_id'])
val_df = pd.read_csv('new/val_df_hipe.csv', sep='\t', usecols=['words','labels','sentence_id'])

train_df = train_df.dropna(subset=['sentence_id'])
train_df = train_df.dropna(subset=['words'])
train_df = train_df.dropna(subset=['labels'])

test_df = test_df.dropna(subset=['sentence_id'])
test_df = test_df.dropna(subset=['words'])
test_df = test_df.dropna(subset=['labels'])

val_df = val_df.dropna(subset=['sentence_id'])
val_df = val_df.dropna(subset=['words'])
val_df = val_df.dropna(subset=['labels'])

model_args = NERArgs()
model_args.train_batch_size = 64
model_args.eval_batch_size = 64
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 3
model_args.use_multiprocessing = False
model_args.save_best_model=False
model_args.use_multiprocessing_for_evaluation = False
model_args.classification_report = True
model_args.evaluate_during_training = False
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
                          'B-FOREST', 'I-FOREST', 'B-GROUP', 'I-GROUP', 'B-MOUNTAIN', 'I-MOUNTAIN', 'I-BUILDING', 'B-BUILDING', 'B-WORK', 'I-WORK', 'B-SCOPE', 'I-SCOPE',
                          'B-loc', 'I-loc']

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
# model.save_model()
print(len(test_df))

results, outputs, preds_list, truths, preds = model.eval_model(test_df)
print(results)
preds_list = [tag for s in preds_list for tag in s]
ll = []
key_list = []

print(truths)
print(preds)
# test_df['labels'] = truths
# test_df['predicted_set'] = preds

# take the label and count is it match with
labels = ['B-SHIP', 'I-SHIP','B-GHETTO', 'I-GHETTO', 'B-STREET', 'I-STREET', 'B-MILITARY', 'I-MILITARY', 'B-DATE', 'I-DATE', 'B-PERSON', 'I-PERSON',
          'B-GPE', 'I-GPE', 'B-TIME', 'I-TIME', 'B-EVENT', 'I-EVENT', 'B-ORG', 'I-ORG', 'B-TIME', 'I-TIME','I-BUILDING', 'B-BUILDING', 'B-WORK', 'I-WORK', 'B-SCOPE', 'I-SCOPE',
          'B-loc', 'I-loc']

print(truths)
print(preds)


classification_report_str = metrics.classification_report(truths,preds,digits=4)

with open('hipe-output.txt', 'w') as output_file:
    output_file.write(classification_report_str)

