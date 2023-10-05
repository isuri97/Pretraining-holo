import pandas as pd
import torch.cuda
from sklearn import metrics
# from sklearn.model_selection import train_test_split
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
df_train = pd.concat([df_train, us_data], axis=0, ignore_index=True)

print(f'training set size {len(df_train)}')
print(f'test set size {len(df_test)}')

print(df_train)
#
model_args = NERArgs()
model_args.train_batch_size = 64
model_args.eval_batch_size = 64
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 1
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.classification_report = True
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

model.train_model(df_train)
model.save_model()

print(len(df_test))
results, outputs, preds_list, truths, preds = model.eval_model(df_test)
print(results)
preds_list = [tag for s in preds_list for tag in s]
ll = []
key_list = []

print(truths)
print(preds)
df_test['original_test_set'] = truths
df_test['predicted_set'] = preds

# take the label and count is it match with
labels = ['B-SHIP', 'I-SHIP','B-GHETTO', 'I-GHETTO', 'B-STREET', 'I-STREET', 'B-MILITARY', 'I-MILITARY', 'B-DATE', 'I-DATE', 'B-PERSON', 'I-PERSON',
          'B-GPE', 'I-GPE', 'B-TIME', 'I-TIME', 'B-EVENT', 'I-EVENT', 'B-ORG', 'I-ORG', 'B-TIME', 'I-TIME']

print(truths)
print(preds)

print(metrics.classification_report(truths,preds,digits=4))

