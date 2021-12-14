#!/usr/bin/env python3

import argparse
import glob
import keras
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import tensorflow as tf

from collections import defaultdict
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sentence_transformers import SentenceTransformer

def load_data(filenames):
  DF = [pq.read_table(f).to_pandas() for f in filenames]
  df = pd.concat(DF).set_index('id')
  return df

def extract_features(df,bs=16):
  smodel = SentenceTransformer('stsb-xlm-r-multilingual')
  feats = []
  for k,g in df.groupby(np.arange(len(df))//bs): # split into bs-sized chunks
    feats.extend(smodel.encode(g['text'].values))
  df['feat'] = feats
  return df

def train_model(Xtrn, Ytrn, depth=1, bs=100, epochs=100, weights=None):
  _,dim = Xtrn.shape
  model = Sequential(Dense(1, input_dim=dim, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy','AUC'])
  model.fit(Xtrn, Ytrn, batch_size=bs, epochs=epochs, sample_weight=weights, verbose=0) # validation_split=0.05, 
  return model

def do_train(df, suffix='multi', depth=1, model_filepattern=None, min_size=2):
  if len(df) < min_size:
    print("WARNING: not enough data to train for ", suffix, file=sys.stderr)
    return None

  Xtrn=np.vstack(df['feat'])
  Ytrn=np.hstack(df['label'])
  model = train_model(Xtrn, Ytrn, depth=depth)

  if model_filepattern:
    model_filename = model_filepattern.format(suffix)
    model.save(model_filename)
  return model

def do_eval(model, df, min_size=1):
  if len(df) < min_size:
    print("WARNING: not enough data to evaluate", file=sys.stderr)
    return None
  Xtst=np.vstack(df['feat'])
  Ytst=np.hstack(df['label'])
  res = model.evaluate(Xtst, Ytst)
  return res

def do_prediction(model, df, min_size=1):
  if len(df) < min_size:
    pred = np.empty(len(df))
    pred[:] = np.nan
  else:
    Xtst = np.vstack(df['feat'])
    pred = model.predict(Xtst).ravel()
  return pred
  
def parse_args():
  parser = argparse.ArgumentParser(description='Train sentiment classifiers from pre-extracted features')
  parser.add_argument('-o', '--output', type=str, default=None, help='Results output filenames')
  parser.add_argument('-p', '--pred', type=str, default=None, help='Predictions output filenames')
  parser.add_argument('-m', '--model', type=str, default=None, help='Model output filename')
  parser.add_argument('-M','--multionly', action='store_true', help='Only train multi-lingual model, skip per-language traning and evaluation (default: false)')
  parser.add_argument('-s','--seed', type=int, default=0, help='Random seed (default: 0)')
  parser.add_argument('files', metavar='files', type=str, nargs='+', help='List of file names to be processed')
  args = parser.parse_args()
  return args

def main():
  args = parse_args()

  np.random.seed(args.seed)
  tf.random.set_seed(args.seed)
  
  if args.model:
    model_filepattern = args.model.replace('.hdf5','')+'-{}.hdf5'
    print("Using model_filepattern=", model_filepattern)
  else:
    model_filepattern = None

  print("Loading data")  
  df = load_data(args.files)
  print("Extracting Sentence Embeddings (this may take a while, especially without GPU.)")
  df = extract_features(df)
  print("Now starting the training")
  
  all_languages = ['multi']+list(df['lang'].unique())
  
  df_trn, df_tst = train_test_split(df, test_size=0.5, random_state=args.seed)
  
  # prepare dataframe in which to store results
  results = pd.DataFrame(columns=['lang','ntrn','ntst','acc','acc-multi','auc','auc-multi'])
  results['lang'] = all_languages
  results.set_index('lang',inplace=True)
  
  results.loc['multi',['ntrn','ntst']] = [len(df_trn),len(df_tst)]

  # train MULTILINGUAL (multi) model
  print("Train and eval for multi model")
  model_multi = do_train(df_trn, suffix='multi', model_filepattern=model_filepattern)
  if not model_multi:
    print("ERROR: Not enough data to train. This should not happen. len(df_trn)=", len(df_trn), file=sys.stderr)
    raise SystemExit 
  
  # evaluate multi model across all test data
  res = do_eval(model_multi, df_tst)
  print("Res=", res)
  if not res: # less than 10 points in test set? Strange!
    print("ERROR: Not enough data to evaluate. This should not happen. len(df_tst)=", len(df_tst), file=sys.stderr)
    raise SystemExit 
  results.loc['multi',['acc','acc-multi','auc','auc-multi']] = [res[1],res[1],res[2],res[2]]

  if not args.multionly:
    # evaluate multi model on each language seperately
    for lang, dfl_tst in df_tst.groupby('lang'):
      res = do_eval(model_multi, dfl_tst)
      if not res:
        print("WARNING: Not enough data to eval multi model on language ",lang, len(dfl_tst), file=sys.stderr)
        continue
      results.loc[lang, ['acc-multi','auc-multi']] = [res[1],res[2]]
    
    # compute and store individual predictions if requested
    if args.pred:
      pred = do_prediction(model_multi, df_tst)
      df_tst['pred-multi'] = pred
      df_tst['pred'] = np.nan # will be filled later

    # train one MONOLINGUAL model per language
    for lang, dfl_tst in df_tst.groupby('lang'):
      print("Train and eval for language", lang)
      dfl_trn = df_trn[df_trn['lang']==lang]
    
      results.loc[lang, ['ntrn','ntst']] = [len(dfl_trn),len(dfl_tst)]

      # train on training data of this language data
      model_lang = do_train(dfl_trn, suffix=lang, model_filepattern=model_filepattern)
      if not model_lang:
        print("WARNING: Did not get a model for language", lang)
        continue

      # eval on test data of this language
      res = do_eval(model_lang, dfl_tst)
      if not res:
        print("WARNING: Not enough data to eval monolingual model on language ",lang, len(dfl_tst), file=sys.stderr)
        continue
      results.loc[lang, ['acc','auc']] = [res[1],res[2]]
      
      if args.pred:
        pred = do_prediction(model_lang, dfl_tst)
        df_tst.loc[df_tst['lang']==lang, 'pred'] = pred
  
  # store results if requested
  if args.output:
    results.to_json(args.output)
  else:
    print("\n=========================RESULTS======================")
    print(results)

  # store predictions if requested
  del df_tst['feat'] # remove features before saving results
  if args.pred:
    df_tst.to_json(args.pred)
  
if __name__ == "__main__":
  main()
