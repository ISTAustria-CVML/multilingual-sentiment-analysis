#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pyarrow as pa
import pyarrow.parquet as pq
import re
import sys

from collections import defaultdict
from itertools import islice

def parse_tweet(obj):
  try:
    if 'extended_tweet' in obj:
      txt = obj['extended_tweet']['full_text']
    else: 
      txt = obj['text']
  except KeyError:
    txt = None
  
  idstr = obj['id_str']
  lang = obj['lang']
  return idstr,txt,lang

try:
  filename = sys.argv[1]
except IndexError:
  filename = 'dummy.json'
  
positive_smileys = "😀😃😄😁😆😅🤣😂🙂😉😊😇🥰😍🤩😘😗😚😙🥲😋😛😜🤪😝🤗👍🤠🥳😎🤓👏" 
negative_smileys = "🤔🤐🤨😒🙄😬🤥🤢🤮🥵🥶🥴😵🤯😕😟🙁😮😯😲😳🥺😦😧😨😰😥😢😱😖😣😞😓😩😫🥱😤😡😠🤬😈👿🖕👎✊👊" 
positive_smileys_re = re.compile(u'['+positive_smileys+']')
negative_smileys_re = re.compile(u'['+negative_smileys+']')
all_smileys = positive_smileys+negative_smileys
all_smileys_re = re.compile(u'['+all_smileys+']')
 
positive_translation_table = str.maketrans("\n\r", "  ", positive_smileys) 
negative_translation_table = str.maketrans("\n\r", "  ", negative_smileys) 
  
def filter(fid, min_length=10):
  known_ids = set()
  for line in fid:
    if not line:
      continue
    try:
      obj = json.loads(line)
    except (json.decoder.JSONDecodeError, TypeError):
      #print("ERROR: entry wasn't a dictionary. skipping.", file=sys.stderr)
      continue

    try:
      if 'id_str' not in obj:
        print("ERROR: 'id' field not found in tweet", file=sys.stderr)
        continue
      if 'created_at' not in obj:
        print("ERROR: 'created_at' field not found in tweet {}".format(tweet['id']), file = sys.stderr)
        continue
      if 'retweeted_status' in obj:
        continue # skip retweets
    except TypeError:
        print("ERROR: not a dict?", line, obj, file=sys.stderr)
        continue 
    idstr, txt, lang = parse_tweet(obj)
    if not txt:
      continue
    if idstr in known_ids:
      continue
    known_ids.add(idstr)
    
    pos = re.findall(positive_smileys_re, txt)
    neg = re.findall(negative_smileys_re, txt)
    if not pos and not neg:
      continue # no smiley
    if pos and neg:
      continue # confusing
    if pos and not neg:
      txt = txt.translate(positive_translation_table)
      label = 1
    elif neg and not pos:
      txt = txt.translate(negative_translation_table)
      label = 0
    
    if len(txt)<min_length:
      continue

    smileys = ''.join(pos+neg)
    #print('"{} {}"'.format(smileys, label))
    yield (idstr, txt, label, lang, smileys)

data = defaultdict(list)

# load data and extract relevant information
bs=128 # batchsize for calling filter (probably not even needed here)
with open(filename, encoding='utf-8') as fid:
  while True:
    lines = list(islice(filter(fid), bs))
    if not lines:
      break
    try:
      ids,texts,labels,langs,smileys = list(zip(*lines))
    except ValueError:
      print(lines)
    data['text'].extend(texts)
    data['label'].extend(labels)
    data['id'].extend(ids)
    data['lang'].extend(langs)
    data['smileys'].extend(smileys)
    
try:
  output_filename = sys.argv[2]
except IndexError:
  output_filename = filename+'.parquet'

print("Writing to ",output_filename)
table = pa.Table.from_pydict(data)
pq.write_table(table, output_filename)
