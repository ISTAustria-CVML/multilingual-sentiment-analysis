#!/usr/bin/env python3
"""Minimal example of how to use the original sentiment 
   analysis model (sentence embedding plus linear classifier)."""

import numpy as np
from sentence_transformers import SentenceTransformer

class Classifier():
  def __init__(self):
    self.smodel = SentenceTransformer('stsb-xlm-r-multilingual')
    params = np.loadtxt("parameters.txt")
    self.w = params[:-1]
    self.b = params[-1]

  def __call__(self, txt):
      vec = self.smodel.encode(txt)
      score = np.dot(vec, self.w)+self.b
      return score

def main():
  print("Preparing model")
  cls = Classifier()
  
  print("Please enter statements to be classified. End with CTRL-D")
  while True:
    txt = input()
    print("Sentiment score: ", cls(txt))

if __name__ == "__main__":
  main()
