#!/usr/bin/env python3
"""Minimal example of how to use the improved sentiment 
   analysis model (finetuned deep network)."""

from transformers import pipeline

def main():
  print("Preparing model (might take a while)")
  cls = pipeline("text-classification", "clampert/multilingual-sentiment-covid19")
  
  print("Please enter statements to be classified. End with CTRL-D")
  while True:
    txt = input()
    print("Sentiment score: ", cls(txt))

if __name__ == "__main__":
  main()
