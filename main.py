import time
import torch
import torch.nn.functional as F
from model_config import ModelConfig
from model import LanguageModel
from random import uniform
from IPython.display import display, clear_output
from torchsummaryX import summary

def main():
    sentences = input("Enter the sentences: ")
    max_tokens = int(input("Enter the maximum number of tokens: "))
    top_p = float(input("Enter the top_p value: "))

    config.set_config(sentences, max_tokens, top_p)

    print(config)

if __name__ == "__main__":
    main()