import os
import gc
import wandb
import random
import json
import string
import re
from tqdm import tqdm
from time import time
import warnings
import pandas as pd
import numpy as np
from collections import Counter
import string

# visuals
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

# env check
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords 
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn

# BERT
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import transformers
transformers.logging.set_verbosity_error()

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


STOPWORDS = set(stopwords.words('english'))
DEVICE = torch.device('cuda:0')

COMP_ID="Detect_Gen_Text"
CONFIG={'competition': COMP_ID, '_wandb_kernel': 'aot', "source_type": "artifact"}
