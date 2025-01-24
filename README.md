# LLM Course Equivalency Arbiter
CSC 869 Data Mining - Group 5 Project

## Getting Started
We used Jupyter Notebook to create the LLM pipeline to create an automated task for students and advisors to search and learn matching course equivalencies, and to create the Skill Extractor code.
We then used Google Colab Notebook to create embeddings and calculate cosine similarity. 

## Our Data
Our raw data files can be found in this link:
[Course Description Spreadsheet](https://github.com/mk-imagine/SFSU-CSC869-DataMining-GroupWork/blob/main/data/CSC869%20Term%20Project%20Dataset.xlsx)

In addition, we saved our progress with pickled dataframes and/or lists here:
[Pickled Data](https://github.com/mk-imagine/SFSU-CSC869-DataMining-GroupWork/tree/main/data)

The files needed to run the code are located in the data folder.

## Running Jupyter notebook for LangChain Pipe
Launch with:
```
jupyter notebook
```
The files we need is in the jupyter notebooks folder:
> langchain_pipe_csc869.ipynb
> skill_extraction_pipe.ipynb

### Installations
Since we are using pickle files, we need to import dill to open these binary files. 
We also need to import multiple modules/packages such as:

```
import dill
import pandas as pd
import numpy as np
from langchain.llms import Cohere, OpenAI, GooglePalm, Anthropic
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from typing import List
from tqdm.auto import tqdm
from pypdf import PdfReader
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
```
### Load Dataset
Load our original dataset and filter:
```
df = pd.read_excel(
    '../data/CSC869 Term Project Dataset.xlsx', 
    sheet_name='Course Descriptions', 
    skiprows=3, 
    names=['School', 'MATH226 eqv', 'MATH226 mult eqv', 'MATH226 alt eqv', 'MATH226 alt mult eqv', 'CSC230 eqv', 'CSC230 alt eqv', 'CSC256 eqv', 'CSC256 multipleEqv'],
    index_col=None
)
df = df.dropna(how='all').set_index("School")
```
### Initializing LLM
We initialize LLM for exploration and testing of topic extraction.
```
palm = GooglePalm(model_name="models/text-bison-001", temperature=0)
```
### Processing DataFrame
To process our dataframe, we need to import the following:
```
from tqdm.auto import tqdm
```
### Load new dataset
After processing and evaluations, we created a new dataset with expanded cases, which we will use for further processing and equivalency predictions.
```
with open('../data/course_desc_and_data.pkl', 'rb') as f:
    df2 = dill.load(f)
```

### Imports for Equivalency Predictions
After evaluating our evaluated dataset, we need to import the following for equivalency assessments:
```
from typing import Literal
```
### Imports for Evaluation of Equivalency Predictions
For evaluations, we need to import the classification report and confusion matrix from SciKit.
```
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDi
```

## Running Jupyter Notebook for Skill Extraction
The data we use for this code is our original dataset of the 62 colleges with the equivalent courses and descriptions. 
> CSC 869 Term Project Dataset.xlsx


### Installations
Prerequisites: Spacy, Scikit
```
# If not installed run these in command line.
pip install -U scikit-learn
pip install spacy
```
Import the following:
```
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")
```
### Load our original dataset:
```
df = pd.read_excel(
    '../data/CSC869 Term Project Dataset.xlsx', 
    sheet_name='Course Descriptions', 
    skiprows=3, 
    names=['School', 'MATH226 eqv', 'MATH226 mult eqv', 'MATH226 alt eqv', 'MATH226 alt mult eqv', 'CSC230 eqv', 'CSC230 alt eqv', 'CSC256 eqv', 'CSC256 multipleEqv'],
    index_col=None
)
```

### Installations for Lightcast API
```
import requests
url = "https://auth.emsicloud.com/connect/token"

payload = "client_id=CLIENT_ID&client_secret=CLIENT_SECRET&grant_type=client_credentials&scope=emsi_open"
headers = {'Content-Type': 'application/x-www-form-urlencoded'}

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)
```
```
import requests
import json
```
### Installations for Nesta
You can use pip to install the library:
```
pip install ojd-daps-skills
```
You will also need to install spaCy’s English language model:
```
python -m spacy download en_core_web_sm
```

```
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills #import the module

es = ExtractSkills(config_name="extract_skills_lightcast", local=True, multi_process=True) 

es.load()
```

## Running Google Colab Notebook for Embeddings and Cosine Similarity
For our embeddings and cosine similarity code, we used Google Colab for the code.
The file for the code can be found in the google colab notebook folder in our github link.
> CSC_869_Term_Project_Embeddings_and_Cosine_Similarity.ipynb

### Installations
We need to pip install dill, openai, tiktoken, and langchain in the notebook:
```
!pip install dill
!pip install openai
!pip install tiktoken
!pip install langchain
```
We next import the following:
```
from google.colab import files
from sklearn.metrics.pairwise import cosine_similarity
import dill
import pandas as pd
import numpy as np
import os
```

### Load the Data
Mount the google drive where we have a copy of our data in a pickle file.
```
from google.colab import drive
drive.mount('/content/drive')
```
```
with open('/content/drive/MyDrive/CSC869/df_eval.pkl','rb') as file:
  df = dill.load(file)

print(df)
```
