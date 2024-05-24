import os
import pandas as pd
from dotenv import load_dotenv
from langchain.globals import set_verbose

from util.constants import *


def initialize():
    load_dotenv()
    set_verbose(os.getenv('LANGCHAIN_VERBOSITY') or True)
    pd.set_option("display.max_rows", 20)
    pd.set_option("display.max_columns", 20)
    for dir in [VECTOR_STORE, INPUT_FILE_PATH, OUTPUT_FILE_PATH, INTERMEDIATE_FILE_PATH, EXTRACTED_FILE_PATH]:
        _create_temp_directory(dir)


def _create_temp_directory(dir):
    if not os.path.exists(dir):
        print(f"Creating directory ...  {dir}")
        os.makedirs(dir)
