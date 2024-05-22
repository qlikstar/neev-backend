import os

from dotenv import load_dotenv
from langchain.globals import set_verbose

from util.constants import *


def initialize():
    load_dotenv()
    set_verbose(os.getenv('LANGCHAIN_VERBOSITY') or True)
    for _ in [VECTOR_STORE, INPUT_FILE_PATH, OUTPUT_FILE_PATH, INTERMEDIATE_FILE_PATH]:
        _create_temp_directory()


def _create_temp_directory():
    temp_dir = TEMP_DIR
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
