# Filepaths
TEMP_DIR = ".tmp/"
VECTOR_STORE = TEMP_DIR + "v-store/"
INPUT_FILE_PATH = TEMP_DIR + "input/"
OUTPUT_FILE_PATH = TEMP_DIR + "output/"
BUFFER_FILE_PATH = TEMP_DIR + "buffer/"
IMAGE_FILE_PATH = TEMP_DIR + "image/"

ALL_FILE_DIRS = [INPUT_FILE_PATH, OUTPUT_FILE_PATH, BUFFER_FILE_PATH, IMAGE_FILE_PATH]

# Sample Questions for demo
SAMPLE_QUESTIONS = \
"""
**Here are some of the sample questions you may ask:**
- What is the age of the President of United States
- Describe the input doc in 10 bullet points
- Show me first few rows of all the files in the buffer as dataframes
- Add a new column named "Amt Kenya" in Bank transactions by adding the "age of the president of Kenya" to the 'Amount' column as and save the results to buffer.
- create a new dataframe with the 2 columns: 1. dataframe name and 2. no of rows in the dataframe and save it as "summary" in buffer
- Now, create a new excel file with "summary" dataframe as the first sheet, and other input dataframes as sheets from the buffer, and save it as "output_all.xlsx"
"""
