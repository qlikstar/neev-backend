import os
import pandas as pd
from util.constants import BUFFER_FILE_PATH


def convert_csv_to_df():
    all_xls_data = ''
    directory = BUFFER_FILE_PATH
    df_dict = {}

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):

            # Construct the full file path
            filepath = os.path.join(directory, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(filepath)
            filename_wo_extn = ".".join(filename.split(".")[:-1])
            all_xls_data = all_xls_data + (f' file: {filepath}, \n<df_{filename_wo_extn}>{df}</df_{filename_wo_extn}>\n'
                                           f'----------------------------------- \n\n\n\n')
            df_name = f"df_{filename_wo_extn}"
            # print(f"DF name: {df_name}")
            df_dict[df_name] = df

    return all_xls_data, df_dict
