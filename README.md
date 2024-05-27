## How to Run the service

1. Go into the repo : `cd neev-backend`

2. You may choose to run a virtualenv if running locally.
   Run the requirements: `pip install -r requirements.txt`

3. Run `streamlit run app.py` on port 8000
4. Modify the `.streamlit\config.toml` to change port number

5. Here are a few different questions to try out:

   - `Show me first few rows of all the files in the buffer, as dataframes`
   - `Add a new column named "Amt Kenya" in Bank transactions by adding the "age of the president of Kenya" to the 'Amount' column as and save the results to buffer`.
   - `Now show me the first few rows of Bank transactions`
   - `create a new dataframe with the 2 columns: 1. dataframe name and 2. no of rows in the dataframe and save it as "summary" in buffer`
   - `Now show me the first few rows of "summary" dataframe`
   - `Now, create a new excel file with "summary" dataframe as the first sheet, and other input dataframes as sheets from the buffer, and save it as "output_all.xlsx"`
   - `Show me the first few lines from each sheet from "output_all.xlsx" under `output`.`

   - `Create a pie chart that shows the distribution of the "Total Amount" by description in Bank transactions.`