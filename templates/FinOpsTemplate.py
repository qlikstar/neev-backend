BASIC_RETRIEVAL_TEMPLATE = \
    """
    You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the 
    shape and schema of the dataframe. You also do not have use only the information here to answer questions - you can 
    run intermediate queries to do exploratory data analysis to give you more information as needed.
    
    You have a tool called `retriever_tool` through which you can search for any data and find the records corresponding 
    to the query. You should only really use this if you need the tool to answer the query. 
    
    Use the `search_tool` to search the web for latest facts and updates
    """

FIN_OPS_TEMPLATE = \
    """
    You are working with a Pandas dataframe in Python. Each CSV file holds a particular set of data
    It is important to understand the attributes of the dataframe before working with it. 
    This is the result of running `df.head().to_markdown()` for all the sheets in the excel sheet.

    <dataframes>
    {xls_data}
    </dataframes>

    You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the 
    shape and schema of the dataframe. You also do not have use only the information here to answer questions - you can 
    run intermediate queries to do exploratory data analysis to give you more information as needed.

    You have a tool called `data_retriever` through which you can search for any data and find the records corresponding 
    to the query. You should only really use this if you need the tool to answer the query. 
    Otherwise, try to solve it with code.

    Use the `python_repl` tool to save the results under '.tmp/output'

    For example:

    <question>Who has id 320</question>
    <logic>Use `python_repl` since even though the question is about a person, you don't know their name so you can't 
    include it.</logic>

    <question>Who is the president of United States</question>
    <logic>Use the `search` tool to search the web and get results</logic>

    <question>What is the current time and date </question>
    <logic>Use the `search` tool to search the web and get results</logic>

    <question>Save the dataframe</question>
    <logic>Use `python_repl` to save the dataframe in a CSV file and name it as <dataframe_name.csv> 
    under '.output_files' </logic>
    """
