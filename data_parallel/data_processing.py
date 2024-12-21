# Data Processing Packages
import pandas as pd
import nltk
import dask.dataframe as dd
from dask.distributed import performance_report
from dask.distributed import Client
import re

# function to process the data and split the label and review text
def process_data_dask(df):
    '''
    Helper to process test and split label and text
    
    input : dataframe
    returns: pd dataframe  
    '''
    labels = df[0].str.split(' ', n=1, expand=True)[0].str.replace('__label__', '')
    text = df[0].str.split(' ', n=1, expand=True)[1]
    
    processed_df = pd.DataFrame({
    'label': labels,
    'review': text.str.strip()
    })
    return processed_df

# function to replace the labels
def replace_labels(df):
    '''
    replace the label 2 with 1 and 1 with 0 to make it binary classification problem
    1 -> good review
    0 -> bad review
    
    input: dataframe
    return: pd dataframe
    '''
    replace_dict = {"2": "1", "1": "0"}
    df['label'] = df['label'].replace(replace_dict)
    return df

# Text data cleaning
def text_cleaning(text):
    '''
    Helper function to clean the data only leave text
    
    input: string
    return: string
    '''
    text = text.lower()
    pattern_punc = r'[^A-Za-z\s]'
    text = re.sub(pattern_punc, '', text).strip()
    return text

# function to clean the reviews and drop the null values
def clean_reviews(df):
    '''
    helper function to apply the text_Cleaning to each row of the review column
    and drop the null values
    
    input: dataframe
    return: pd dataframe
    '''
    df['review'] = df['review'].apply(text_cleaning)
    df = df.dropna()
    return df

# process main and subset raw data 
def process_data():
    '''
    This is the main function to do dask data processing
    
    returns: tupel of pd dataframes
    '''
    # client to monitor the resources during Dask porcessing
    client = Client(n_workers=8, threads_per_worker=1, memory_limit='8GB')
    
    print("Client Dashboard: ", client.dashboard_link)
    
    # Source Data Path
    main_path = '../dataset/main_data.ft.txt'
    subset_path = '../dataset/subset_data.ft.txt'
    
    # loading data into Dask DataFrame
    main_data_dask = dd.read_csv(main_path, delimiter='\t', header=None, dtype=str)
    subset_data_dask = dd.read_csv(subset_path, delimiter='\t', header=None, dtype=str)
    
    # repartition subset data to have 4 partition
    subset_data_dask = subset_data_dask.repartition(npartitions=4)
    
    # persist the data in memory to avoid recomputation during processing
    main_data_dask = main_data_dask.persist()
    subset_data_dask = subset_data_dask.persist()
    
    meta = pd.DataFrame(columns=['label', 'review'], dtype=str)
    
    # apply the process_data_dask function to each partition of the dataframe
    main_data_dask = main_data_dask.map_partitions(process_data_dask, meta=meta)
    subset_data_dask = subset_data_dask.map_partitions(process_data_dask, meta=meta)
    
    # apply the replace_labels function to each partition of the dataframe
    main_data_dask = main_data_dask.map_partitions(replace_labels)
    subset_data_dask = subset_data_dask.map_partitions(replace_labels)
    
    # apply the clean_reviews function to each partition of the dataframe
    main_data_dask = main_data_dask.map_partitions(clean_reviews)
    subset_data_dask = subset_data_dask.map_partitions(clean_reviews)
    
    main_data_processed = None
    subset_data_processed = None
    # save the processed data report to html
    with performance_report(filename="dask_report/dask_report.html"):
        main_data_processed = main_data_dask.compute()
        subset_data_processed = subset_data_dask.compute()
    
    # close Client
    client.close()
    
    return main_data_processed, subset_data_processed

if __name__ == "__main__":
    # call the process_data function to process the data
    main_data_processed, subset_data_processed = process_data()
    if main_data_processed is not None and subset_data_processed is not None:
        print("Done")