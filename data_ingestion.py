import requests
import os
import ast
import re
import pandas as pd
import configparser
import json
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
# Jun: save job postings info before cleaning to check how to clean the data
from pathlib import Path

cfp = configparser.RawConfigParser()
cfp.read('config.ini')

jsearch_url = cfp.get('jsearch', 'url')
jsearch_api_key = cfp.get('jsearch', 'api_key')

milvus_uri = cfp.get('jobmatch', 'uri')
milvus_token = cfp.get('jobmatch', 'token')

# 3. Get ContainerClient
container_client = blob_service_client.get_container_client(container_name)

querystring_ds = {
    "query": "Data scientist",
    "page": "1",
    "num_pages": "20",
    "date_posted": "today",
    "employment_types": "FULLTIME, CONTRACTOR, PARTTIME, INTERN",
    "job_requirements": "under_3_years_experience, no_experience, no_degree",
    "exclude_job_publishers": "Dice, jooble, Clearance Jobs, Geebo, Talent.com"
}

querystring_ml = {
    "query": "Machine learning",
    "page": "1",
    "num_pages": "20",
    "date_posted": "today",
    "employment_types": "FULLTIME, CONTRACTOR, PARTTIME, INTERN",
    "job_requirements": "under_3_years_experience, no_experience, no_degree",
    "exclude_job_publishers": "Dice, jooble, Clearance Jobs, Geebo, Talent.com"
}

querystring_ai = {
    "query": "AI",
    "page": "1",
    "num_pages": "20",
    "date_posted": "today",
    "employment_types": "FULLTIME, CONTRACTOR, PARTTIME, INTERN",
    "job_requirements": "under_3_years_experience, no_experience, no_degree",
    "exclude_job_publishers": "Dice, jooble, Clearance Jobs, Geebo, Talent.com"
}

headers = {
	"X-RapidAPI-Key": jsearch_api_key,
	"X-RapidAPI-Host": "jsearch.p.rapidapi.com"
}

response_ds = requests.get(jsearch_url, headers=headers, params=querystring_ds)
response_ml = requests.get(jsearch_url, headers=headers, params=querystring_ml)
response_ai = requests.get(jsearch_url, headers=headers, params=querystring_ai)

job_postings_json_ds = response_ds.json()
job_postings_json_ml = response_ml.json()
job_postings_json_ai = response_ai.json()

# This json file is created under the same path with vector_db
file_path = "data-ds.json"
with open(file_path, "w") as f:
    json.dump(job_postings_json_ds['data'], f) # this generates the file data-ds.json
file_path = "data-ml.json"
with open(file_path, "w") as f:
    json.dump(job_postings_json_ml['data'], f)
file_path = "data-ai.json"
with open(file_path, "w") as f:
    json.dump(job_postings_json_ai['data'], f)

df_ds = pd.read_json('data-ds.json', encoding = 'utf-8')
df_ml = pd.read_json('data-ml.json', encoding = 'utf-8')
df_ai = pd.read_json('data-ai.json', encoding = 'utf-8')
df = pd.concat([df_ds, df_ml, df_ai], axis=0)

df_select = df[
      ['job_id','job_title', 'employer_name', 'employer_logo', 'employer_website',
       'job_publisher', 'job_employment_type',
       'job_apply_link', 'job_description',
       'job_is_remote', 'job_city', 'job_state',
       'job_latitude', 'job_longitude', 'job_benefits',
       'job_highlights']
].copy()

# Jun: get the path of vector_db.py
srcpath = os.path.abspath(__file__)

# Jun: get the folder where vector_db is located
srcdir = os.path.dirname(srcpath)

# Jun: create the file path by combining the folder and the file name
filepath = Path(os.path.join(srcdir, 'before_clean.csv'))

# Jun: save the df_select to before_clean.csv
df_select.to_csv(filepath) 


def convert_to_dict(val):
    if isinstance(val, dict):
        return val  # It's already a dictionary, so return as is
    elif isinstance(val, str):
        try:
            return ast.literal_eval(val)  # Convert string to dictionary
        except (ValueError, SyntaxError):
            return None  # Handle any errors gracefully
    return None  # In case of any unexpected type


def clean_job_postings(df):
  # Combine job city and state into a single location column
  df['job_location'] = df['job_city'] + ', ' + df['job_state']
  # Create a unique identifier for job postings
  df['info'] = df['job_title'] + '|' + df['job_location'] + '|' + df['employer_name']
  # Remove duplicate job postings based on 'info' and 'job_description'
  df = df.drop_duplicates(subset='info', ignore_index=True)  # Reset index after removing duplicates
  df = df.drop_duplicates(subset='job_description', ignore_index=True)
  # Filter out job postings from unwanted publishers
  df = df[~df['job_publisher'].str.contains('Geebo', na=False)]  # Exclude 'Geebo' postings
  # df['job_required_experience'] = df['job_required_experience'].apply(convert_to_dict) # Jun: convert string to dict
  # df_exp = pd.json_normalize(df['job_required_experience']) # Jun: extract key:value paires as multiple columns
  # required_experience = df_exp['required_experience_in_months'] # Jun: only keep 'required_experience_in_months'
  # df = pd.concat([df.drop(columns=['job_required_experience']), df_exp], axis=1)
  # df = pd.concat([df, required_experience], axis=1) # Jun: add'required_experience_in_months' as a new column to original df
  # df['required_experience_in_months'] = df['required_experience_in_months'].fillna(0.0)
  # df['required_experience'] = df['required_experience_in_months'].astype(int) / 12
  
  # Identify citizenship requirements based on specific keywords
  df['citizenship'] = df['job_description'].str.contains(r'clearance|SCI|US citizenship|US Citizen|U.S. citizenship', case=False, na=False)
  
  # Define a helper function to classify degree requirements
  def classify_degree_requirement(job_description):
    """
    Classify the degree requirement into two categories:
    - "No degree required"
    - "Degree required (Bachelor degree or above)"
    """
    if pd.isna(job_description):
        return "No degree required"
    
    # Check for any degree requirement (Bachelor's, Master's, PhD, etc.)
    if re.search(r"PhD|doctoral|doctorate|Master's|Master|MBA|M\.Sc\.|MA|MS|Bachelor's|Bachelor|B\.Sc\.|BA|BS|college degree|university degree|academic qualification|graduate degree|undergraduate degree|postgraduate degree", 
             job_description, re.IGNORECASE):
        return "Degree required (Bachelor degree or above)"
      
    # Default case
    return "No degree required"

# Apply the helper function to the 'job_description' column    
  df['degree_requirement'] = df['job_description'].apply(classify_degree_requirement)
  
  df['remote or not'] = df['job_is_remote'].apply(lambda x: str(x).strip().lower() == 'true')

  return df

# don't forget to read_csv!
df_select = clean_job_postings(pd.read_csv(filepath)) 

# Set up the DataFrame
job_postings = df_select
job_postings = job_postings.dropna(subset=['info', 'job_description'])
job_postings = job_postings.fillna('') # na is not filled if use in clean_job_postings
job_postings.to_csv('jobs.csv', index=False)


if __name__ == '__main__':
    
    connections.connect("default",
                        uri=milvus_uri,
                        token=milvus_token)
    print(f"Connecting to DB: {milvus_uri}")

    # Initialize embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Define Milvus collection schema
    fields = [
        FieldSchema(name="job_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="job_description_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="job_description_raw", dtype=DataType.VARCHAR, max_length=50000),
        FieldSchema(name="info", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="degree", dtype=DataType.VARCHAR, max_length=255), # new
        FieldSchema(name="remote", dtype=DataType.BOOL, max_length=100), # new
        FieldSchema(name="location_state", dtype=DataType.VARCHAR, max_length=255), # new
        FieldSchema(name="employment_type", dtype=DataType.VARCHAR, max_length=255), # new
        FieldSchema(name="citizenship", dtype=DataType.BOOL, max_length=100),
        FieldSchema(name="job_apply_link", dtype=DataType.VARCHAR, max_length=255)
    ]
    
    schema = CollectionSchema(fields, description="Job listing data with job description embeddings and metadata")
    
    # Create collection (if it doesn't already exist)
    collection_name = "job_listings"
    if collection_name not in utility.list_collections():
        job_collection = Collection(name=collection_name, schema=schema)
    else:
        job_collection = Collection(name=collection_name)
    
    # Prepare lists for each field in the schema to match Milvus requirements
    id_list = []
    vector_list = []
    description_list = []
    info_list = []
    degree_list = []
    remote_list = []
    location_list = []
    emp_type_list = []
    citizenship_list = []
    link_list = []
    
    # Populate lists with DataFrame data
    for index, row in job_postings.iterrows():
        job_description_vector = embedder.encode(row['job_description']).tolist()
        
        # Append data for each field
        id_list.append(row['job_id'])  # Optional if using auto-generated ID in Milvus
        vector_list.append(job_description_vector)
        description_list.append(row['job_description'])
        info_list.append(row['info'])
        degree_list.append(row['degree_requirement'])
        remote_list.append(row['remote or not'])
        location_list.append(row['job_state'])
        emp_type_list.append(row['job_employment_type'])
        citizenship_list.append(row['citizenship'])
        link_list.append(row['job_apply_link'])
    
    # Insert data into Milvus collection
    job_collection.upsert([
        id_list,
        vector_list,
        description_list,
        info_list,
        degree_list,
        remote_list,
        location_list,
        emp_type_list,
        citizenship_list,
        link_list
    ])
    
    print("DataFrame indexed into Milvus successfully.")
