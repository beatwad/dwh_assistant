# Data Warehouse (DWH) assistant for helping in writing SQL queries to DWH

Use OpenAI/Replicate/Yandex GPT API and schema of PostgresSQL database to help users to write SQL queries to this database.

Also uses RAG for storing of successful pairs user_query: sql_query. 
If user query is similar enough to one that is stored in RAG - don't ask LLM about it and just load SQL query from RAG.

## Configure .env file

Before start you need to add go to .env_example file and add API keys for SQLAlchemy user database and APIs for access to LLMs

After configuration is finished, rename file .env_example to .env

## Configure config/config.yaml file

Set llm_params/llm_model_name variable to "openai" if you want to use OpenAI API and set OPENAI_API_KEY variable in .env file.

Set llm_params/llm_model_name variable to "replicate" if you want to use Replicate API and set REPLICATE_API_TOKEN variable in .env file. 

Set llm_params/llm_model_name variable to "yandex_gpt" if you want to use Yandex GPT API and set YANDEX_API_KEY and YANDEX_FOLDER_ID variables in .env file.

## Configure database schema

Put names of tables in your database in file `services/table_names.yaml`

Put schema of your database in DBML format in file `services/database_schema.dbml`

## Change LLM prompt

Can be found in `services/prompt.txt`

If you want to use test database from this repository - don't change these files.

## Create Docker image

In ap_template folder run:

`docker build . -t dwh`

## Create test database

You can use data/create_db/create_order_db.py file to create test PostgreSQL database.

## Service start

`docker run --network="host" -p 5000:5000 -v ${PWD}/ap_template:/app/ap_template dwh`


