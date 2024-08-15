# DWH assistant for helping in writing SQL queries to databases

Use OpenAI/Replicate/Yandex GPT API and schema of PostgresSQL database to help users to write SQL queries to this database.

## Configure .env file

Before start you need to add go to .env_example file and configure PostgresSQL database settings

Set MODEL variable to "openai" if you want to use OpenAI API and set OPENAI_API_KEY variable.

Set MODEL variable to "replicate" if you want to use Replicate API and set REPLICATE_API_TOKEN variable. 

Set MODEL variable to "yandex_gpt" if you want to use Yandex GPT API and set YANDEX_API_KEY and YANDEX_FOLDER_ID variables

After configuration is finished, rename file .env_example to .env

## Configure database schema

Put names of tables in your database in file services/table_names.yaml

Put schema of your database in DBML format in file services/database_schema.dbml

If you want to use example database from this repository - don't change these files and just run /create_db/create_db.py

## Create and activate virtual environment

`make setup`

`source .venv/bin/activate`

## Service start

`bash run.sh`

Default address: http://localhost:5000/


