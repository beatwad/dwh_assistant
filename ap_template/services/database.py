from typing import Union, Dict

import os
import yaml
import pandas as pd
import psycopg2

from dotenv import load_dotenv
from config.load_config import load_config

config_dict = load_config()

pg_config = config_dict["database"]
host = pg_config["pg_host"]
port = pg_config["pg_port"]
dbname = pg_config["pg_dbname"]
user = pg_config["pg_user"]

load_dotenv()
password = os.getenv("PG_PASSWORD")

prompt_config = config_dict["prompt_params"]
table_names_path = prompt_config["table_names_path"]
dbml_schema_path = prompt_config["database_schema_path"]


def execute_sql_query(sql_query: str) -> Dict[str, Union[str, None]]:
    """
    Executes an SQL query in Postgres database and returns the results in a dictionary.

    Parameters
    ----------
    sql_query : str
        SQL query string.

    Returns
    -------
    dict
        A dictionary with two keys 
        - 'result' as a pandas DataFrame containing the query results, 
        - 'error' with an error message if the execution failed.
    """
    
    # Connect to your postgres DB
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)

    # Open a cursor to perform database operations
    cur = conn.cursor()

    # Execute a query
    try:
        cur.execute(sql_query)
    except Exception as e:
        return {"result": None, "error" : str(e)}
    
    # Retrieve query results
    records = cur.fetchall()
    records = pd.DataFrame(records)
    records.columns = [desc[0] for desc in cur.description]

    return {"result": records, "error" : None}


def load_table_names(table_names_path: str = table_names_path) -> list:
    with open(table_names_path, "r") as f:
        table_names = yaml.safe_load(f)
        table_names = table_names["table_names"]

    return table_names


def load_dbml_schema(dbml_schema_path: str = dbml_schema_path) -> str:
    """Load DBML schema from file"""
    with open(dbml_schema_path, "r") as f:
        dbml_schema = f.read()

    return dbml_schema


def build_dbml_schema(table_names : list) -> str:
    """
    Generates a DBML schema for specified tables 
    in a PostgreSQL database schema using a static SQL query. 
    This version formats the output as per the provided DBML schema example,
    replacing 'double precision' data type with 'double' 
    for dbdiagram.io compatibility.

    Parameters
    ----------
    table_names : list
        List of table names for which the DBML schema should be generated.

    Returns
    -------
    str
        DBML schema as a string for the specified tables in the schema.
    """
    table_str = ""
    
    type_dict = {
        "integer": "int", 
        "character varying": "varchar",
        "timestamp without time zone": "timestamp",
        "timestamp with time zone": "timestamp",
        "double precision": "double",
    }

    for table in table_names:
        query = f"""
        SELECT 
            table_name, 
            column_name, 
            data_type
        FROM information_schema.columns
        WHERE 
            table_schema = 'public'
            AND table_name IN ('{table}')
        ORDER BY table_name, ordinal_position; 
        """
        res = execute_sql_query(query)["result"]
    
        for table in res["table_name"].unique():
            table_str += f"Table {table} " +"{\n"
            table_columns_types = res.loc[res["table_name"] == table, ["column_name", "data_type"]]
            for _, row in table_columns_types.iterrows():
                column_name = row["column_name"]
                data_type = row["data_type"]
                if data_type in type_dict:
                    data_type = type_dict[data_type]
                table_str += f"  {column_name} {data_type}\n"
            table_str += "}\n"

    return table_str
