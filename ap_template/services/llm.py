from typing import Tuple, Dict

import os
import re
import json
import replicate
import requests
import chromadb

from datetime import datetime

from openai import OpenAI

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from config.load_config import load_config


config_dict = load_config()

prompt_config = config_dict["prompt_params"]
# path to LLM prompt
prompt_path = prompt_config["prompt_path"]
# max length of previous dialogue between user and LLM that is used as prompt
max_dialogue_length = prompt_config["max_dialogue_length"]

rag_config = config_dict["rag_params"]
# how many queries must be returned by retriever
top_k = rag_config["top_k"] 
chromadb_path = rag_config["chromadb_path"]
# return query only if similarity score between query and the most similar node >= retrieval_thresh 
retrieval_threshold = rag_config["retrieval_thresh"]
# how many days query should be stored in RAG
query_timeout = rag_config["query_timeout_days"] 


# TODO: use your Vector Database to add examples of query-sql_query to prompt
# TODO: add tests
# TODO: wrap code in the Docker container

def init_vector_storage_retriever(
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 1
        ) -> Tuple[VectorStoreIndex, VectorIndexRetriever]:
    """
    Initialize a retriever model for finding the most relevant answer to query in vector database.

    Parameters:
        model_name (str): The name of the model to load. Default is 'sentence-transformers/all-MiniLM-L6-v2'.
            Model names can be found at the Hugging Face model hub: https://huggingface.co/models

    Returns:
        VectorStoreIndex: retriever for vector database.

    Raises:
        ValueError: If the model name is empty.
        RuntimeError: If the model fails to load.
    """

    # Explicitely set LLM to None to prevent using OpenAI API key
    Settings.llm = None

    # initialize chromadb client
    db = chromadb.PersistentClient(path=chromadb_path)

    # get collection
    chroma_collection = db.get_or_create_collection("quickstart")

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context, 
        embed_model=f"local:{model_name}"
    )

    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k, use_metadata=False)

    return index, retriever


# get vector index and retriever for RAG
index, retriever = init_vector_storage_retriever(model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=top_k)



def retrieve_most_relevant_answer(user_query: str, 
                                  retrieval_threshold=retrieval_threshold) -> Dict[str, str]:
    """
    Retrieve from vector database anwer to the most relevant question
    
    Parameters
    ----------
    user_query : str
        The user's query in natural language.
    threshold : float
        The threshold of similarity: if similarity between user query
        and the most similar query from vector database is 
        more than a threshold - return its corresponding answer

    Returns
    -------
    str: answer
    """
    nodes = retriever.retrieve(user_query)
    
    if len(nodes) > 0 and nodes[0].score >= retrieval_threshold:
        dt_now = datetime.now()
        query_time = nodes[0].metadata["time"]
        query_time = datetime.strptime(query_time, "%Y-%m-%d %H:%M:%S.%f")
        # return query from RAG only if it's not too old
        # else try to update it with the latest LLM response
        if (dt_now - query_time).seconds // 3600 <= query_timeout // 24:
            return {
                    "status": "success",
                    "answer": nodes[0].metadata["answer"],
                    "error": "",
                }
    
    return {
        "status": "success",
        "answer": None,
        "error": "",
    } 
    

def generate_prompt(user_query: str, schema_data: str) -> Tuple[str, str]:
    """
    Generates a prompt for LLM, including the database schema and the user query.
    
    Parameters
    ----------
    user_query : str
        The user's query in natural language.
    schema_data : str
        The database schema in DBML format.

    Returns
    -------
    Tuple[str,str]: system prompt and prompt
    """
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    system_prompt = "" # this variable is used for Yandex GPT
    
    prompt += f"Database schema in DBML format:\n\n {schema_data}"
    prompt += f"\n{user_query}"

    return system_prompt, prompt 


def replicate_query(user_query: str) -> Dict[str, str]:
    """
    Sends a user query to a pre-defined LLM model via Replicate API 
    using an API token for authentication. Returns a structured response 
    including the model's answer or an error message.

    Parameters
    ----------
    user_query : str
        The user's query in natural language.

    Returns
    -------
    dict
        A dictionary with the processing status, the model's answer, 
        and an error description if applicable.
    """
    token = os.getenv("REPLICATE_API_TOKEN")

    api = replicate.Client(api_token=token)
    try:
        answer = api.run(
            "mistralai/mistral-7b-instruct-v0.2",
                input={
                    "prompt": user_query,
                    "debug": False,
                    "top_k": 50,
                    "top_p": 0.9,
                    "temperature": 0.3,
                    "max_new_tokens": 512,
                    "min_new_tokens": -1,
                    "prompt_template": "<s>[INST] {prompt} [/INST]",
                    "repetition_penalty": 1.15,
                    }
            )
        answer = ''.join(answer)
    except Exception as e:
        error = str(e)
        status = "failure"
        answer = ""
    else:
        error = ""
        status = "success"
    
    res = {"status": status, "answer": answer, "error": error}

    return res

def openai_query(user_query: str) -> Dict[str, str]:
    """
    Sends a user query to a pre-defined LLM model via OpenAI API 
    using an API token for authentication. Returns a structured response 
    including the model's answer or an error message.

    Parameters
    ----------
    user_query : str
        The user's query in natural language.

    Returns
    -------
    dict
        A dictionary with the processing status, the model's answer, 
        and an error description if applicable.
    """
    # Init client
    client = OpenAI()
    # Send request to OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # or gpt-4,
            messages=[
                {
                    "role": "user",
                    "content": user_query,
                }
            ],
            temperature=0.7  # Level of model randomness
        )
        answer = response.choices[0].message.content
    except Exception as e:
        error = str(e)
        status = "failure"
        answer = ""
    else:
        error = ""
        status = "success"
    
    res = {"status": status, "answer": answer, "error": error}

    return res


def yandex_gpt_query(user_query: str) -> Dict[str, str]:
    """
    Sends a user query to YandexGPT via Yandex Cloud API using an API key.
    Returns a structured response including the model's answer or an error message.

    Parameters
    ----------
    user_query : str
        The user's query in natural language.

    Returns
    -------
    dict
        A dictionary with the processing status, the model's answer,
        and an error description if applicable.
    """

    API_KEY = os.getenv("YANDEX_API_KEY")
    FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

    model_uri = f"gpt://{FOLDER_ID}/yandexgpt/latest"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "x-folder-id": FOLDER_ID,
    }

    data = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": 0.3,
            "maxTokens": 2000,
        },
        "messages": [
            {
                "role": "system",
                "text": "you are assistant",
            },
            {
                "role": "user",
                "text": user_query,
            },
        ],
    }

    response = requests.post(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        headers=headers,
        data=json.dumps(data),
    )

    if response.status_code != 200:
        return {
            "status": "failure",
            "answer": "",
            "error": f"Error from YandexGPT API: {response.text}",
        }

    response_data = response.json()
    result_text = (
        response_data.get("result", {})
        .get("alternatives", [{}])[0]
        .get("message", {})
        .get("text", "")
    )

    return {
        "status": "success",
        "answer": result_text,
        "error": "",
    }


def extract_sql_query(query: str) -> str:
    """Take LLM answer and extract SQL query from it."""
    r = re.compile('```')
    matches = [m.start() for m in r.finditer(query)]
    
    if len(matches) >= 2:
        begin = matches[0] + 3
        end = matches[-1]
    else:
        begin = end = None

    if begin and end:
        query = query[begin:end]
        query = re.sub("sql", "", query)
    else:
        query = ""

    return query


def natural_language_to_sql(user_query: str, query_history: str, 
                            schema_data: str, model: str = "openai") -> Dict[str, str]:
    """
    Converts a user query from natural language into an SQL query using LLM.

    Parameters
    ----------
    user_query : str
        The user's query in natural language.
    schema_data : str
        The database schema in DBML format.
    model : str
        Type of model, can be 
        - openai
        - replicate
        - yandex_gpt 

    Returns
    -------
    dict
        A dictionary with the processing result, including the SQL query or an error description if applicable.
    """
    if model not in ["openai", "replicate", "yandex_gpt"]:
        raise ValueError("model value must be 'openai', 'replicate' or 'yandex_gpt'")
    
    _, prompt = generate_prompt(query_history, schema_data)

    # get answer from RAG or from model
    retrieved_answer = retrieve_most_relevant_answer(user_query)
    
    if retrieved_answer["answer"] is not None:
        answer = retrieved_answer
    elif model == "openai":
        answer = openai_query(prompt)
    elif model == "replicate":
        answer = replicate_query(prompt)
    else:
        answer = yandex_gpt_query(prompt)

    # extract SQL query from answer 
    if answer["error"]:
        result = {
            "status": "failure", 
            "sql": "", 
            "error_description": answer["error"],
            "raw_response": answer["answer"],
            }
    else:

        # add user_query, sql_query pair to RAG if there is no similar enough query in it
        if retrieved_answer["answer"] is None:
            sql_query = extract_sql_query(answer["answer"])
        else:
            sql_query = answer["answer"]

        if not sql_query:
            status = "failure"
            error_description = "Failed to parse model response."
        else:
            status = "success"
            error_description = ""
        result = {
            "status": status, 
            "sql": sql_query, 
            "error_description": error_description,
            "raw_response": answer["answer"],
            }

    return result


def add_new_query_to_rag(user_query: str, sql_query: str) -> None:
    """If new query is correct - add it to RAG"""
    dt_now = str(datetime.now())
    document = Document(text=user_query, metadata={"answer": sql_query, "time": dt_now})
    document.excluded_embed_metadata_keys = ["answer", "time"]
    index.insert(document)


def prune_dialogue(dialogue: str) -> str:
    """
    If dialogue length is more than max context window size - 
    prune it buy deleting the very first dialogues so
    it can fit into the model context window
    
    Parameters
    ----------
    dialogue : str
        Dialogue between user and LLM.

    Returns
    -------
    str
        Pruned dialogue between user and LLM.
    """
    if len(dialogue) <= max_dialogue_length:
        return dialogue
    
    total_length = 0
    pruned_dialogue = []

    dialogue = dialogue.split("User's request: ")

    for d in dialogue[::-1]:
        if total_length + len(d) > max_dialogue_length:
            break
        else:
            total_length += len(d)
            pruned_dialogue.append(d)
    
    pruned_dialogue = pruned_dialogue[::-1]
    pruned_dialogue = ("User's request: ").join(pruned_dialogue)
        
    return "User's request: " + pruned_dialogue
