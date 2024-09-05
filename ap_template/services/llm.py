from typing import Tuple, Dict

import os
import re
import json
import replicate
import requests
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.langchain.base import LangchainEmbedding
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

load_dotenv()

service_start = os.getenv("SERVICE_START")
prompt_path = os.getenv("PROMPT_PATH")
chromadb_path = os.getenv("CHROMADB_PATH")

max_dialogue_length = os.getenv("MAX_DIALOGUE_LENGTH")
retrieval_threshold = os.getenv("RETRIEVAL_THRESH")
top_k = os.getenv("TOP_K")

max_dialogue_length = int(max_dialogue_length)
retrieval_threshold = float(retrieval_threshold)
top_k = int(top_k)

# TODO: save in RAG only direct query from user, and compare it with another direct queries from user 
# TODO: update answers of RAG if they are older that 30 days
# TODO: use your Vector Database to add examples of query-sql_query to prompt
# TODO: add tests
# TODO: wrap code in docker container

def init_vector_storage_retriever(
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k = 1,
        ) -> Tuple[VectorStoreIndex, SemanticSimilarityEvaluator]:
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

    # Initialize embedding model
    embed_model = resolve_embed_model(f"local:{model_name}")

    # Create a service context with the embedding model
    # service_context = ServiceContext.from_defaults(llm=None, embed_model=embedding)
    Settings.llm = None
    Settings.embed_model = embed_model

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
        embed_model=embed_model
    )

    evaluator = SemanticSimilarityEvaluator(
        embed_model=embed_model,
        similarity_threshold=retrieval_threshold,
    )

    return index, evaluator


# get vector index and retriever for RAG
index, evaluator = init_vector_storage_retriever(model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=1)
retriever = index.as_retriever(similarity_top_k=1)


# user_query = "Выведи все страны и количество клиентов в каждой из них "
# sql_query = ""
# document = Document(text=user_query, metadata={"answer": sql_query})
# index.insert(document)
# nodes = retriever.retrieve(user_query)

# for node in nodes:
#     print(node.id_)
#     print(node.text)
#     print(node.score)

# query_embedding = index._embed_model.get_query_embedding(user_query)
# document_embedding = index._embed_model.get_query_embedding(nodes[0].text)

# from llama_index.core.evaluation import SemanticSimilarityEvaluator

# # Initialize BERT embedding model
# embed_model = resolve_embed_model("local:sentence-transformers/all-MiniLM-L6-v2")

# evaluator = SemanticSimilarityEvaluator(
#     embed_model=embed_model,
#     similarity_threshold=0.8,
# )

# result = evaluator.evaluate(
#     response=nodes[0].text,
#     reference=user_query,
# )

# print(result.score)


def retrieve_most_relevant_answer(user_query: str, threshold: float = 0.8) -> Dict[str, str]:
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
    
    if len(nodes) > 0:
        result = evaluator.evaluate(
            response=nodes[0].text,
            reference=user_query,
        )
        passing = result.passing
    else:
        passing = True
    
    if len(nodes) == 0 or not passing:
        return {
            "status": "success",
            "answer": None,
            "error": "",
        } 
    
    return {
        "status": "success",
        "answer": nodes[0].metadata["answer"],
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


def natural_language_to_sql(user_query: str, schema_data: str, model: str = "openai") -> Dict[str, str]:
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
    
    _, prompt = generate_prompt(user_query, schema_data)

    # get answer from RAG or from model
    retrieved_answer = retrieve_most_relevant_answer(user_query, retrieval_threshold)
    
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
            if sql_query:
                document = Document(text=user_query, metadata={"answer": sql_query})
                index.insert(document)
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
