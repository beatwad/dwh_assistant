from typing import Tuple, Dict

import os
import re
import json
import replicate
import requests
import chromadb

from random import sample
from datetime import datetime

from openai import OpenAI

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from config.load_config import load_config

config_dict = load_config()
prompt_config = config_dict["prompt_params"]
llm_config = config_dict["llm_params"]
rag_config = config_dict["rag_params"]


class LLM:
    """Class for LLM API and RAG interaction"""

    def __init__(self):
        # path to LLM prompt
        self.prompt_path = prompt_config["prompt_path"]
        # max length of previous dialogue between user and LLM that is used as prompt
        self.max_dialogue_length = prompt_config["max_dialogue_length"]
        # how many queries must be returned by retriever
        self.top_k = rag_config["top_k"] 
        # path to ChromaDB database
        self.chromadb_path = rag_config["chromadb_path"]
        # return query only if similarity score between query and the most similar node >= retrieval_thresh 
        self.retrieval_threshold = rag_config["retrieval_thresh"]
        # how many days query should be stored in RAG
        self.query_timeout = rag_config["query_timeout_days"]
        # The name of the model to load. Default is 'sentence-transformers/all-MiniLM-L6-v2'.
        # Model names can be found at the Hugging Face model hub: https://huggingface.co/models
        self.rag_model_name = rag_config["rag_model_name"]
        # openai, replicate or yandex_gpt
        self.llm_model_name = llm_config["llm_model_name"]
        # get vector index and retriever for RAG
        self.index, self.retriever = self.init_vector_storage_retriever()


    def init_vector_storage_retriever(
            self
            ) -> Tuple[VectorStoreIndex, VectorIndexRetriever]:
        """
        Initialize a retriever model for finding the most relevant answer to query in vector database.

        Returns:
            VectorStoreIndex: retriever for vector database.

        Raises:
            ValueError: If the model name is empty.
            RuntimeError: If the model fails to load.
        """

        # Explicitely set LLM to None to prevent using OpenAI API key
        Settings.llm = None

        # initialize chromadb client
        db = chromadb.PersistentClient(path=self.chromadb_path)

        # get collection
        chroma_collection = db.get_or_create_collection("quickstart")

        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # load your index from stored vectors
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context, 
            embed_model=f"local:{self.rag_model_name}"
        )   
        retriever = VectorIndexRetriever(index=index, similarity_top_k=self.top_k, use_metadata=False)

        return index, retriever


    def retrieve_most_relevant_answer(self, user_query: str) -> Dict[str, str]:
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
        Dict[str, str]: dictionary with answer, its status, error flag and
        ID of node with the most similar to user_query text.
        """
        nodes = self.retriever.retrieve(user_query)
        
        if len(nodes) > 0 and nodes[0].score >= self.retrieval_threshold:
            dt_now = datetime.now()
            query_time = nodes[0].metadata["time"]
            query_time = datetime.strptime(query_time, "%Y-%m-%d %H:%M:%S.%f")
                
            # if node with answer was not updated for too long and query text is similar enough to node text -
            # try to update it with the latest LLM response by deleting current node and inserting the new node
            # with fresh LLM response
            if (dt_now - query_time).total_seconds() // 3600 >= self.query_timeout * 24 and nodes[0].score >= 0.95:
                return {
                        "status": "success",
                        "answer": None,
                        "error": "",
                        "node_id": nodes[0].id_
                    }
            # else just return query from RAG
            else:
                return {
                        "status": "success",
                        "answer": nodes[0].metadata["answer"],
                        "error": "",
                        "node_id": ""
                    }
        
        return {
            "status": "success",
            "answer": None,
            "error": "",
            "node_id": ""
        } 
        

    def generate_prompt(self, user_query: str, schema_data: str) -> Tuple[str, str]:
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
        with open(self.prompt_path, 'r') as f:
            prompt = f.read()
        
        system_prompt = "" # this variable is used for Yandex GPT
        
        prompt += self.add_examples_from_rag()

        prompt += f"Database schema in DBML format:\n\n {schema_data}"
        prompt += f"\n{user_query}"

        return system_prompt, prompt 


    def add_examples_from_rag(self) -> str:
        """
        Get three additional examples from RAG
        
        Returns
        -------
        str
            Additional prompt with three new examples of
            succesful pairs user_query: sql_query
        """
        additional_prompt = ""
        
        ids = self.index.storage_context.vector_store._get(limit=100, where={}).ids
        
        if len(ids) >= 3:
            ids = sample(ids, 3)

            i = 3

            for id in ids:
                node = self.index.storage_context.vector_store.get_nodes([id])[0]
                query = node.text
                answer = node.metadata["answer"]
                additional_prompt += f'''\n{i}. User's request: "{query}"\n   SQL query:\n``` {answer} ```\n'''
                i += 1

        additional_prompt += "Database schema in DBML format:"

        return additional_prompt


    def replicate_query(self, user_query: str) -> Dict[str, str]:
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

    def openai_query(self, user_query: str) -> Dict[str, str]:
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


    def yandex_gpt_query(self, user_query: str) -> Dict[str, str]:
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


    def extract_sql_query(self, query: str) -> str:
        """
        Take LLM answer and extract SQL query from it.

        Parameters
        ----------
        user_query : str
            The user's query in natural language.

        Returns
        -------
        str
            Extracted SQL query
        """
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


    def natural_language_to_sql(self, user_query: str, query_history: str, 
                                schema_data: str) -> Dict[str, str]:
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
        if self.llm_model_name not in ["openai", "replicate", "yandex_gpt"]:
            raise ValueError("model value must be 'openai', 'replicate' or 'yandex_gpt'")
        
        _, prompt = self.generate_prompt(query_history, schema_data)

        # get answer from RAG or from model
        retrieved_answer = self.retrieve_most_relevant_answer(user_query)
        
        if retrieved_answer["answer"] is not None:
            answer = retrieved_answer
        elif self.llm_model_name == "openai":
            answer = self.openai_query(prompt)
        elif self.llm_model_name == "replicate":
            answer = self.replicate_query(prompt)
        else:
            answer = self.yandex_gpt_query(prompt)

        # extract SQL query from answer 
        if answer["error"]:
            result = {
                "status": "failure", 
                "sql": "",
                "is_rag": False,
                "node_id": "",
                "error_description": answer["error"],
                "raw_response": answer["answer"],
                }
        else:
            # add user_query, sql_query pair to RAG if there is no similar enough query in it
            if retrieved_answer["answer"] is None:
                sql_query = self.extract_sql_query(answer["answer"])
                is_rag = False
            else:
                sql_query = answer["answer"]
                is_rag = True

            if not sql_query:
                status = "failure"
                error_description = "Failed to parse model response."
            else:
                status = "success"
                error_description = ""
            
            result = {
                "status": status, 
                "sql": sql_query,
                "is_rag": is_rag,
                "node_id": retrieved_answer["node_id"],
                "error_description": error_description,
                "raw_response": answer["answer"],
                }

        return result


    def add_new_query_to_rag(self, user_query: str, sql_query: str, node_id: str) -> None:
        """
        If new query is correct - add it to RAG, optionally delete old node with the
        most similar to user_query text.

        Parameters
        ----------
        user_query : str
            The user's query in natural language.
        sql_query : str
            Response from LLM in SQL query form.
        node_id: str
            Index of the old node with the most similar to
            user_query text. That node should be deleted. 
        """
        dt_now = str(datetime.now())
        document = Document(text=user_query, metadata={"answer": sql_query, "time": dt_now})
        document.excluded_embed_metadata_keys = ["answer", "time"]
        if node_id:
            self.index.storage_context.vector_store.delete_nodes([node_id])
        self.index.insert(document)
