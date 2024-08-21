from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.embeddings.langchain.base import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.service_context_elements.llm_predictor import LLMPredictor

from dotenv import load_dotenv

# load_dotenv()

# Step 1: Define the 5 sentences and the 6th sentence
sentences = [
    "The cat is on the roof.",
    "A dog barks loudly at night.",
    "Birds are chirping early in the morning.",
    "The sun rises in the east.",
    "Rain brings freshness to the air."
]

# The sentence to compare with
new_sentence = "Dogs bark at night."

# Convert sentences into Document objects
documents = [Document(text=f"Sentence {i+1}: {sent}") for i, sent in enumerate(sentences)]

# Initialize BERT embedding model
bert_embedding_model = HuggingFaceEmbeddings(
    model_name="google/bert_uncased_L-2_H-128_A-2"
)

# Wrap BERT embedding model in LangchainEmbedding to use in LLamaIndex
embedding = LangchainEmbedding(bert_embedding_model)

# Create a service context with the embedding model
service_context = ServiceContext.from_defaults(llm=None, embed_model=embedding)

# Parse sentences into nodes using SimpleNodeParser
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# Create a vector store index with BERT embeddings
index = VectorStoreIndex(nodes, embedding=embedding, service_context=service_context)

# Create a retriever from the index
retriever = index.as_retriever(similarity_top_k=1)
nodes = retriever.retrieve(new_sentence)

print(nodes[0].text)

# # Use RetrieverQueryEngine to perform the query without LLM
# query_engine = RetrieverQueryEngine.from_args(retriever=retriever)

# query_response = query_engine.query(new_sentence)

# # Extract and print the most similar sentence
# most_similar_sentence = query_response[0].node.get_text()
# print("The most similar sentence is:", most_similar_sentence)
