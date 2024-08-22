from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.langchain.base import LangchainEmbedding
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import SimpleNodeParser


# Step 1: Define the 5 sentences and the 6th sentence
data = [
    {"query": "What is the capital of France?", "answer": "Paris"},
    {"query": "What is the capital of Germany?", "answer": "Berlin"},
    {"query": "What is the largest ocean?", "answer": "Pacific Ocean"},
]

# The sentence to compare with
new_sentence = "What is the capital of Great Britain?"

# Convert sentences into Document objects
documents = [Document(text=item["query"], metadata={"answer": item["answer"]}) for item in data]

# Initialize BERT embedding model
bert_embedding_model = HuggingFaceEmbeddings(
    model_name="google/bert_uncased_L-2_H-128_A-2",
    # model_name="google-bert/bert-base-multilingual-cased",
)

# Wrap BERT embedding model in LangchainEmbedding to use in LLamaIndex
embedding = LangchainEmbedding(bert_embedding_model)

# Create a service context with the embedding model
# service_context = ServiceContext.from_defaults(llm=None, embed_model=embedding)
Settings.llm = None
Settings.embed_model = embedding

# Parse sentences into nodes using SimpleNodeParser
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# Create a vector store index with BERT embeddings
index = VectorStoreIndex(nodes, embedding=embedding)
retriever = index.as_retriever(similarity_top_k=1)

def find_the_closest_answer(retriever, new_sentence):
    # Create a retriever from the index
    nodes = retriever.retrieve(new_sentence)

    return nodes[0].metadata["answer"]
