import pytest
import datetime
from freezegun import freeze_time
from unittest.mock import patch, MagicMock
from services.llm import LLM

# from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext

@pytest.fixture
@patch('services.llm.chromadb.PersistentClient')
@patch('services.llm.VectorStoreIndex.from_vector_store')
def llm_instance(mock_chromadb, mock_vector_store_index):
    """Fixture to provide a default LLM instance for testing, with necessary patches."""
    # Mock the behavior of chromadb.PersistentClient
    mock_vector_store = MagicMock()
    mock_chromadb.return_value.get_or_create_collection.return_value = mock_vector_store

    # Mock the behavior of VectorStoreIndex.from_vector_store
    mock_index = MagicMock()
    mock_vector_store_index.return_value = mock_index

    instance = LLM()
    assert instance.retriever is not None
    assert instance.index is not None

    return instance


@patch('services.llm.chromadb.PersistentClient')
@patch('services.llm.VectorStoreIndex.from_vector_store')
def test_init_vector_storage_retriever(mock_chromadb, mock_vector_store_index, llm_instance):
    """Test that the vector storage retriever is initialized properly."""
    # Mock the behavior of chromadb.PersistentClient
    mock_vector_store = MagicMock()
    mock_chromadb.return_value.get_or_create_collection.return_value = mock_vector_store

    # Mock the behavior of VectorStoreIndex.from_vector_store
    mock_index = MagicMock()
    mock_vector_store_index.return_value = mock_index

    index, retriever = llm_instance.init_vector_storage_retriever()

    assert retriever is not None
    assert index is not None
    mock_chromadb.assert_called_once()


@freeze_time("2024-01-09 00:00:00.000000")
@pytest.mark.parametrize('score, metadata, expected',
                         [
                          (0.99, {"answer": "Test Answer", "time": "2024-01-01 00:00:00.000000"}, 
                           {"status": "success", "answer": None, "node_id": "123"}),
                           (0.85, {"answer": "Test Answer", "time": "2024-01-01 00:00:00.000000"}, 
                           {"status": "success", "answer": "Test Answer", "node_id": ""}),
                           (0.99, {"answer": "Test Answer", "time": "2024-01-07 00:00:00.000000"}, 
                           {"status": "success", "answer": "Test Answer", "node_id": ""}),
                          ], ids=repr)
def test_retrieve_most_relevant_answer_success(llm_instance, score, metadata, expected):
    """Test that the retriever successfully retrieves the most relevant answer."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [MagicMock(score=score, metadata=metadata, id_="123")]

    llm_instance.retriever = mock_retriever
    result = llm_instance.retrieve_most_relevant_answer("Test Query")

    assert result["status"] == expected["status"]
    assert result["answer"] == expected["answer"]
    assert result["node_id"] == expected["node_id"]
    assert result["error"] == ""


# def test_retrieve_most_relevant_answer_failure(llm_instance):
#     """Test that the retriever returns None when no relevant answer is found."""
#     mock_retriever = MagicMock()
#     mock_retriever.retrieve.return_value = []

#     llm_instance.retriever = mock_retriever
#     result = llm_instance.retrieve_most_relevant_answer("Test Query")

#     assert result["status"] == "success"
#     assert result["answer"] is None
#     assert result["error"] == ""

# def test_generate_prompt(llm_instance):
#     """Test prompt generation with a user query and schema."""
#     user_query = "SELECT * FROM users"
#     schema_data = "users table schema"
    
#     with patch("builtins.open", mock_open(read_data="Test Prompt")):
#         system_prompt, prompt = llm_instance.generate_prompt(user_query, schema_data)

#         assert "Test Prompt" in prompt
#         assert "users table schema" in prompt
#         assert user_query in prompt
#         assert system_prompt == ""

# def test_add_examples_from_rag(llm_instance):
#     """Test that examples are added from RAG correctly."""
#     mock_index = MagicMock()
#     mock_index.storage_context.vector_store._get.return_value.ids = ['id1', 'id2', 'id3']
#     mock_node = MagicMock(text="Sample query", metadata={"answer": "Sample SQL"})

#     mock_index.storage_context.vector_store.get_nodes.return_value = [mock_node, mock_node, mock_node]
#     llm_instance.index = mock_index

#     additional_prompt = llm_instance.add_examples_from_rag()

#     assert "Sample query" in additional_prompt
#     assert "Sample SQL" in additional_prompt
#     assert additional_prompt.count("SQL query") == 3

# def test_extract_sql_query_valid(llm_instance):
#     """Test that a valid SQL query is extracted from LLM response."""
#     llm_response = """
#     Here is your SQL query:
#     ``` SELECT * FROM users WHERE id = 1; ```
#     """
#     extracted_sql = llm_instance.extract_sql_query(llm_response)

#     assert extracted_sql.strip() == "SELECT * FROM users WHERE id = 1;"

# def test_extract_sql_query_invalid(llm_instance):
#     """Test that an invalid SQL query returns an empty string."""
#     llm_response = """
#     Here is your SQL query:
#     """
#     extracted_sql = llm_instance.extract_sql_query(llm_response)

#     assert extracted_sql == ""

# def test_natural_language_to_sql_rag_success(llm_instance):
#     """Test natural language to SQL conversion when an answer is retrieved from RAG."""
#     mock_retrieved_answer = {
#         "answer": "SELECT * FROM users",
#         "error": "",
#         "node_id": "node_123"
#     }
#     with patch.object(llm_instance, 'retrieve_most_relevant_answer', return_value=mock_retrieved_answer):
#         result = llm_instance.natural_language_to_sql("Test Query", "Test History", "Test Schema")

#         assert result["status"] == "success"
#         assert result["sql"] == "SELECT * FROM users"
#         assert result["is_rag"] is True

# def test_natural_language_to_sql_model_success(llm_instance):
#     """Test natural language to SQL conversion when no answer is retrieved from RAG, and model is queried."""
#     mock_retrieved_answer = {"answer": None, "error": ""}
#     mock_model_answer = {"status": "success", "answer": "``` SELECT * FROM users; ```", "error": ""}
    
#     with patch.object(llm_instance, 'retrieve_most_relevant_answer', return_value=mock_retrieved_answer), \
#          patch.object(llm_instance, 'openai_query', return_value=mock_model_answer):
#         result = llm_instance.natural_language_to_sql("Test Query", "Test History", "Test Schema")

#         assert result["status"] == "success"
#         assert result["sql"].strip() == "SELECT * FROM users;"
#         assert result["is_rag"] is False

# def test_natural_language_to_sql_failure(llm_instance):
#     """Test natural language to SQL conversion failure when no valid response is retrieved."""
#     mock_retrieved_answer = {"answer": None, "error": ""}
#     mock_model_answer = {"status": "failure", "answer": "", "error": "Model failed"}

#     with patch.object(llm_instance, 'retrieve_most_relevant_answer', return_value=mock_retrieved_answer), \
#          patch.object(llm_instance, 'openai_query', return_value=mock_model_answer):
#         result = llm_instance.natural_language_to_sql("Test Query", "Test History", "Test Schema")

#         assert result["status"] == "failure"
#         assert result["sql"] == ""
#         assert result["error_description"] == "Model failed"


