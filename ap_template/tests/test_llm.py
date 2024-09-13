import pytest

from unittest.mock import patch, MagicMock
from services.llm import retrieve_most_relevant_answer, generate_prompt, extract_sql_query, natural_language_to_sql, add_new_query_to_rag

# Mock configuration
mock_schema_data = "DBML schema data"
mock_user_query = "SELECT * FROM users;"
mock_prompt = "Sample Prompt"
mock_query_history = "User's previous queries"
mock_sql_query = "SELECT * FROM users WHERE id = 1"

# Test for retrieve_most_relevant_answer
@patch('services.llm.retriever.retrieve')
def test_retrieve_most_relevant_answer(mock_retrieve):
    test_cases = [
        (0.99, {"answer": "SELECT * FROM users;", "time": "2023-01-01 00:00:00.000000"}, 
         {"status": "success", "answer": None, "node_id": "123"}),
        (0.8, {"answer": "SELECT * FROM users;", "time": "2023-01-01 00:00:00.000000"}, 
         {"status": "success", "answer": "SELECT * FROM users;", "node_id": ""}),
        #  (0.99, {"answer": "SELECT * FROM users;", "time": "2023-01-01 00:00:00.000000"}, # mock datetime.now()
        #  {"status": "success", "answer": "SELECT * FROM users;", "node_id": ""}),
    ]
    
    for score, metadata, expected in test_cases:
        mock_retrieve.return_value = [MagicMock(score=score, id_="123", metadata=metadata)]
        
        result = retrieve_most_relevant_answer("User query")
        print(result)
        
        assert result["status"] == expected["status"]
        assert result["answer"] == expected["answer"]
        assert result["node_id"] == expected["node_id"]
        assert "error" in result

# # Test for generate_prompt
# @patch('builtins.open', new_callable=pytest.mock.mock_open, read_data=mock_prompt)
# @patch('llm.add_examples_from_rag', return_value="")
# def test_generate_prompt(mock_add_examples_from_rag, mock_open):
#     system_prompt, generated_prompt = generate_prompt(mock_user_query, mock_schema_data)
    
#     assert mock_prompt in generated_prompt
#     assert mock_schema_data in generated_prompt
#     assert mock_user_query in generated_prompt

# # Test for extract_sql_query
# def test_extract_sql_query():
#     mock_llm_answer = "```SELECT * FROM users;```"
#     extracted_query = extract_sql_query(mock_llm_answer)
    
#     assert extracted_query == "SELECT * FROM users;"

# # Test for extract_sql_query with no match
# def test_extract_sql_query_no_match():
#     mock_llm_answer = "No SQL query in here."
#     extracted_query = extract_sql_query(mock_llm_answer)
    
#     assert extracted_query == ""

# # Test for natural_language_to_sql
# @patch('llm.generate_prompt', return_value=("system_prompt", "prompt"))
# @patch('llm.retrieve_most_relevant_answer', return_value={"answer": None, "node_id": ""})
# @patch('llm.openai_query', return_value={"status": "success", "answer": "```SELECT * FROM users;```", "error": ""})
# def test_natural_language_to_sql(mock_openai_query, mock_retrieve, mock_generate_prompt):
#     result = natural_language_to_sql(mock_user_query, mock_query_history, mock_schema_data, model="openai")
    
#     assert result["status"] == "success"
#     assert result["sql"] == "SELECT * FROM users;"
#     assert result["error_description"] == ""

# # Test for add_new_query_to_rag
# @patch('llm.index')
# def test_add_new_query_to_rag(mock_index):
#     mock_document = MagicMock()
#     add_new_query_to_rag(mock_user_query, mock_sql_query, "node_id_123")
    
#     assert mock_index.storage_context.vector_store.delete_nodes.called
#     assert mock_index.insert.called
