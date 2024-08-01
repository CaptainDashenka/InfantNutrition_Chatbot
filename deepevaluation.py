from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from deepeval.integrations.llama_index import DeepEvalFaithfulnessEvaluator

# Read LlamaIndex's quickstart on more details, you will need to store your data in "YOUR_DATA_DIRECTORY" beforehand
documents = SimpleDirectoryReader("./data", recursive=True).load_data()
index = VectorStoreIndex.from_documents(documents)
rag_application = index.as_query_engine()

# An example input to your RAG application
user_input = "When can babies start eating solid food?"

# LlamaIndex returns a response object that contains
# both the output string and retrieved nodes
response_object = rag_application.query(user_input)

evaluator = DeepEvalFaithfulnessEvaluator()
evaluation_result = evaluator.evaluate_response(
    query=user_input, response=response_object
)
print(evaluation_result)
