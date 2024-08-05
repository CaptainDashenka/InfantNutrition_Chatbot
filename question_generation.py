from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.llms.openai import OpenAI

reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
docs = reader.load_data()

data_generator = DatasetGenerator.from_documents(docs)
# the numer of questions generated had to be limited to avoid API call token limits by Open AI
eval_questions = data_generator.generate_questions_from_nodes(num=200)

print(eval_questions)