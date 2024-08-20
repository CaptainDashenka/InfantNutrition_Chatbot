import warnings
import csv
import pandas
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import dataclass


# create the same vector store index as the chatbot 
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data_reduced", recursive=True)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        system_prompt="""You are an expert on infant nutrition and your job is to answer questions. 
        Assume that all questions are related to the domain of infant and toddler feeding and nutrition. 
        Keep your answers concise and based on facts. If the answer is unknown then reply; This information is not in my knowlegde base.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()
rag_application = index.as_query_engine()

# read the questions from the file and create a test case dataset
dataset = EvaluationDataset()
file_path = 'questions.txt'
with open(file_path, mode='r', newline='') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    counter = 0
    for row in csv_reader:
        #print(row[0])
        question_input = row[0]

        # LlamaIndex returns a response object that contains both the output string and retrieved nodes
        response_object = rag_application.query(question_input)

        # Process the response object to get the output string and retrieved nodes
        if response_object is None:
            warnings.warn("Warning: vector store index query returned empty object")
        else:
            actual_output = response_object.response
            retrieval_context = [node.get_content() for node in response_object.source_nodes]

        # Create a test case and metric as usual
        test_case = LLMTestCase(
            input=question_input,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )

        dataset.add_test_case(test_case)

        counter += 1
        if counter >= 3:
            break


answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
faithfullness_metric = FaithfulnessMetric()
contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7)
# result[dataclass] = list[]
result = dataset.evaluate([answer_relevancy_metric, faithfullness_metric])

mydf = pandas.DataFrame([vars(s) for s in result])
print(mydf)

'''
# this block is using the llama index integration library by deepeval. not that useful for multiple questions and metrics
from deepeval.integrations.llama_index import DeepEvalFaithfulnessEvaluator
evaluator = DeepEvalFaithfulnessEvaluator()
evaluation_result = evaluator.evaluate_response(query=user_input, response=response_object)
print(evaluation_result)
'''
