from haystack.telemetry import tutorial_running

tutorial_running(1)

import logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
logger = logging.getLogger("haystack")

# Initialize Document Store
from haystack.document_stores import InMemoryDocumentStore
document_store = InMemoryDocumentStore(use_bm25=True)

# Prepare Documents
from haystack.utils import fetch_archive_from_http
doc_dir = "data/build_your_first_question_answering_system"
fetch_archive_from_http(
    url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip",
    output_dir=doc_dir,
)

# Convert the downloaded files to Haystack Document Object and write them to Document Store
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)

#Initialize the retriever
from haystack.nodes import BM25Retriever
retriever = BM25Retriever(document_store=document_store)

#Initialize the reader
from haystack.nodes import FARMReader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

#Create Retriever-Reader Pipeline
from haystack.pipelines import ExtractiveQAPipeline
pipe = ExtractiveQAPipeline(reader, retriever)

prediction = pipe.run(
    query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)
from pprint import pprint
pprint(prediction)

logger.info("Simplify printing answers *******************************")

from haystack.utils import print_answers
print_answers(prediction, details="minimum")  ## Choose from `minimum`, `medium`, and `all`


