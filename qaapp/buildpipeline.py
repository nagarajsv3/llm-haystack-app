from haystack.telemetry import tutorial_running
tutorial_running(11)

import logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# Initializing the DocumentStore
from haystack.document_stores import InMemoryDocumentStore
document_store = InMemoryDocumentStore(use_bm25=True)

# Fetching and writing documents
from haystack.utils import fetch_archive_from_http, convert_files_to_docs, clean_wiki_text

# Download and prepare data - 517 Wikipedia articles for Game of Thrones
doc_dir = "data/tutorial11"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt11.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# convert files to dicts containing documents that can be indexed to our datastore
got_docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
document_store.delete_documents()
document_store.write_documents(got_docs)

from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader

bm25_retriever = BM25Retriever()

from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents

p_retrieval = DocumentSearchPipeline(bm25_retriever)
# res = p_retrieval.run(query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}})
# print_documents(res, max_text_len=200)

# Initializa Core Components
from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader

# Initialize Sparse Retriever
bm25_retriever = BM25Retriever(document_store=document_store)

# Initialize embedding Retriever
embedding_retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
document_store.update_embeddings(embedding_retriever, update_existing_embeddings=False)

# Initialize Reader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# PreBuilt Pipelines
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
p_extractive_premade = ExtractiveQAPipeline(reader=reader, retriever=bm25_retriever)
res = p_extractive_premade.run(
    query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)
print_answers(res, details="minimum")


from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents

p_retrieval = DocumentSearchPipeline(embedding_retriever)
res = p_retrieval.run(query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}})
print_documents(res, max_text_len=200)


