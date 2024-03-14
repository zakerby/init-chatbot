from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.storage.storage_context import StorageContext

from qdrant_client import QdrantClient

llm_model = Ollama(
    model="gemma:2b", 
    base_url="http://localhost:11434",
    verbose=True
)


def send_prompt(prompt):
    response = llm_model.invoke(prompt)
    return response


def train_model():
    print("Reading JSON from training_data")
    documents = SimpleDirectoryReader("./training_data").load_data()
    # Initializing the vector store with Qdrant
    client = QdrantClient(path="./qdrant_data")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="jira_tickets"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Initializing the Large Language Model (LLM) with Ollama
    # The request_timeout may need to be adjusted 
    # depending on the system's performance capabilities
    service_context = ServiceContext.from_defaults(
        llm=llm_model,
        embed_model="local")

    # Creating the index, which includes embedding the documents 
    # into the vector store
    index = VectorStoreIndex.from_documents(documents,
                                            service_context=service_context,
                                            storage_context=storage_context)
    
    return index

