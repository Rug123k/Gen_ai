# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings



def load_pdf_files(data):
    loader=DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)

    documents=loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filters a list of Document objects to only include those with content length greater than min_length.
    """
    minimal_docs:List[Document]=[]
    for doc in docs:
        src=doc.metadata.get('source')
        minimal_docs.append(
            Document(
            page_content=doc.page_content,
            metadata={"source": src}
            )
        )
    return minimal_docs

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks= text_splitter.split_documents(extracted_data)
    return text_chunks


def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        # model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu" }
        )
    return embeddings

