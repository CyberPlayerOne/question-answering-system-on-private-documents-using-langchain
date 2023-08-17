#!/usr/bin/env python

"""
Question-Answering Pipeline

1. Prepare the document (once per document)
a) Load the data into LangChain Documents;
b) Split the documents into chunks;
c) Embed the chunks into numeric vectors;
d) Save the chunks and the embeddings to a vector database

2. Search (once per query)
a) Embed the user's question;
b) Using the question's embedding qand the chunk embedding, rank the vectors by similarity to the question embedding. The nearest vectors represent chunks similar to the question.

3. Ask (once per query)
a) Insert the question and the most relevant chunks into a message to a GPT model;
b) Return GPT's answer.
"""
import os
from dotenv import load_dotenv, find_dotenv

# Load the environment variables.
load_dotenv(find_dotenv(), override=True)


def load_document(file):
    """
    Load a file into LangChain Documents
    :param file:
    :return:
    """
    import os
    name, extension = os.path.splitext(file)

    print(f'Loading {file}')
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    print(f'Total {len(data)} pages are loaded.')

    return data


def chunk_data(data, chunk_size=256 * 8):
    """
    Split langChain Document into smaller chunks
    :param data:
    :param chunk_size:
    :return:
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks


def insert_or_fetch_embeddings(index_name, chunks):
    """
    Embedding and Uploading to a Vector Database (Pinecone)
    :param chunks:
    :param index_name:
    :return:
    """
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ...', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('OK.')
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('OK.')

    return vector_store


def delete_pinecone_index(index_name='all'):
    import pinecone
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ...')
        for index in indexes:
            pinecone.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pinecone.delete_index(index_name)
        print('Ok')


def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    answer = chain.run(q)
    return answer


def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history


if __name__ == '__main__':
    # document_in_pinecone = False
    document_in_pinecone = True

    if not document_in_pinecone:
        # ########### Step 1 #############
        data = load_document('./Documents/2303.18223.pdf')  # 'https://arxiv.org/pdf/2303.18223.pdf'
        # # print(data[1].page_content)

        # ########### Step 2 #############
        chunks = chunk_data(data)
        # # print(len(chunks))
        # # print(chunks[10].page_content)

        # ########### Step 3 #############
        delete_pinecone_index()
    else:
        chunks = None
    # ########### Step 4 #############
    index_name = 'ask-a-document'
    vector_store = insert_or_fetch_embeddings(index_name, chunks)

    # ########### Step 5 #############
    q = 'What is the whole document about?'
    answer = ask_and_get_answer(vector_store, q)
    print(answer)

    import time

    i = 1
    print('Type Quit or Exit to quit.')

    chat_history = []
    while True:
        q = input(f'Question #{i}: ')
        i = i + 1
        if q.lower() in ['quit', 'exit']:
            print('Quitting ... bye bye!')
            time.sleep(2)
            break

        # answer = ask_and_get_answer(vector_store, q)
        # print(f'\nAnswer: {answer}')

        result, chat_history = ask_with_memory(vector_store, q, chat_history)
        print(f'\nAnswer: {result["answer"]}')

        print(f'\n {"-" * 50} \n')
