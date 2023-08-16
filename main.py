import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


def load_document(file):
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

# data = load_document('https://arxiv.org/pdf/1706.03762.pdf')
# print(data[1].page_content)
