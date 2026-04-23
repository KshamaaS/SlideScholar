from .vdb import vdb

import json

def read_chunks(chunk_path, **kwargs):
    """
    Read chunks from a JSON file and create a vector database.
    
    Args:
        chunk_path (str): The file path to the JSON file containing chunks.
    
    Returns:
        vdb: The vector database containing the chunks.
    """
    db = vdb(**kwargs)

    with open(chunk_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    db.add_documents(chunks)

    return db