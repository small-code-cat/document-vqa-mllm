from box import Box
RAG_CONFIG = Box({
    'api_key': '',
    'base_url': '',
    'templates': {
        'visrag_query_template': 'Represent this query for retrieving relevant documents: {query}',
        'visrag_response_template': 'Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:',
        'gemini_resposne_template': '''
        Based on the content of the document page images retrieved in relation to the user's question, answer the user's question below as clearly and concisely as possible.
        
        Only provide a direct, paragraph-style answer. Do not include any introductions, notes, or extra commentary.
        
        Question:
        {query}
        
        Your answer:
        ''',
    },
    'topk': 3,
    'device': 'cuda:0',
    "retrievers": {
        'VisRAG-Retriever(MiniCPM-V2.0)': {'path':''},
    },
    "generators": {
        'MiniCPM-V2.6':{'path':''},
        'gemini-2.5-pro':{}
    },
    'layout_model': {'path': ''}
})
