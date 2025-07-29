from box import Box

RAG_CONFIG = Box({
    'api_key': '',
    'base_url': '',
    'templates': {
        'visrag_response_template': 'Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:',
        'gemini_resposne_template': '''
        Based on the content of the document page images retrieved in relation to the user's question, answer the user's question below as clearly and concisely as possible.

        Only provide a direct, paragraph-style answer. Do not include any introductions, notes, or extra commentary.

        Question:
        {query}

        Your answer:
        ''',
    },
    'page_topk': 5,
    'layout_topk': 10,
    'device': 'cuda:0',
    "retrievers": {
        'VisRAG-Retriever(MiniCPM-V2.0)': {
            'path': '',
            'bs': 32,
            'bs_query': 32,
            'device': 'cuda:0',
            'query_template': 'Represent this query for retrieving relevant documents: {query}',
        },
    },
    "generators": {
        'MiniCPM-V2.6': {'path': ''},
        'gemini-2.5-pro': {}
    },
    'layout_model': {
        'path': ''}
})
