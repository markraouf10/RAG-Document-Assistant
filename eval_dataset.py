QA_PAIRS = [
    {
        "question": "What does BERT stand for?",
        "ground_truth": "BERT stands for Bidirectional Encoder Representations from Transformers, a language representation model introduced by Google AI Language."
    },
    {
        "question": "What are the two pre-training tasks used in BERT?",
        "ground_truth": "BERT is pre-trained using two unsupervised tasks: Masked Language Model (MLM), where 15% of tokens are randomly masked and the model predicts them, and Next Sentence Prediction (NSP), where the model predicts whether two sentences follow each other."
    },
    {
        "question": "What are the two model sizes of BERT?",
        "ground_truth": "BERT comes in two sizes: BERT BASE with L=12 layers, H=768 hidden size, A=12 attention heads and 110M parameters, and BERT LARGE with L=24 layers, H=1024 hidden size, A=16 attention heads and 340M parameters."
    },
    {
        "question": "What is the attention mechanism in the Transformer?",
        "ground_truth": "An attention function maps a query and a set of key-value pairs to an output, where the output is computed as a weighted sum of the values, and the weight assigned to each value is computed by a compatibility function of the query with the corresponding key."
    },
    {
        "question": "What is Scaled Dot-Product Attention?",
        "ground_truth": "Scaled Dot-Product Attention takes queries and keys of dimension dk and values of dimension dv, computes the dot products of the query with all keys, divides each by the square root of dk, and applies a softmax function to obtain the weights on the values. The formula is: Attention(Q,K,V) = softmax(QK^T / sqrt(dk))V."
    },
    {
        "question": "What is multi-head attention?",
        "ground_truth": "Multi-head attention linearly projects queries, keys and values h times with different learned projections, performs attention in parallel on each projection yielding dv-dimensional output values, then concatenates and projects the results. The Transformer uses h=8 parallel attention heads."
    },
    {
        "question": "What is positional encoding in the Transformer?",
        "ground_truth": "Since the Transformer contains no recurrence or convolution, positional encodings are added to the input embeddings to inject information about token positions. The paper uses sine and cosine functions of different frequencies: PE(pos, 2i) = sin(pos/10000^(2i/dmodel)) and PE(pos, 2i+1) = cos(pos/10000^(2i/dmodel))."
    },
    {
        "question": "What is Retrieval Augmented Generation (RAG)?",
        "ground_truth": "RAG models combine pre-trained parametric memory (a seq2seq model) with non-parametric memory (a dense vector index of Wikipedia accessed with a neural retriever) for language generation. The model retrieves top-K documents for a query and uses them as context to generate the output."
    },
    {
        "question": "What retriever does the RAG paper use?",
        "ground_truth": "The RAG paper uses Dense Passage Retriever (DPR) as its retrieval component, which follows a bi-encoder architecture where both document and query encoders are based on BERT BASE. It uses Maximum Inner Product Search (MIPS) to find the top-K documents."
    },
    {
        "question": "What are the two RAG model variants proposed in the paper?",
        "ground_truth": "The paper proposes two RAG variants: RAG-Sequence, which uses the same retrieved document to generate the complete output sequence, and RAG-Token, which can use a different retrieved document for each output token, allowing content from several documents to be combined."
    },
    {
        "question": "What generator does RAG use?",
        "ground_truth": "RAG uses BART-large as its generator component, a pre-trained seq2seq transformer with 400M parameters. To combine the input with retrieved content, the input and retrieved passage are simply concatenated before being fed to BART."
    },
    {
        "question": "What BLEU score did the Transformer achieve on WMT 2014 English-to-German translation?",
        "ground_truth": "The big Transformer model achieved 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results including ensembles by over 2 BLEU."
    },
    {
        "question": "What datasets were used to evaluate RAG on open-domain question answering?",
        "ground_truth": "RAG was evaluated on four open-domain QA datasets: Natural Questions (NQ), TriviaQA (TQA), WebQuestions (WQ), and CuratedTrec (CT), achieving state-of-the-art results on all four."
    },
    {
        "question": "What masking rate does BERT use for the Masked Language Model pre-training task?",
        "ground_truth": "BERT randomly masks 15% of all WordPiece tokens in each sequence. Of the chosen tokens, 80% are replaced with the [MASK] token, 10% are replaced with a random token, and 10% are kept unchanged."
    },
    {
        "question": "What is the main advantage of RAG over purely parametric models?",
        "ground_truth": "Unlike parametric models that cannot easily expand or revise their memory and may produce hallucinations, RAG combines parametric and non-parametric memory so knowledge can be directly revised and expanded, accessed knowledge can be inspected and interpreted, and the non-parametric memory can be replaced to update world knowledge without any retraining."
    },
]
