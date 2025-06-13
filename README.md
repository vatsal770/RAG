# RAG

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline using both **dense** and **sparse** retrieval approaches. Two language models—**HuggingFaceTB/SmolLM-1.7B-Instruct** and **Google's Gemma-2B-IT**—are utilized as back-end large language models (LLMs) to answer questions referring to retrieved document context.

## Key Components

### 1. Text Splitting


| Splitter                         | Strategy                | Overlap Support | Use Case                          |
| -------------------------------- | ----------------------- | --------------- | --------------------------------- |
| `RecursiveCharacterTextSplitter` | Recursive by separators | ✅               | General-purpose (.txt, .html)                  |
| `NltkTextSplitter`               | Sentence-based (NLTK)   | ❌               | Well-formed paragraphs/documents (.txt, .docx, news articles, formal writing) |
| `SpacyTextSplitter`              | Sentence/Token-based    | ✅               | Your text has complex structure (quotes, titles, etc.)              |
| `MarkdownTextSplitter`           | Markdown-aware blocks   | ✅               | Legal docs, markdown files    |


* **Tool Used**: `RecursiveCharacterTextSplitter` from LangChain
* **Why**: Handles nested document structures with adjustable chunk size and overlap.
* **Chunk Size**: Typically set in characters, not words (e.g., 300 means 300 characters).

### 2. Embedding Models

#### Dense Embeddings


| Model Name                               | Dimensions | Trained On             | Strengths                          |
| ---------------------------------------- | ---------- | ---------------------- | ---------------------------------- |
| `BAAI/bge-small-en-v1.5`                 | 384        | English text corpus    | Lightweight and Strong for tasks like semantic search and retrieval              |
| `BAAI/bge-large-en`                      | 1024       | Large English datasets | Better semantic understanding but requires more memory     |
| `sentence-transformers/all-MiniLM-L6-v2` | 384        | General web corpora    | The model uses a contrastive objective, allowing it to capture semantic relationships between sentences effectively                  |
| `intfloat/e5-large`                      | 1024       | NLI, QA, BEIR tasks    | Slower but excellent for long-document retrieval |
| `jinaai/jina-embeddings-v2-base-en`      | 768        | Multilingual + English | Processing long documents, achieving high performance in both mono-lingual and cross-lingual tasks               |


* **Model**: `BAAI/bge-small-en-v1.5`

* **BAAI/bge-small-en-v1.5**: High-performing dense embedding on BEIR (Benchmarking Information Retrieval) with low resource usage.
  * Trained specifically for dense retrieval tasks.
  * Supports similarity search with cosine distance in vector space.
  * Lightweight and efficient (small model size, \~34M parameters).

```python
vectorstore_dense = Chroma.from_documents(
    documents=documents,
    embedding=embed_model_dense,
)
```

#### Sparse Embeddings


| Model Name                          | Technique | Language | Strengths                                      | Notes                               |
| ----------------------------------- | --------- | -------- | ---------------------------------------------- | ----------------------------------- |
| `BM25Okapi`                         | BM25      | Any      | Fast, interpretable, no training needed        | Tokenized word overlap              |
| `RedHatAI/bge-small-en-v1.5-sparse` | SPLADE    | English  | Combines sparse structure with learned weights | Needs sparse vector store support   |


* **Approach**: BM25 using `rank_bm25` Python library

* **BM25Okapi**: Baseline sparse retriever — useful for comparative performance with dense models.
  * Lexical matching using TF-IDF-style scoring.
  * Term Frequency (TF): Measures how often a word appears in a document.
  * Inverse Document Frequency (IDF): Reduces the weight of common words across multiple documents while increasing the weight of rare words.
  * No training or embeddings needed.
  * Performs well when exact term match is crucial.

```python
bm25 = BM25Okapi([text.split(" ") for text in texts])
```

### 3. Language Models (LLMs)


| Model Name                           | Size      | Open Source | Strengths                                   | Notes                            |
| ------------------------------------ | --------- | ----------- | ------------------------------------------- | -------------------------------- |
| `HuggingFaceTB/SmolLM-1.7B-Instruct` | 1.7B      | ✅           | Lightweight, fast inference, instruct-tuned | Good for local setups           |
| `google/gemma-2b-it`                 | 2B        | ✅ (key)     | Instruction-tuned, multilingual             | Needs HuggingFace Auth token          |
| `meta-llama/Llama-2-7b-chat-hf`      | 7B        | ✅ (key)     | High-quality chat with good reasoning                | Needs Meta sign-in for model download |
| `mistralai/Mistral-7B-Instruct-v0.1` | 7B        | ✅           | High-speed processing                | Mistral AI has its models readily available for use and modification               |
| `OpenAI/gpt-3.5-turbo`               | N/A (API) | ❌           | SOTA in reasoning/QA                        | Chargable according to the usage                  |


#### 🔹 HuggingFaceTB/SmolLM-1.7B-Instruct

* **SmolLM-1.7B**: Small size, fast, efficient — best for CPU/GPU-constrained environments.
* Instruction-tuned 1.7B parameter model
* The model is trained on a dataset of instructions and corresponding responses to better understand and respond to clear directives
* Uses Tokenizer = GPT2TokenizerFast (Breaks input into bytes rather than characters/words. It then looks for pairs of bytes that appear together frequently in the text and replaces them with a single byte. This process is repeated until the entire text is encoded.)
* Lightweight, faster inference
* Open-source and doesn't require authentication

#### 🔹 google/gemma-2b-it

* **Gemma-2B-it**: Google’s instruction-tuned open model, good multilingual and reasoning ability.
* Google’s fine-tuned 2.34B parameter model has higher accuracy than SmolLM on benchmarks like MMLU and TruthfulQA
* Uses Tokenizer = GemmaTokenizerFast (unigram language model chooses best token combination with probabilistic splits)
* **Requires API token** (authentication needed)
* Better generalization on longer-form reasoning

```python
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
```

### 4. Retrieval and QA


| Retriever                          | Method             | Strengths                                    |
| ---------------------------------- | ------------------ | -------------------------------------------- |
| `Chroma.as_retriever()`            | Dense (similarity) | Fast, GPU/CPU support, easy integration      |
| `BM25Okapi`                        | Sparse             | No training, high interpretability but limited semantics          |
| `FaissRetriever`                   | Dense (FAISS)      | Fast indexing with handling datasets with millions (or even billions) of vectors |
| `ElasticSearchRetriever`           | Hybrid             | Supports keyword search, vector search, hybrid search  |
| `MultiVectorRetriever` (LangChain) | Multi-query        | Generates multiple synthetic queries for each chunk but slower setup   |


| Method/Class                        | Use Case                         | Custom Prompt | Conversational |
| ----------------------------------- | -------------------------------- | -------------- | --------------- |
| `RetrievalQA.from_chain_type()`     | Quick setup with chain templates | ❌              | ❌               |
| `RetrievalQA.from_llm()`            | Custom prompts with LLM          | ✅              | ❌               |
| `ConversationalRetrievalChain(...)` | Chatbot with memory              | ✅              | ✅               |

#### Dense Retrieval

```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
```

#### Sparse Retrieval with BM25

```python
sparse_qa_chain = (
    {"context": lambda x: format_docs(sparse_retrieval_bm25(x["question"])),
     "question": RunnablePassthrough()}
    | prompt
    | llm
)
```


## Why Both Dense and Sparse?

* **Dense retrieval** finds semantically similar chunks, better for complex queries.
* **Sparse retrieval** performs well when query and document share exact keywords.
* **Hybrid setups** Combining both often results in improved recall.

---

## 🛠️ Setup

```bash
# Create Conda env
conda create -n rag_env python=3.12.3 -y
conda activate rag_env
pip install -r requirements.txt
```
