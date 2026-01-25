# LLM & Agent Architecture in Bitcoin News Summary

This document explains how the LLM (Large Language Model) and agent components work in `bitcoin_news_summary.ipynb`.

## Overview

The notebook uses **LangChain** to orchestrate multiple AI components:

1. **News Verification Agent** — ReAct agent for fact-checking headlines
2. **RAG Pipeline** — Retrieval-Augmented Generation for context-aware summarization
3. **Summarization Chain** — LLM-powered news analysis
4. **Sentiment Classifier** — Direct LLM inference for headline sentiment

All components use **Ollama** for local LLM inference (no API keys required for the LLM itself).

---

## 1. News Verification Agent

### Purpose
Cross-validates fetched news headlines against web sources to filter out unreliable or fake news before processing.

### Architecture: ReAct Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    ReAct Agent Loop                         │
├─────────────────────────────────────────────────────────────┤
│  Input: News headline to verify                             │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │ Thought  │───▶│  Action  │───▶│  Observation         │  │
│  │          │    │ (Search) │    │  (Search results)    │  │
│  └──────────┘    └──────────┘    └──────────────────────┘  │
│       ▲                                    │                │
│       └────────────────────────────────────┘                │
│                    (repeat up to 3x)                        │
│                                                             │
│  Output: {"status": "verified|uncertain|unverified",        │
│           "reason": "explanation"}                          │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| LLM | `ChatOllama(model="qwen2.5:7b", temperature=0.1)` | Low temperature for deterministic reasoning |
| Tool | `DuckDuckGoSearchRun()` | Web search for corroboration |
| Agent | `create_react_agent()` | ReAct reasoning framework |
| Executor | `AgentExecutor(max_iterations=3)` | Limits reasoning loops |

### Code Flow

```python
# 1. Initialize verification LLM (low temp for consistency)
verification_llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

# 2. Define the search tool
search_tool = DuckDuckGoSearchRun()
tools = [Tool(name="Web_Search", func=search_tool.run, ...)]

# 3. Create ReAct agent with custom prompt
verification_agent = create_react_agent(verification_llm, tools, VERIFICATION_AGENT_PROMPT)

# 4. Execute for each headline
result = agent_executor.invoke({"input": headline})
```

### Caching Strategy
Results are cached daily to avoid redundant API calls:
```
./cache/verified_articles_YYYY-MM-DD.json
```

---

## 2. RAG Pipeline (Retrieval-Augmented Generation)

### Purpose
Retrieves the most relevant news chunks to provide context for the summarization LLM.

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   News      │     │   Text      │     │  Embedding  │
│  Articles   │────▶│  Splitter   │────▶│   Model     │
│             │     │ (1000 char) │     │ (nomic)     │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │     │  Similarity │     │  ChromaDB   │
│  Embedding  │────▶│   Search    │◀────│ Vector Store│
│             │     │  (top-K)    │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Retrieved  │
                    │   Context   │
                    └─────────────┘
```

### Pipeline Steps

| Step | Component | Configuration |
|------|-----------|---------------|
| 1. Document Creation | `Document` | Combines title + content with metadata |
| 2. Text Splitting | `RecursiveCharacterTextSplitter` | chunk_size=1000, overlap=200 |
| 3. Embeddings | `OllamaEmbeddings("nomic-embed-text")` | Local embedding model |
| 4. Vector Store | `Chroma.from_documents()` | In-memory vector database |
| 5. Retriever | `vectorstore.as_retriever()` | top-K similarity search |

### Code Flow

```python
# 1. Convert articles to Documents
documents = [Document(page_content=text, metadata={...}) for art in articles]

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
```

---

## 3. Summarization Chain

### Purpose
Generates a structured market analysis based on retrieved news context and BTC price data.

### Architecture: RetrievalQA Chain

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Query     │     │  Retriever   │     │   Retrieved  │
│  "Summarize  │────▶│   (RAG)      │────▶│   Documents  │
│   news..."   │     │              │     │   (top-10)   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Final      │     │     LLM      │     │   Prompt     │
│   Summary    │◀────│  (qwen2.5)   │◀────│  Template    │
│              │     │              │     │ + BTC Price  │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Prompt Template Structure

The prompt instructs the LLM to produce:
1. **Market Summary** — 1-2 sentence overview
2. **Key Drivers** — Top 3-5 bullish/bearish factors
3. **Sentiment** — Overall market sentiment
4. **Notable Events** — Major regulatory/institutional/technical events
5. **Outlook** — Forward-looking view

### Code Flow

```python
# 1. Define prompt with BTC price context
prompt = PromptTemplate(
    template=SUMMARY_PROMPT_TEMPLATE,
    partial_variables={
        "price": f"{current_price:,.2f}",
        "change": f"{change_pct:+.2f}",
        ...
    },
)

# 2. Initialize LLM (higher temp for creativity)
llm = ChatOllama(model=LLM_MODEL, temperature=0.3)

# 3. Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Stuffs all docs into context
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# 4. Invoke
result = qa_chain.invoke({"query": "Summarize the key Bitcoin news..."})
```

### Chain Type: "stuff"
All retrieved documents are concatenated ("stuffed") into a single prompt. This works well when:
- Documents are small enough to fit in context
- You need the LLM to see all information at once

---

## 4. Sentiment Analysis

### Purpose
Classifies each headline as Bullish/Neutral/Bearish for visualization.

### Architecture: Direct LLM Call

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Headlines   │     │   Prompt     │     │     LLM      │
│  (up to 20)  │────▶│  (JSON req)  │────▶│  (qwen2.5)   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │  JSON Array  │
                                          │  [{title,    │
                                          │  sentiment}] │
                                          └──────────────┘
```

### Code Flow

```python
# Direct LLM invocation (no RAG needed)
sentiment_response = llm.invoke(SENTIMENT_PROMPT.format(headlines=headlines_text))

# Parse JSON response
sentiment_results = json.loads(response_text)
```

---

## Model Configuration Summary

| Component | Model | Temperature | Purpose |
|-----------|-------|-------------|---------|
| Verification Agent | qwen2.5:7b | 0.1 | Deterministic fact-checking |
| Embeddings | nomic-embed-text | N/A | Document vectorization |
| Summarization | qwen2.5:7b | 0.3 | Balanced creativity/accuracy |
| Sentiment | qwen2.5:7b | 0.3 | Classification |

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FULL PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  News    │    │ Verification │    │   Filtered   │               │
│  │  APIs    │───▶│    Agent     │───▶│   Articles   │               │
│  │          │    │   (ReAct)    │    │              │               │
│  └──────────┘    └──────────────┘    └──────────────┘               │
│                                             │                        │
│                         ┌───────────────────┴───────────────────┐   │
│                         ▼                                       ▼   │
│                  ┌──────────────┐                      ┌──────────┐ │
│                  │     RAG      │                      │Sentiment │ │
│                  │   Pipeline   │                      │ Analysis │ │
│                  └──────────────┘                      └──────────┘ │
│                         │                                       │   │
│                         ▼                                       │   │
│                  ┌──────────────┐                               │   │
│                  │Summarization │                               │   │
│                  │    Chain     │                               │   │
│                  └──────────────┘                               │   │
│                         │                                       │   │
│                         ▼                                       ▼   │
│                  ┌─────────────────────────────────────────────────┐│
│                  │              OUTPUT DISPLAY                     ││
│                  │  • Market Summary    • Sentiment Pie Chart      ││
│                  │  • Source Articles   • Price Chart              ││
│                  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key LangChain Concepts Used

| Concept | Usage |
|---------|-------|
| **Document** | Wraps article text + metadata for processing |
| **Text Splitter** | Breaks documents into manageable chunks |
| **Embeddings** | Converts text to vectors for similarity search |
| **Vector Store** | ChromaDB stores and retrieves document embeddings |
| **Retriever** | Interface for similarity-based document retrieval |
| **RetrievalQA** | Combines retriever + LLM for question answering |
| **ReAct Agent** | Reasoning + Acting pattern for tool use |
| **AgentExecutor** | Manages agent loop with iteration limits |
| **PromptTemplate** | Structured prompts with variable substitution |

---

## Dependencies

```
langchain           # Core orchestration
langchain-ollama    # Ollama integration
langchain-community # Community tools (DuckDuckGo, Chroma)
chromadb            # Vector database
langgraph           # Agent graphs (installed but not directly used)
```

## Local Inference with Ollama

The notebook runs entirely locally using Ollama:
- No API keys required for LLM/embeddings
- Models must be pulled first: `ollama pull qwen2.5:7b` and `ollama pull nomic-embed-text`
- Ollama server must be running on `http://localhost:11434`
