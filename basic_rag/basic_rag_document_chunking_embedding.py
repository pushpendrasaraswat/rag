# ==============================================================================
# RAG PIPELINE - DOCUMENT LOADING, CHUNKING, AND EMBEDDING GENERATION
# ==============================================================================
# This script builds the foundation of a Retrieval-Augmented Generation (RAG)
# system by:
#   1. Loading raw text documents from disk
#   2. Chunking them into smaller, retrievable pieces
#   3. Converting those chunks into vector embeddings using a sentence
#      transformer model
# ==============================================================================

from openai import OpenAI
from load_key.load_key import Settings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from openai.types.chat import (ChatCompletionSystemMessageParam,ChatCompletionUserMessageParam)


# ------------------------------------------------------------------------
# SECTION 1: LOAD RAW DOCUMENTS FROM DISK
# ------------------------------------------------------------------------
# Read each source document (HR policy, engineering standards, onboarding
# guide, product knowledge base, and security policy) into memory as plain
# text strings. These will later be split into chunks for retrieval.

with open('../content/company_hr_policy.txt', 'r') as f:
    hr_document = f.read()

with open('../content/engineering_standards.txt', 'r') as f:
    eng_document = f.read()

with open('../content/onboarding_guide.txt', 'r') as f:
    onb_document = f.read()

with open('../content/product_knowledge_base.txt', 'r') as f:
    prod_document = f.read()

with open('../content/security_policy.txt', 'r') as f:
    sec_document = f.read()


# ------------------------------------------------------------------------
# SECTION 2: PRINT RAW DOCUMENT CONTENTS
# ------------------------------------------------------------------------
# Quick sanity check — print each document's full content to the console,
# separated by dividers, so we can visually confirm the files loaded
# correctly before doing any processing on them.

print(hr_document)
print('=' * 100)
print('#' * 100)
print(eng_document)
print('=' * 100)
print('#' * 100)
print(onb_document)
print('=' * 100)
print('#' * 100)
print(prod_document)
print('=' * 100)
print('#' * 100)
print(sec_document)


# ------------------------------------------------------------------------
# SECTION 3: BASIC DOCUMENT SIZE STATS (CHARACTER COUNTS)
# ------------------------------------------------------------------------
# Report the raw character length of each document. Useful for gauging
# overall document size before chunking / embedding decisions.

print(f"HR Document Length: {len(hr_document)}")
print(f"Engineering Document Length: {len(eng_document)}")
print(f"Onboarding Document Length: {len(onb_document)}")
print(f"Product Document Length: {len(prod_document)}")
print(f"Security Document Length: {len(sec_document)}")

print('=' * 100)
print('#' * 100)


# ------------------------------------------------------------------------
# SECTION 4: APPROXIMATE TOKEN COUNTS (WORD-SPLIT BASED)
# ------------------------------------------------------------------------
# A rough approximation of token count using whitespace splitting
# (not a true tokenizer, but useful as a quick proxy for how "chunky"
# each document is before embedding).

print(f"HR Policy Token Count: {len(hr_document.split())}")
print(f"Engineering Standards Token Count: {len(eng_document.split())}")
print(f"Onboarding Document Token Count: {len(onb_document.split())}")
print(f"Product Knowledge Document Token Count: {len(prod_document.split())}")
print(f"Security Document Token Count: {len(sec_document.split())}")


# ------------------------------------------------------------------------
# SECTION 5: CHUNKING FUNCTION (PARAGRAPH-BASED SPLITTING)
# ------------------------------------------------------------------------
# Splits a document into chunks along blank-line boundaries (i.e. treats
# each paragraph as a chunk). This is a simple, non-overlapping,
# paragraph-level chunking strategy (as opposed to fixed-size, sliding
# window/overlapping, recursive, or semantic chunking approaches).
#
# Filtering rules applied to each candidate paragraph:
#   - Skip paragraphs shorter than 50 characters (too small to be useful
#     as a standalone retrievable chunk, likely headers/whitespace noise)
#   - Skip paragraphs that start with "==" (these are assumed to be
#     section divider lines rather than real content)
#
# Each surviving chunk is stored as a dict containing:
#   - 'text': the chunk's paragraph content
#   - 'source': a label identifying which document the chunk came from
#     (used later for citation/traceability during retrieval)

def chunk_document(text, source):
    # split the text based on two new lines(\n\n)
    paragraph = text.strip().split('\n\n')

    chunks = []

    for para in paragraph:
        para = para.strip()
        if len(para) < 50 or para.startswith('=='):
            continue
        chunks.append({'text': para, 'source': source})

    return chunks


# ------------------------------------------------------------------------
# SECTION 6: APPLY CHUNKING TO EACH DOCUMENT
# ------------------------------------------------------------------------
# Run the chunking function on every loaded document, tagging each
# resulting chunk with its originating policy/document name so we know
# where each chunk came from once everything is merged together.

hr_chunks = chunk_document(hr_document, 'HR Policy')
print("hr chunk :: ", hr_chunks)

eng_chunks = chunk_document(eng_document, 'Engineering Policy')
print("eng chunk :: ", eng_chunks)

onb_chunks = chunk_document(onb_document, 'Onboarding Policy')
print("onb chunk :: ", onb_chunks)

prod_chunks = chunk_document(prod_document, 'Product Policy')
print("prod chunk :: ", prod_chunks)

sec_chunks = chunk_document(sec_document, 'Security Policy')
print("sec chunk :: ", sec_chunks)


# ------------------------------------------------------------------------
# SECTION 7: CHUNK COUNT SUMMARY
# ------------------------------------------------------------------------
# Print how many chunks were produced per document, to sanity-check that
# chunking behaved as expected (e.g. no document ended up with zero
# chunks, no unexpectedly huge chunk counts).

print(f"Number of chunks in HR Policy: {len(hr_chunks)}")
print(f"Number of chunks in Engineering Policy: {len(eng_chunks)}")
print(f"Number of chunks in Onboarding Policy: {len(onb_chunks)}")
print(f"Number of chunks in Product Policy: {len(prod_chunks)}")
print(f"Number of chunks in Security Policy: {len(sec_chunks)}")


# ------------------------------------------------------------------------
# SECTION 8: COMBINE ALL CHUNKS INTO ONE KNOWLEDGE BASE
# ------------------------------------------------------------------------
# Merge chunks from every document into a single list. This unified list
# represents the full retrievable knowledge base that will later be
# embedded and indexed for similarity search.

all_chunks = hr_chunks + eng_chunks + onb_chunks + prod_chunks + sec_chunks
print(f"Total number of chunks in my knowledge base: {len(all_chunks)}")


# ------------------------------------------------------------------------
# SECTION 9: LOAD SENTENCE TRANSFORMER MODEL FOR EMBEDDINGS
# ------------------------------------------------------------------------
# Import SentenceTransformer, which converts text into dense vector
# embeddings that capture semantic meaning. These embeddings are what
# allow similarity-based retrieval later on (e.g. cosine similarity
# between a query embedding and chunk embeddings).

from sentence_transformers import SentenceTransformer

# Quick smoke-test sentences to verify the model loads and encodes
# correctly before running it over the full chunked knowledge base.
sentences = [
    "This is an example sentence",
    "each sentence is converted"
]

# Load the pretrained 'all-MiniLM-L6-v2' model — a lightweight, fast
# sentence-embedding model commonly used for semantic search / RAG.
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encode the test sentences into embeddings (vector representations).
embeddings = model.encode(sentences)

print("#" * 100)
print("embedding size is  :: ", len(embeddings))
print(embeddings)
print("#" * 100)

# Print the dimensionality of a single embedding vector (i.e. how many
# floating-point numbers represent one sentence) along with the full
# embeddings array again for inspection.
print(len(embeddings[0]))
print(embeddings)


### Ingest Chunks into ChromaDB
"""
Creates a Chroma client — your connection/handle to the vector database.
By default, this is an in-memory, ephemeral database — meaning all data lives in RAM and is lost when your
Python process ends (no data persists to disk unless you configure persistence explicitly, e.g., chromadb.PersistentClient(path="...")).
"""
import chromadb
chromaClient = chromadb.Client()
"""
Creates a collection — think of this like a "table" in a traditional database, but specifically designed to hold:
Documents (raw text chunks)
Embeddings (vector representations of those texts)
Metadata (extra info tags, e.g., source, date, category)
IDs (unique identifiers for each entry)
Named 'companyDocs' — likely intended to store chunks of company documents (policies, reports, manuals, etc.) for later retrieval.
"""
collection = chromaClient.create_collection(name='companyDocs')



documents = []
ids = []
metadatas = []

for i, chunk in enumerate(all_chunks):
    documents.append(chunk['text'])
    ids.append(f"chunk_{i}")
    metadatas.append({"source": chunk['source']})

print("*"*100)
print("documents::",  documents[50])
print("id :: ",ids[50])
print("metadat :: ",metadatas[50])
print("*"*100)

# Add all the text chunks to the ChromaDB collection.
# Chroma will store each document along with its unique ID
# and the metadata associated with that document.
#
# If an embedding function was configured when the collection
# was created, Chroma will also generate embeddings for these
# documents automatically.

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)
print(f"Stored {len(documents)} chunks in chroma db")
print(f"Sourced from 5 different files")


results = collection.peek()
print(results)


### Understanding cosine similarity and Embeddings
from sentence_transformers import SentenceTransformer
sentence1 = "I love going out, trekking, riding bikes"
sentence2 = "I am fond of solo travelling, especially hiking adventurous traits"
sentence3 = "I hate politics and political news"

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings1 = model.encode(sentence1)
embeddings2 = model.encode(sentence2)
embeddings3 = model.encode(sentence3)

print(embeddings1.shape)
print(embeddings2.shape)
print(embeddings3.shape)


# stack embeddings into a single matrix
embeddings = np.vstack([embeddings1, embeddings2, embeddings3])

# compute cosine similarity matrix(3 * 3)
similarity_matrix = cosine_similarity(embeddings)
print(similarity_matrix)

# labels for readability
labels = ['Sentence1\n(trekking/biking)', 'Sentence2\n(solo travelling)', 'sentence3\n(politics)']

plt.figure(figsize=(6,5))

sns.heatmap(
    similarity_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    xticklabels=labels,
    yticklabels=labels,
    vmin=0,vmax=1,
    square=True

)

plt.title('Cosine Similarity between sentences')
plt.tight_layout()
plt.show()


### Retriever Pipeline
def retrieve(question, n_results=3):
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    return results['documents'][0], results['metadatas'][0]


chunks, sources = retrieve("What is the work from home policy?", 3)

print(chunks)
print('='*60)
print(sources)


### Create RAG Pipeline
Settings.validate()
client = OpenAI(
    api_key=Settings.GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def ask_rag(question, n_results=3, verbose=True):
    chunks, sources = retrieve(question, n_results)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print(f"{'-' * 60}")
        print(f"Retrieved {len(chunks)} chunks")

        for i, (chunk, source) in enumerate(zip(chunks,sources)):
            print(f"[{source['source']}]  {chunk}...")
        print(f"{'-' * 60}")

    context = '\n\n'.join(chunks)
    messages = [
        ChatCompletionSystemMessageParam(
            role="system",
            content='''
                    You are a helpful company assistant. Answer questions using only the provided context.
                    If the context does not contain the answer, say I dont know or i dont have enough information to answer this
                    Be concise
                '''
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content= f"Context:\n{context}\n\n Question: {question}"
        )
    ]



    response = client.chat.completions.create(
    model="gemini-3.6-flash",
    messages=messages
    )

    answer = response.choices[0].message.content

    if verbose:
        print(f"Answer: {answer}")
        print(f"{'-' * 60}")

    return answer



ask_rag("WHat is work from home policy?")

