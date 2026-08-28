

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


print(hr_document)
print('='*100)
print('#'*100)
print(eng_document)
print('='*100)
print('#'*100)
print(onb_document)
print('='*100)
print('#'*100)
print(prod_document)
print('='*100)
print('#'*100)
print(sec_document)




print(f"HR Document Length: {len(hr_document)}")
print(f"Engineering Document Length: {len(eng_document)}")
print(f"Onboarding Document Length: {len(onb_document)}")
print(f"Product Document Length: {len(prod_document)}")
print(f"Security Document Length: {len(sec_document)}")

print('=' * 100)
print('#' * 100)

print(f"HR Policy Token Count: {len(hr_document.split())}")
print(f"Engineering Standards Token Count: {len(eng_document.split())}")
print(f"Onboarding Document Token Count: {len(onb_document.split())}")
print(f"Product Knowledge Document Token Count: {len(prod_document.split())}")
print(f"Security Document Token Count: {len(sec_document.split())}")


## Chunking the documents (Fixed size, Overlapping, Recursive, Semantic)

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

hr_chunks = chunk_document(hr_document, 'HR Policy')
print("hr chunk :: ",hr_chunks)
eng_chunks = chunk_document(eng_document, 'Engineering Policy')
print("eng chunk :: ",eng_chunks)
onb_chunks = chunk_document(onb_document, 'Onboarding Policy')
print("onb chunk :: ",onb_chunks)
prod_chunks = chunk_document(prod_document, 'Product Policy')
print("prod chunk :: ",prod_chunks)
sec_chunks = chunk_document(sec_document, 'Security Policy')
print("sec chunk :: ",sec_chunks)


print(f"Number of chunks in HR Policy: {len(hr_chunks)}")
print(f"Number of chunks in Engineering Policy: {len(eng_chunks)}")
print(f"Number of chunks in Onboarding Policy: {len(onb_chunks)}")
print(f"Number of chunks in Product Policy: {len(prod_chunks)}")
print(f"Number of chunks in Security Policy: {len(sec_chunks)}")


all_chunks = hr_chunks + eng_chunks + onb_chunks + prod_chunks + sec_chunks
print(f"Total number of chunks in my knowledge base: {len(all_chunks)}")


## Sentence Transformers for converting text -> embeddings
from sentence_transformers import SentenceTransformer
sentences = [
    "This is an example sentence",
    "each sentence is converted"
]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print("#"*100)
print("embedding size is  :: ",len(embeddings))
print(embeddings)
print("#"*100)
print(len(embeddings[0]))
print(embeddings)
