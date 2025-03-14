+++
date = '2025-03-11T22:47:36Z'
draft = false
title = 'Shakespeare and the Bible: An AI Investigation'
categories = ['Ollama', 'PostgreSQL', 'pgvector']
tags = ['Ollama', 'PostgreSQL', 'pgvector']
+++

## Summary

Could the greatest playwright of all time have secretly shaped one of the most influential religious texts in history? Some believe William Shakespeare left his mark on the King James Bible hidden in plain sight. With the power of AI, we’ll investigate whether there’s any truth to this conspiracy.

You can read about the conspiracy here:
- [Did Shakespeare write the King James Bible?](https://www.eden.co.uk/blog/did-shakespeare-write-the-king-james-bible) 
- [Shakespeare and the King James Bible – Some Tentative Conclusions](https://quiteirregular.wordpress.com/2019/03/21/shakespeare-and-the-king-james-bible-some-tentative-conclusions/)
- [Shakespeare and the Psalms Mystery: Did Shakespeare help write the King James Bible?](https://www.amazon.co.uk/Shakespeare-Psalms-Mystery-write-James/dp/1912067595) this is a book on the subject.

- The general conclusion from these sources is that the theory is false.
- Shakespeare lived during the time of the King James Bible translation, but there is no historical evidence linking him to the project.

Here, I’m going to use machine learning to investigate whether this theory holds any weight.

This post builds upon [PostgreSQL for AI: Storing and Searching Embeddings with pgvector]({{< relref "post/pgvector.md" >}}). 


![](/img/shakespeare.jpg)

---

## The conspiracy theory

The core belief is that William Shakespeare, either alone or as a key member of the translation team, played a significant role in authoring the King James Version (KJV) of the Bible, published in 1611.

### **Evidence** and Interpretations:
* **Linguistic Similarities**: Proponents point to perceived stylistic similarities between Shakespeare's works and the KJV, suggesting a common author.
* **Psalm 46**:
The 46th word from the beginning of the psalm is "shake."
The 46th word from the end of the psalm (when you omit "Selah," a Hebrew musical notation) is "spear."
This "shake-spear" combination is then taken as evidence of William Shakespeare's hidden signature within the text. Interestingly, the King James Bible was completed in 1611 the same year Shakespeare turned 46
* **The Era**: 
The fact that Shakespeare lived during the time the KJV was translated adds to the intrigue for some.

---

## Determining whether Shakespeare wrote parts of the bible

To test whether Shakespeare contributed to the Bible, we need an approach similar to plagiarism detection. Since plagiarism detection is not absolute but probabilistic, we rely on similarity thresholds and statistical comparisons.

### **✅ Key Indicators for a Strong Conclusion**
For a **strong case that Shakespeare contributed to the Bible**, we should look for **multiple strong indicators across different similarity metrics**.

| **Factor** | **What Would Be Indicative?** |
|------------|-------------------------------|
| **Vector Similarity (PGVector)** | **Above 0.85** for multiple Bible verses and Shakespeare sentences |
| **TF-IDF Cosine Similarity** | **Above 0.90** (very strong overlap in word choice) |
| **Jaccard Similarity** | **Above 0.85** (significant word overlap) |
| **N-Gram Overlap** | **Above 0.80** (strong similarity in phrasing) |
| **Levenshtein Similarity** | **Above 0.80** (high similarity despite minor changes) |
| **Consistent High Similarity Across Multiple Passages** | Many different verses match Shakespeare’s writing style |
| **Matching Unique Shakespearean Phrases** | Direct phrases appearing in both texts |
| **Linguistic & Stylistic Analysis** | Matches in rhythm, word patterns, and syntax |
| **External Historical Evidence** | Shakespeare was known to be involved in early Bible translations (King James Version) |

💡 **If multiple methods consistently show strong similarity (≥ 0.85)**, it suggests Shakespeare may have influenced those Bible passages.


### **📌 What Would Be Inconclusive?**
🔸 **If only a few Bible verses match Shakespeare closely** (chance similarity)  
🔸 **If scores are mostly between 0.60 – 0.75** (moderate similarity, but not definitive)  
🔸 **If no unique Shakespearean phrases appear** (high similarity could be coincidental)  
🔸 **If stylistic/literary analysis contradicts findings**  


### **✅ How Plagiarism Tests Work in Academia**
Plagiarism detection software (e.g., **Turnitin**, **Grammarly Plagiarism Checker**) uses **similarity thresholds**:
- **Above 90%** → **Definite plagiarism** (strong evidence of copying)  
- **80% - 89%** → **Highly suspect** (likely copied with some modifications)  
- **60% - 79%** → **Possible, but needs further investigation**  
- **Below 60%** → **Likely coincidental or common phrasing**  

📌 **For Shakespeare & the Bible:**  
- **A similarity level of 85%+ across multiple passages** would be **highly suggestive**.
- **A single match above 85% is not enough** we need **consistent results** across many verses.

---


## Running the Analysis


### Step 1: Loading our data

I'm using the King James Version of the Bible and I'm using the complete works of Shakespeare from Gutenberg I'm going to load the verses of the Bible into the database one verse at a time. I'm going to load the sentences from Project Gertenberg into another database table.

#### Load bible text

```python
import re
import csv

# Load Bible text file
with open("./data/kjvdat.txt", "r", encoding="utf-8") as f:
    bible_text = f.readlines()

print(bible_text[:50])
```
```
['Gen|1|1| In the beginning God created the heaven and the earth.~\n', 'Gen|1|2| And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters.~\n' ..]
```


#### Load shakespeare text

```python
# Function to split text into sentences & insert into database

file_path = "./data/shakespeare.txt"
text = ""
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Split text into sentences using regex (handles '.', '!', '?')
sentences = re.split(r'(?<=[.!?])\s+', text)

shakespeare_texts = []
for sentence in sentences:
    shakespeare_texts.append(sentence.strip())

```

---

### Step 2: Connect to PostgreSQL and create tables

```python
import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables from .env file
load_dotenv()

def get_connection():
    """Establishes a secure connection using environment variables."""
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

conn = get_connection()
cursor = conn.cursor()
```

Create the embeddings tables to hold the data

```python
import psycopg2

conn = get_connection()
cursor = conn.cursor()

# Enable pgvector extension
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Create table for Bible embeddings
cursor.execute("""
    CREATE TABLE IF NOT EXISTS bible_embeddings (
        id SERIAL PRIMARY KEY,
        verse TEXT NOT NULL,
        text TEXT NOT NULL,
        embedding vector(1024) -- Adjust dimension based on your model
    );
""")

# Create table for Shakespeare embeddings
cursor.execute("""
    CREATE TABLE IF NOT EXISTS shakespeare_embeddings (
        id SERIAL PRIMARY KEY,
        ref INTEGER NOT NULL,
        sentence TEXT NOT NULL,
        embedding vector(1024) -- Adjust dimension based on your model
    );
""")

conn.commit()
conn.close()

print("✅ PostgreSQL tables for Bible & Shakespeare embeddings created successfully!")
```

---

### Step 3: Load the embeddings for each of the sentences and verses

As before we will use ollama to generate embeddings.
We will default to the model: `mxbai-embed-large` 

```python
import ollama

# Function to get embeddings using Ollama (with error handling)
def get_embedding(text):
    try:
        embedding_data = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        embedding_data = embedding_data["embedding"]  # Extract embedding
        return embedding_data
    except Exception as e:
        print(f"❌ Unexpected error generating embedding: {e}")
    return None  # Return None if an error occurs

```

Create and insert the bible embeddings.

```python
import psycopg2

# Function to insert Bible embeddings into PostgreSQL
def store_bible_embeddings(bible_texts):
    print(f"🔄 Storing Bible embeddings in PostgreSQL... {len(bible_texts)}")
    conn = get_connection()
    cursor = conn.cursor()

    total = len(bible_texts)
    for i, (verse, text) in enumerate(bible_texts, start=1):
        embedding = get_embedding(text)
        cursor.execute(
            "INSERT INTO bible_embeddings (verse, text, embedding) VALUES (%s, %s, %s);",
            (verse, text, embedding)
        )

        if i % 1000 == 0 or i == total:
            percent_done = (i / total) * 100
            print(f"🔄 Inserted {i}/{total} Bible verses ({percent_done:.2f}% done)")
        conn.commit()

    conn.close()
    print("✅ Bible embeddings stored in PostgreSQL!")
store_bible_embeddings(bible_texts)

```

```
🔄 Storing Bible embeddings in PostgreSQL... 31102
🔄 Inserted 1000/31102 Bible verses (3.22% done)
🔄 Inserted 2000/31102 Bible verses (6.43% done)
...
🔄 Inserted 31000/31102 Bible verses (99.67% done)
🔄 Inserted 31102/31102 Bible verses (100.00% done)
✅ Bible embeddings stored in PostgreSQL!
```

Create and insert the shakespeare sentences.

```python
# Function to insert Shakespeare embeddings into PostgreSQL
def store_shakespeare_embeddings(shakespeare_sentences):
    conn = get_connection()
    cursor = conn.cursor()

    total = len(shakespeare_sentences)
    for i, (ref, sentence) in enumerate(shakespeare_sentences, start=1):
        embedding = get_embedding(sentence)
        cursor.execute(
            "INSERT INTO shakespeare_embeddings (ref, sentence, embedding) VALUES (%s, %s, %s);",
            (ref, sentence, embedding)
        )

        if i % 1000 == 0 or i == total:
            percent_done = (i / total) * 100
            print(f"🔄 Inserted {i}/{total} Shakespeare sentences ({percent_done:.2f}% done)")

    conn.commit()
    conn.close()
    print("✅ Shakespeare embeddings stored in PostgreSQL!")

store_shakespeare_embeddings(shakespeare_texts)

```

```
🔄 Inserted 1000/70307 Shakespeare sentences (1.42% done)
...
🔄 Inserted 70307/70307 Shakespeare sentences (100.00% done)
✅ Shakespeare embeddings stored in PostgreSQL!
```

---

### Step 4: Find similar sentences

This is an example function to search for similar sentences.


```python

def find_similar_bible_verses(shakespeare_sentence, top_k=5):
    embedding = get_embedding(shakespeare_sentence)  # Get embedding as a list

    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT verse, text, 1 - (embedding <=> %s) AS similarity
    FROM bible_embeddings
    ORDER BY embedding <=> %s
    LIMIT %s;
    """
    cursor.execute(query, (embedding, embedding, top_k))
    results = cursor.fetchall()
    conn.close()

    return results

# Example Query
query_text = "To be, or not to be, that is the question."
bible_matches = find_similar_bible_verses(query_text)

print(f"\n📖 **Bible Verses Similar to Shakespeare's Quote: {query_text}**")
for verse, text, similarity in bible_matches:
    print(f"{verse} - {text} (Similarity: {similarity:.4f})")


```

---

### Step 5: Create a table to store the simlar sentences

```python
# Connect to PostgreSQL
conn = get_connection()
cursor = conn.cursor()

# Create the similarity table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS bible_shakespeare_similarities (
        id SERIAL PRIMARY KEY,
        bible_verse TEXT NOT NULL,
        shakespeare_sentence TEXT NOT NULL,
        tfidf_cosine REAL,
        jaccard_similarity REAL,
        ngram_overlap REAL,
        levenshtein_similarity REAL,
        vector_similarity REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

conn.commit()
conn.close()

print("✅ PostgreSQL table 'bible_shakespeare_similarities' created successfully!")
```

```
✅ PostgreSQL table 'bible_shakespeare_similarities' created successfully!
```

---

### Step 6: Find bible verses similar to the shakespeare sentences

**Note**: There are 30K+ bible verses and 70K shakespeare sentences in our test set. Brute force comparison will be about 2.1 billion entries which is a bit large. 
I am going to take a different approach I am going to search for the most similar vectors and work with them.


Find similar bible verses

```python

# Function to find the most similar Bible verses for a given Shakespeare sentence
def find_top_similar_bible_verses(shakespeare_sentence, top_k=5, similarity_threshold=0.80):
    embedding = get_embedding(shakespeare_sentence)  # Get embedding

    conn = get_connection()
    cursor = conn.cursor()

    query = f"""
    SELECT id, verse, text, 1 - (embedding <=> %s) AS similarity
    FROM bible_embeddings
    ORDER BY embedding <=> %s
    LIMIT %s;
    """
    cursor.execute(query, (str(embedding), str(embedding), top_k))
    results = cursor.fetchall()
    conn.close()

    # Filter out low-similarity results based on threshold
    filtered_results = [(bid, verse, text, sim) for bid, verse, text, sim in results if sim >= similarity_threshold]
    return filtered_results  # Only return high-confidence matches
```
When I run  this test I get this output

```
📖 **Bible Verses Similar to Shakespeare's Quote: To be, or not to be, that is the question.**
Co2 8:12 - For if there be first a willing mind, it is accepted according to that a man hath, and not according to that he hath not.~ (Similarity: 0.6588)
Mat 11:3 - And said unto him, Art thou he that should come, or do we look for another?~ (Similarity: 0.6555)
Kg2 2:10 - And he said, Thou hast asked a hard thing: nevertheless, if thou see me when I am taken from thee, it shall be so unto thee; but if not, it shall not be so.~ (Similarity: 0.6362)
Job 39:9 - Will the unicorn be willing to serve thee, or abide by thy crib?~ (Similarity: 0.6322)
Gal 1:10 - For do I now persuade men, or God? or do I seek to please men? for if I yet pleased men, I should not be the servant of Christ.~ (Similarity: 0.6302)
```


Lets save these results in our similarity table.

```python

def store_top_similarities(shakespeare_id, shakespeare_sentence, bible_results):
    """
    Stores the top similarity matches in the bible_shakespeare_similarities table.
    """
    if not bible_results:
        return  # Skip if no high-similarity matches

    conn = get_connection()
    cursor = conn.cursor()

    for bible_id, verse, text, similarity in bible_results:
        cursor.execute("""
            INSERT INTO bible_shakespeare_similarities 
            (bible_verse, shakespeare_sentence, vector_similarity)
            VALUES (%s, %s, %s);
        """, (text, shakespeare_sentence, similarity))

    conn.commit()
    conn.close()
    print(f"✅ Stored matches for Shakespeare sentence ID {shakespeare_id}")

```


Process all the shakespeare sentences storing the most similar (>.80) in the table.

```python

def process_all_shakespeare_sentences(shakespeare_sentences, top_k=5, similarity_threshold=0.80):
    """
    Loops through all Shakespeare sentences, finds the most similar Bible verses,
    and stores the results in the database.
    """
    total_sentences = len(shakespeare_sentences)

    for i, shakespeare_sentence in enumerate(shakespeare_sentences, start=1):
        similar_bible_verses = find_top_similar_bible_verses(shakespeare_sentence, top_k, similarity_threshold)

        # Store only high-similarity results
        store_top_similarities(i, shakespeare_sentence, similar_bible_verses)

        # Log progress every 1000 searches
        if i % 1000 == 0 or i == total_sentences:
            percent_done = (i / total_sentences) * 100
            print(f"🔄 Processed {i}/{total_sentences} Shakespeare sentences ({percent_done:.2f}% done)")

# Fetch Shakespeare sentences and process them
shakespeare_sentences = fetch_shakespeare_texts()  # Fetches all sentences from Shakespeare's works
process_all_shakespeare_sentences(shakespeare_sentences)

```

```
...
✅ Stored matches for Shakespeare sentence ID 69835
🔄 Processed 70000/70307 Shakespeare sentences (99.56% done)
🔄 Processed 70307/70307 Shakespeare sentences (100.00% done)
```

This will give us a subset of the total data to work with further

---

### Step 7: Define further similarity tests

Now that we have stored **similar Bible–Shakespeare sentence pairs**, we will **compute additional similarity metrics** for each pair in the `bible_shakespeare_similarities` table.

#### **🔍 Similarity Metrics for Detecting Common Authorship**  
Each of these similarity measures helps determine **how closely two sentences resemble each other**. While no single metric is perfect, using multiple approaches provides a **more reliable** assessment of whether two sentences could have been written by the same author.

---

#### **1️⃣ Levenshtein Similarity (Edit Distance)**
##### **📌 What It Is?**  
Levenshtein Distance measures the **minimum number of edits** (insertions, deletions, substitutions) required to turn one sentence into another.

##### **⚙️ What It Does?**  
- Identifies **small variations** in wording and spelling.  
- Captures **typos, rewording, and minor differences** between two sentences.  

##### **🤔 Why Use It for Common Authorship?**  
- If two sentences have **very few changes (Levenshtein similarity > 0.80)**, they may have been written by the same person with **slight rewording**.  
- Helps detect **self-editing**—authors often modify their phrasing **while keeping the core meaning intact**.  

🔹 **Example:**  
```
"The quick brown fox jumps over the lazy dog."  
→ "The fast brown fox leaped over the lazy dog."  
Levenshtein similarity: **0.85 (High)**
```

---

##### **2️⃣ Jaccard Similarity (Word Overlap)**
##### **📌 What It Is?**  
Jaccard Similarity compares the **sets of unique words** in two sentences and measures how much they overlap.

##### **⚙️ What It Does?**  
- Measures **exact word overlap** between sentences.  
- Ignores word order but focuses on **shared vocabulary**.  

##### **🤔 Why Use It for Common Authorship?**  
- Writers tend to **reuse the same vocabulary** across different works.  
- If two sentences **share a high percentage of words (Jaccard > 0.85)**, they may have been written by the same person.

🔹 **Example:**  
```
Sentence 1: "All the world’s a stage, and men are merely players."  
Sentence 2: "The world is a stage, and people play their parts."  
Jaccard similarity: **0.80 (High)**
```
✔ **Indicates strong authorial similarity due to shared words.**  

---

##### **3️⃣ N-Gram Overlap (Phrase Similarity)**
##### **📌 What It Is?**  
N-Gram Similarity looks at **small sequences of words** (n-grams) instead of individual words. A **higher overlap** suggests **similar phrasing or structure**.

##### **⚙️ What It Does?**  
- Captures **phrase-level similarities**, not just word overlap.  
- More **resistant to rewording** than Jaccard Similarity.  

##### **🤔 Why Use It for Common Authorship?**  
- Writers tend to have **specific phrasings or word patterns** they favor.  
- If **common multi-word sequences (N-grams) appear across different texts**, it suggests a **consistent writing style**.

🔹 **Example (3-gram analysis):**  
```
Sentence 1: "To be, or not to be, that is the question."  
Sentence 2: "To be or not to be is the real question."  
3-Gram overlap similarity: **0.75 (Moderate-High)**
```
✔ Suggests that the **writer uses similar phrasing patterns**.

---

##### **4️⃣ TF-IDF Cosine Similarity (Word Frequency Match)**
##### **📌 What It Is?**  
TF-IDF (Term Frequency - Inverse Document Frequency) measures **how important a word is** in a sentence while reducing the impact of common words like **“the”, “is”, “and”**.

##### **⚙️ What It Does?**  
- Measures **word importance** rather than just frequency.  
- Uses **cosine similarity** to compare sentence vectors.  

##### **🤔 Why Use It for Common Authorship?**  
- **Authors often emphasize the same words** in different works.  
- If two sentences have a **high TF-IDF similarity (> 0.85)**, it means they **use similar important words**.

🔹 **Example:**  
```
Sentence 1: "Love looks not with the eyes, but with the mind."  
Sentence 2: "True love sees not with the eyes, but with thought."  
TF-IDF Cosine Similarity: **0.88 (Very High)**
```
✔ Suggests **a similar writing style based on word importance**.

---

##### **📊 Which Metric is Best for Common Authorship?**
| **Metric**        | **Best For**                           | **When High Similarity Suggests Same Author** |
|-------------------|----------------------------------|------------------------------------|
| **Levenshtein**   | Small edits, minor rewording   | When similarity **> 0.80** |
| **Jaccard**       | Exact word overlap             | When similarity **> 0.85** |
| **N-Gram**        | Phrase and structure match     | When similarity **> 0.75** |
| **TF-IDF Cosine** | Word importance match         | When similarity **> 0.85** |

✔ **Best Approach? Use all four together** → If all scores are **high**, it's a strong sign of common authorship!  

---

##### **✅ Steps to Compute Full Results**
1. **Fetch pairs from `bible_shakespeare_similarities`**
2. **Compute similarity metrics**
   - **Levenshtein Similarity** (edit distance)
   - **Jaccard Similarity** (word overlap)
   - **N-Gram Overlap** (phrase similarity)
   - **TF-IDF Cosine Similarity** (word frequency match)
3. **Update the table with computed scores**
4. **Log progress every 1000 updates**

---

##### **1️⃣ Define Similarity Functions**
```python
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.util import ngrams

nltk.download('punkt')

# ✅ Levenshtein Similarity (Edit Distance)
def levenshtein_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

# ✅ Jaccard Similarity (Word Overlap)
def jaccard_similarity(text1, text2):
    set1, set2 = set(text1.lower().split()), set(text2.lower().split())
    return len(set1 & set2) / len(set1 | set2)

# ✅ N-Gram Similarity (Phrase Overlap)
def ngram_overlap(text1, text2, n=3):
    ngrams1, ngrams2 = set(ngrams(text1.split(), n)), set(ngrams(text2.split(), n))
    return len(ngrams1 & ngrams2) / max(len(ngrams1 | ngrams2), 1)

# ✅ TF-IDF Cosine Similarity
def compute_tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
```

```
[nltk_data] Downloading package punkt to C:\nltk_data...
[nltk_data]   Unzipping tokenizers\punkt.zip.
```



---

##### **2️⃣ Fetch High-Similarity Pairs from PostgreSQL**
We retrieve only the **Bible–Shakespeare pairs** that need additional analysis.

```python
import psycopg2

def fetch_high_similarity_pairs():
    """
    Fetches all Bible–Shakespeare pairs that need additional similarity metrics.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, bible_verse, shakespeare_sentence
        FROM bible_shakespeare_similarities
        WHERE tfidf_cosine IS NULL;  -- Fetch only rows missing similarity scores
    """)
    
    results = cursor.fetchall()
    conn.close()
    return results  # Returns list of (id, bible_verse, shakespeare_sentence)
```

---

##### **3️⃣ Compute Similarity Scores & Update the Database**
Now, we **compute scores** for each Bible–Shakespeare pair and update the table.

```python
def update_additional_similarities():
    """
    Computes Levenshtein, Jaccard, N-Gram, and TF-IDF scores and updates the database.
    """
    pairs = fetch_high_similarity_pairs()  # Get Bible-Shakespeare pairs
    total_pairs = len(pairs)

    conn = get_connection()
    cursor = conn.cursor()

    for i, (pair_id, bible_verse, shakespeare_sentence) in enumerate(pairs, start=1):
        # Compute similarity scores
        levenshtein_score = levenshtein_similarity(bible_verse, shakespeare_sentence)
        jaccard_score = jaccard_similarity(bible_verse, shakespeare_sentence)
        ngram_score = ngram_overlap(bible_verse, shakespeare_sentence, n=3)
        tfidf_score = compute_tfidf_similarity(bible_verse, shakespeare_sentence)

        # Update PostgreSQL record with new scores
        cursor.execute("""
            UPDATE bible_shakespeare_similarities
            SET levenshtein_similarity = %s, 
                jaccard_similarity = %s, 
                ngram_overlap = %s, 
                tfidf_cosine = %s
            WHERE id = %s;
        """, (float(levenshtein_score), float(jaccard_score), float(ngram_score), float(tfidf_score), pair_id))

        print(f"🔄 Updated similarity metrics for pair {bible_verse} - {shakespeare_sentence}")
        # Log progress every 1000 pairs
        if i % 1000 == 0 or i == total_pairs:
            percent_done = (i / total_pairs) * 100
            print(f"🔄 Processed {i}/{total_pairs} pairs ({percent_done:.2f}% done)")

    conn.commit()
    conn.close()
    print("✅ Similarity metrics updated in PostgreSQL!")
```

---

##### **4️⃣ Add Columns for Storing Additional Metrics (If Needed)**
If your table **doesn’t already have** columns for these similarity metrics, add them.

```python
conn = get_connection()
cursor = conn.cursor()

cursor.execute("""
    ALTER TABLE bible_shakespeare_similarities
    ADD COLUMN IF NOT EXISTS tfidf_cosine REAL,
    ADD COLUMN IF NOT EXISTS jaccard_similarity REAL,
    ADD COLUMN IF NOT EXISTS ngram_overlap REAL,
    ADD COLUMN IF NOT EXISTS levenshtein_similarity REAL;
""")

conn.commit()
conn.close()
print("✅ Added similarity columns to PostgreSQL table!")
```

---

##### **5️⃣ Run the Full Analysis**
Now, run the script to compute and store **all similarity metrics**:

```python
update_additional_similarities()
```

```
🔄 Updated similarity metrics for pair And I was with you in weakness, and in fear, and in much trembling.~ - “Mine enemy was strong, my poor self weak,
And far the weaker with so strong a fear.
🔄 Processed 1358/1358 pairs (100.00% done)
✅ Similarity metrics updated in PostgreSQL!
```

### Step 8: Generate a report of our results

To generate a **report of the most similar passages** and their **percentage similarity**, we need to **query the `bible_shakespeare_similarities` table** and **group by Bible verses or Shakespeare sentences** to find the most relevant matches.

---

#### **✅ SQL Query to Find the Most Similar Passages**
This query selects **the highest similarity scores** for each Bible verse and Shakespeare sentence.

```sql
SELECT 
    bible_verse, 
    shakespeare_sentence, 
    vector_similarity AS embedding_similarity,
    tfidf_cosine AS tfidf_similarity,
    jaccard_similarity,
    ngram_overlap,
    levenshtein_similarity
FROM bible_shakespeare_similarities
ORDER BY vector_similarity DESC, tfidf_cosine DESC
LIMIT 50;  -- Adjust limit to get more results
```

**🔹 What This Query Does:**
- Retrieves **Bible verses & Shakespeare sentences** that are most similar.
- Orders by **highest vector (embedding) similarity** first.
- Uses **TF-IDF, Jaccard, N-Gram, and Levenshtein scores** for comparison.
- Limits the result to **50 top matches** (adjustable).

---

#### **📌 Grouping Results by Bible Verses**
If we want to find **the most similar Shakespeare passage for each Bible verse**, we use:

```sql
SELECT 
    bible_verse, 
    shakespeare_sentence, 
    MAX(vector_similarity) AS max_embedding_similarity,
    MAX(tfidf_cosine) AS max_tfidf_similarity
FROM bible_shakespeare_similarities
GROUP BY bible_verse
ORDER BY max_embedding_similarity DESC
LIMIT 50;
```

✔ This ensures that **each Bible verse appears only once** with **its closest Shakespeare match**.

---


#### **✅ Python Code to Generate the Report**
Once we have the SQL query, we can **fetch the results** and save them in a **CSV or display them**.

```python
import psycopg2
import pandas as pd

def generate_similarity_report():
    """
    Fetches the most similar Bible–Shakespeare passages and generates a report.
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT 
            bible_verse, 
            shakespeare_sentence, 
            vector_similarity AS embedding_similarity,
            tfidf_cosine AS tfidf_similarity,
            jaccard_similarity,
            ngram_overlap,
            levenshtein_similarity
        FROM bible_shakespeare_similarities
        ORDER BY vector_similarity DESC, tfidf_cosine DESC
        LIMIT 50;
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()

    # Convert to DataFrame
    df = pd.DataFrame(results, columns=[
        "Bible Verse", "Shakespeare Sentence", "Embedding Similarity",
        "TF-IDF Similarity", "Jaccard Similarity", "N-Gram Overlap", "Levenshtein Similarity"
    ])

    # Save to CSV
    df.to_csv("bible_shakespeare_similarity_report.csv", index=False)
    print("✅ Report generated: bible_shakespeare_similarity_report.csv")

    return df

# Generate and display the report
report_df = generate_similarity_report()
from IPython.display import display

# Generate and display the report in Jupyter Notebook
report_df = generate_similarity_report()

# Display the DataFrame
display(report_df)
```

|Bible Verse                                                                                                                                                         |Shakespeare Sentence                                                                       |Embedding Similarity|TF-IDF Similarity|Jaccard Similarity|N-Gram Overlap|Levenshtein Similarity|
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|--------------------|-----------------|------------------|--------------|----------------------|
|Thou shalt not steal.~                                                                                                                                              |“Thou shalt not steal”?                                                                    |0.92331326          |1.0              |0.33333334        |0.0           |0.8888889             |
|And he wrote a letter after this manner:~                                                                                                                           |Look, here’s a letter from him.                                                            |0.8985709           |0.09349477       |0.16666667        |0.0           |0.5833333             |
|Hear my prayer, O God; give ear to the words of my mouth.~                                                                                                          |My lord, I pray you, hear me.                                                              |0.8959283           |0.19976382       |0.11764706        |0.0           |0.16091955            |
|I am one that bear witness of myself, and the Father that sent me beareth witness of me.~                                                                           |This is a witness that I am thy son.                                                       |0.8837818           |0.23765056       |0.2               |0.0           |0.384                 |
|Hear my prayer, O God; give ear to the words of my mouth.~                                                                                                          |I pray you, let-a me speak a word with your ear.                                           |0.88193315          |0.04723062       |0.0               |0.0           |0.4528302             |
|Hear my prayer, O LORD, give ear to my supplications: in thy faithfulness answer me, and in thy righteousness.~                                                     |My lord, I pray you, hear me.                                                              |0.8800851           |0.27836126       |0.15              |0.0           |0.1                   |
..
---




### Step 9 Generate a Similarity Distribution Report

You can generate the **similarity distribution report using SQL** by grouping sentence pairs based on their **vector similarity ranges**.


```python
import pandas as pd
from IPython.display import display

def fetch_similarity_distribution():
    """
    Fetches the count of sentence pairs in different similarity percentage bins from the database.
    """
    # Establish database connection
    conn = get_connection()
    cursor = conn.cursor()

    # SQL query to categorize similarity scores
    query = """
        SELECT 
            CASE 
                WHEN vector_similarity >= 0.80 AND vector_similarity < 0.85 THEN '80-84%'
                WHEN vector_similarity >= 0.85 AND vector_similarity < 0.90 THEN '85-89%'
                WHEN vector_similarity >= 0.90 AND vector_similarity < 0.95 THEN '90-94%'
                WHEN vector_similarity >= 0.95 AND vector_similarity < 1.00 THEN '95-99%'
                WHEN vector_similarity = 1.00 THEN '100%'
            END AS similarity_range,
            COUNT(*) AS sentence_count
        FROM bible_shakespeare_similarities
        WHERE vector_similarity >= 0.80
        GROUP BY similarity_range
        ORDER BY similarity_range;
    """

    # Execute query
    cursor.execute(query)
    results = cursor.fetchall()

    # Close the database connection
    conn.close()

    # Convert results to a Pandas DataFrame
    df = pd.DataFrame(results, columns=["Similarity Range", "Sentence Count"])

    return df

# Generate and display the similarity distribution report
similarity_report_df = fetch_similarity_distribution()

# Display results in Jupyter Notebook
display(similarity_report_df)

# Save results to a CSV file
similarity_report_df.to_csv("similarity_distribution_report.csv", index=False)

print("✅ Report saved as 'similarity_distribution_report.csv'")

```

---

#### **📌 Explanation**
- **`CASE` statement** categorizes similarity scores into **80-84%, 85-89%, 90-94%, 95-99%, and 100% bins**.
- **`COUNT(*)`** counts the number of sentence pairs in each similarity bin.
- **`WHERE vector_similarity >= 0.80`** ensures we only include results with **at least 80% similarity**.
- **`GROUP BY similarity_range`** groups results by similarity range.
- **`ORDER BY similarity_range`** ensures results are sorted in increasing order.

---

#### **📊 Final Result**
|Similarity Range|Sentence Count|
|----------------|--------------|
|80-84%          |1289          |
|85-89%          |68            |
|90-94%          |1             |

![Bible Dist](/img/bible_dist.png)

---

## **🔍 Analysis of Results**

Our AI-driven analysis sought to determine whether Shakespeare contributed to the **King James Bible** by identifying linguistic similarities between his works and biblical verses. Using **vector embeddings, statistical similarity metrics, and direct phrase matching**, we evaluated Shakespeare's complete works to see if they shared a **unique authorial fingerprint**.

### **1️⃣ Key Findings**
📌 **Vector Embeddings:**  
- Over **1,200 Bible verses** showed **notable similarity (≥ 80%)** to Shakespeare’s sentences.  
- The highest-scoring match had a similarity of **~92.3%**, suggesting some degree of stylistic overlap.  
- However, high vector similarity alone **does not confirm authorship** it only suggests similar word usage and structure.

📌 **TF-IDF & Word-Based Similarity:**  
- The **TF-IDF cosine similarity** results were **inconsistent**.  
  - While some Shakespearean passages scored highly against Bible verses (**up to 100% similarity**), most hovered between **60-80%**, indicating only **moderate overlap** in important word usage.  

📌 **Jaccard & N-Gram Similarity:**  
- **Jaccard similarity** (measuring word overlap) showed **low scores overall**, meaning Shakespeare and the Bible **do not share many identical words** beyond common terms.  
- **N-Gram Overlap** (phrase-level similarity) also showed weak connections—suggesting that even if Shakespeare’s language and the Bible’s prose are somewhat related, they **do not share significant multi-word expressions**.

📌 **Levenshtein Similarity (Edit Distance):**  
- Even in the **most similar** cases, the **Levenshtein similarity rarely exceeded 85%**, indicating **significant textual differences** between Shakespeare and the Bible.
- This means that while Shakespearean and biblical writing styles **share poetic qualities**, they were **not written by the same author**.


### **2️⃣ What This Means**
🔹 The **vector-based similarities** suggest that Shakespearean language and biblical prose **share some stylistic elements**—possibly due to the poetic nature of both texts.  
🔹 However, **other statistical methods (Jaccard, N-Gram, Levenshtein)** did not support the idea that Shakespeare had a direct role in authoring Bible passages.  
🔹 If Shakespeare had **directly contributed** to the King James Bible, we would expect to see **higher and more consistent similarity scores across multiple metrics**—which was **not the case**.  

### **3️⃣ Why These Results Occur**
The **linguistic similarities** between Shakespeare and the Bible may be explained by:
- **Common Early Modern English phrasing:** The Bible and Shakespeare’s works were written in the **same historical period**, using similar syntax, vocabulary, and poetic structures.
- **Shared literary influences:** Both texts drew from **Latin, Greek, and early English translations of religious texts**, leading to natural overlaps.
- **Stylistic conventions of the time:** The **rhythmic and dramatic** elements found in Shakespeare’s works also appear in the Bible, but this was **a common feature of English literature at the time**.


### **Next Steps & Further Research**
1️⃣ **Refining the Sentence-Level Comparison:**  
- We could **split Bible verses into individual sentences** instead of full verses to enable a **more direct** comparison with Shakespeare’s works.  

2️⃣ **Running a Full Brute-Force Comparison (21B Entries):**  
- While computationally expensive, a **full exhaustive comparison** could yield **deeper insights** by checking **every single** Shakespearean sentence against **every single** Bible verse.  


---

## Code data and examples

You can find the code/data etc. used for this post here: 

[shakespeare bible](https://github.com/ernanhughes/shakespeare-bible)

---


## Conclusion

Our analysis found some superficial similarities between Shakespeare's works and the King James Bible. However, these were largely explainable by common linguistic patterns rather than direct authorship.

The most telling result is that while vector embeddings showed similarity (1200+ close matches), the other similarity metrics did not confirm a strong link. This suggests that while Shakespearean language and biblical phrasing have some overlap, there is no conclusive evidence of Shakespeare's authorship in the King James Bible.

**Final Verdict**: While Shakespeare’s works and the King James Bible share stylistic elements, our AI analysis found no concrete evidence of his authorship. The similarities we detected can be explained by linguistic patterns rather than direct authorship.

The conspiracy may remain alive in whispers and speculation, but as far as the data shows Shakespeare was not a secret author of the Bible.
