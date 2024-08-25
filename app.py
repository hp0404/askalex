from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd
import pyalex
from openai import OpenAI

RESEARCH_QUESTION = "What is the role of BRCA2 in breast cancer?"
MAX_WORDS = 500
MAX_LENGTH = 300

model = "phi3.5:3.8b-mini-instruct-q4_0"
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
search_template = (
    "I would like to search the literature to find answer for the following question. "
    + "Give me 2 to 3 keywords that I should include in my literature search. "
    + 'List the most important keyword first and concatenate them by "+". '
    + 'Make them concise, for example: use "ABCC1" instead of "ABCC1 gene". '
    + "For example, for the question "
    + '"What is the biological rationale for an association between the gene ABCC1 and cardiotoxicity?" '
    + 'The keywords are "ABCC1+cardiotoxicity+biological rationale". '
    + "\n\nQuestion: {question}\nAnswer: "
)

rag_template = (
    "You are an intelligent assistant helping users with their questions. "
    + "Use 'you' to refer to the individual asking the questions even if they ask with 'I'. "
    + "Answer the following question using only the data provided in the sources below. "
    + "For tabular information return it as an html table. Do not return markdown format. "
    + "If you cannot answer using the sources below, say you don't know. "
    + "\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer: "
)


@lru_cache(maxsize=None)
def answer(
    prompt: str,
    max_tokens: int = 1024,
    model: str = "phi3.5:3.8b-mini-instruct-q4_0",
    temperature: float = 0.3,
) -> str:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        model=model,
        temperature=temperature,
    )
    return response.choices[0].message.content


def get_keywords(question: str) -> str:
    keywords = answer(prompt=search_template.format(question=question))
    return keywords.split("\n")[0].strip()


def remove_last_keyword(s: str) -> str:
    return s.rsplit("+", 1)[0]


def shorten_abstract(text: str) -> str:
    words = text.split()
    return " ".join(words[:MAX_LENGTH]) if len(words) > MAX_WORDS else text


@lru_cache(maxsize=None)
def get_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_works(keywords: str) -> Optional[List[dict]]:
    while keywords:
        works = pyalex.Works().search_filter(abstract=keywords).get(per_page=100)
        if works:
            return works
        keywords = remove_last_keyword(keywords)
    return None


def create_abstracts_dataframe(works: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "title": e["title"],
                "abstract": shorten_abstract(e["abstract"]),
                "url": e["doi"],
            }
            for e in works
        ]
    )
    df["embedding"] = df.abstract.apply(get_embedding)
    return df


def search_docs(df: pd.DataFrame, query: str, top_n: int = 10) -> Optional[str]:
    if df is None:
        return None

    query_embedding = get_embedding(query)
    df["similarities"] = df.embedding.apply(
        lambda x: cosine_similarity(x, query_embedding)
    )
    top_results = df.nlargest(top_n, "similarities")

    context = "\n\n".join(
        [
            f"### {record['title']}\n{record['abstract']}"
            for record in top_results.to_dict(orient="records")
        ]
    )
    return context


if __name__ == "__main__":
    question = "How different countries view Ukrainian NATO alignment?"
    keywords = get_keywords(question)
    works = search_works(keywords)
    if not works:
        print("No relevant works found.")

    abs_df = create_abstracts_dataframe(works)
    documents = search_docs(abs_df, question)

    if not documents:
        print("No relevant documents found.")

    prompt = rag_template.format(context=documents, question=question)
    answered = answer(prompt=prompt)
    print(answered)
