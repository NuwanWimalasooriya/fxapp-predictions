import os
import json
import re
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import faiss
from openai import OpenAI


class SimpleDoc:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class SimpleFAISS:
    def __init__(self, texts: List[str], metadatas: List[dict], vectors: np.ndarray):
        self.texts = texts
        self.metadatas = metadatas
        self.vectors = vectors.astype('float32')
        dim = self.vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.vectors)

    def similarity_search(self, query: str, k: int = 5):
        qvec = embed_texts([query])
        D, I = self.index.search(qvec.astype('float32'), k)
        results = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append(SimpleDoc(self.texts[idx], self.metadatas[idx]))
        return results


def embed_texts(texts: List[str], model: str = None) -> np.ndarray:
    # Determine whether to use local embeddings.
    env_val = os.environ.get('USE_LOCAL_EMBEDDINGS')
    local_model_name = os.environ.get('LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

    try:
        from sentence_transformers import SentenceTransformer
        has_local = True
    except Exception:
        has_local = False

    if env_val is not None:
        use_local = str(env_val).lower() in ('1', 'true', 'yes')
    else:
        # If no explicit env var, prefer local when available and no OPENAI_API_KEY present
        use_local = has_local and not bool(os.environ.get('OPENAI_API_KEY'))

    if use_local and has_local:
        model_local = SentenceTransformer(local_model_name)
        vectors = model_local.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return np.array(vectors).astype('float32')

    # Otherwise, try OpenAI embeddings and fall back to local if API fails
    try:
        client = OpenAI()
        model = model or os.environ.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        resp = client.embeddings.create(model=model, input=texts)
        vectors = [d.embedding for d in resp.data]
        return np.array(vectors).astype('float32')
    except Exception:
        # attempt local fallback if available
        try:
            from sentence_transformers import SentenceTransformer
            model_local = SentenceTransformer(local_model_name)
            vectors = model_local.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return np.array(vectors).astype('float32')
        except Exception:
            # re-raise original error if local fallback not available
            raise


def build_index(df: pd.DataFrame) -> SimpleFAISS:
    texts = df.apply(lambda r: f"id:{r['id']} pattern:{r['disc1']} {r['disc2']} {r['disc3']} {r['disc4']}", axis=1).tolist()
    metadatas = df.to_dict(orient='records')
    vectors = embed_texts(texts)
    return SimpleFAISS(texts, metadatas, vectors)


PROMPT_NEXT = """
You are given a set of example trials, each showing four discs as Red (R) or White (W):
{examples}

Current (most recent) observation: {current}

Using the retrieved examples, estimate the probability that the NEXT trial (after the current observation) will have each position be Red or White.
Return ONLY a JSON object with this exact shape:
{{
    "pos1": {{"R": 0.5, "W": 0.5}},
    "pos2": {{"R": 0.5, "W": 0.5}},
    "pos3": {{"R": 0.5, "W": 0.5}},
    "pos4": {{"R": 0.5, "W": 0.5}}
}}

Numbers must be between 0 and 1 and sum to 1 per position. Do not include any other text.
"""



def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def predict_next_probabilities(current_pattern: List[str], index: SimpleFAISS, k: int = 10, temperature: float = 0.0) -> Tuple[dict, str, Any]:
    qtext = ' '.join(current_pattern)
    docs = index.similarity_search(qtext, k=k)

    # Decide whether to use a local heuristic LLM or call OpenAI
    env_local_llm = os.environ.get('USE_LOCAL_LLM')
    prefer_local_llm = False
    if env_local_llm is not None:
        prefer_local_llm = str(env_local_llm).lower() in ('1', 'true', 'yes')
    else:
        # if no explicit setting, prefer local when OPENAI_API_KEY is not set
        prefer_local_llm = not bool(os.environ.get('OPENAI_API_KEY'))

    def _local_heuristic(docs, index):
        # Use the 'id' in metadata to find the next trial (id+1) in the index metadatas
        id_map = {int(m['id']): m for m in index.metadatas if 'id' in m}
        # Ordered history (by id)
        ordered = sorted(id_map.items(), key=lambda x: x[0])
        history = [ [m.get('disc1'), m.get('disc2'), m.get('disc3'), m.get('disc4')] for _, m in ordered ]
        counts = {1: {'R': 0.0, 'W': 0.0}, 2: {'R': 0.0, 'W': 0.0}, 3: {'R': 0.0, 'W': 0.0}, 4: {'R': 0.0, 'W': 0.0}}
        matched = 0

        # Base counts from retrieved docs (weight 1)
        for d in docs:
            mid = d.metadata.get('id')
            try:
                mid_int = int(mid)
            except Exception:
                continue
            nxt = id_map.get(mid_int + 1)
            if not nxt:
                continue
            matched += 1
            next_pattern = [nxt.get('disc1'), nxt.get('disc2'), nxt.get('disc3'), nxt.get('disc4')]
            for i, v in enumerate(next_pattern, start=1):
                if v in ('R', 'W'):
                    counts[i][v] += 1.0

        # Extra weight for historical repeating occurrences of the current pattern
        repeat_weight = 2.0
        # find indices in history matching current_pattern
        repeat_indices = [i for i, p in enumerate(history) if p == current_pattern]
        repeat_matches = 0
        for idx in repeat_indices:
            if idx + 1 < len(history):
                nextp = history[idx + 1]
                repeat_matches += 1
                for i, v in enumerate(nextp, start=1):
                    if v in ('R', 'W'):
                        counts[i][v] += repeat_weight

        total_matches = matched + repeat_matches

        # Normalize into probabilities
        probs = {}
        for i in range(1, 5):
            r = counts[i]['R']
            w = counts[i]['W']
            s = r + w
            if s == 0:
                probs[f'pos{i}'] = {'R': 0.5, 'W': 0.5}
            else:
                probs[f'pos{i}'] = {'R': r / s, 'W': w / s}

        meta = {'retrieved_count': len(docs), 'parsed': True, 'local_matches': total_matches}
        return meta, json.dumps(probs), probs

    if prefer_local_llm:
        return _local_heuristic(docs, index)

    # Otherwise attempt OpenAI chat completion, but fall back to local heuristic on failure
    client = OpenAI()
    examples = '\n'.join([d.page_content for d in docs])
    prompt = PROMPT_NEXT.format(examples=examples, current=qtext)
    model = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
    try:
        messages = [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        answer = resp.choices[0].message.content
        parsed = _extract_json(answer)
        meta = {'retrieved_count': len(docs), 'parsed': parsed is not None}
        return meta, answer, parsed
    except Exception:
        return _local_heuristic(docs, index)
