import pandas as pd
from typing import Dict, List

def row_to_pattern(row: pd.Series) -> List[str]:
    return [row['disc1'].strip(), row['disc2'].strip(), row['disc3'].strip(), row['disc4'].strip()]

def empirical_prob(query_pattern: List[str], df: pd.DataFrame, k: int = None) -> Dict[str, float]:
    """Compute empirical probability of Red at each of the four positions.

    - query_pattern: list of 4 values like ['R','W','R','W'] used to filter similar rows.
    - df: dataframe with columns ['disc1'..'disc4']
    - k: if provided, use top-k most similar (by Hamming distance)
    """
    # compute hamming distance for each row
    def hamming(pat1, pat2):
        return sum(1 for a, b in zip(pat1, pat2) if a != b)

    patterns = df.apply(row_to_pattern, axis=1).tolist()
    distances = [hamming(query_pattern, p) for p in patterns]
    idxs = list(range(len(patterns)))
    idxs.sort(key=lambda i: distances[i])
    if k:
        idxs = idxs[:k]

    counts = [0, 0, 0, 0]
    for i in idxs:
        pat = patterns[i]
        for pos, val in enumerate(pat):
            if val.upper() == 'R':
                counts[pos] += 1

    total = len(idxs)
    if total == 0:
        return {f'pos{p+1}': 0.0 for p in range(4)}

    probs = {f'pos{p+1}': counts[p] / total for p in range(4)}
    return probs

if __name__ == '__main__':
    df = pd.read_csv('data/1min-discs.csv')
    q = ['R','W','R','W']
    print('Empirical prob for', q, '->', empirical_prob(q, df, k=5))
