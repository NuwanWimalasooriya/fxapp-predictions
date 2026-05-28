import argparse
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env if present
load_dotenv()
from empirical import empirical_prob
from rag_predictor import build_index, predict_next_probabilities


def pattern_from_str(s: str):
    parts = s.strip().split()
    if len(parts) != 4:
        raise ValueError('Pattern must have 4 values: e.g. "R W R W"')
    return [p.upper() for p in parts]


def append_pattern_to_csv(csv_path: str, pattern: list) -> int:
    df = pd.read_csv(csv_path)
    if 'id' in df.columns and len(df) > 0:
        new_id = int(df['id'].max()) + 1
    else:
        new_id = 1
    row = {'id': new_id, 'disc1': pattern[0], 'disc2': pattern[1], 'disc3': pattern[2], 'disc4': pattern[3]}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return new_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/1min-discs.csv')
    parser.add_argument('--pattern', required=True, help='Query pattern like: "R W R W"')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--add', action='store_true', help='Append the pattern to the CSV before predicting')
    parser.add_argument('--no-rag', action='store_true', help='Skip RAG/LLM calls and show empirical-only output')
    parser.add_argument('--save-output', help='Write RAG output (raw + parsed JSON) to this file')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    query = pattern_from_str(args.pattern)

    emp = empirical_prob(query, df, k=args.k)
    print('Empirical probabilities (top-k):', emp)

    # optionally append to CSV
    if args.add:
        new_id = append_pattern_to_csv(args.data, query)
        print(f'Appended pattern as id {new_id} to {args.data}')
        df = pd.read_csv(args.data)

    if args.no_rag:
        print('\n--no-rag set: skipping RAG/LLM calls.')
        return

    print('\nBuilding RAG index (this will call the embeddings API)...')
    index = build_index(df)
    meta, rag_text, parsed = predict_next_probabilities(query, index, k=args.k)
    print('\nRAG retrieved meta:', meta)
    print('\nLLM RAG raw answer:\n', rag_text)

    if args.save_output:
        out = {
            'meta': meta,
            'raw': rag_text,
            'parsed': parsed,
        }
        with open(args.save_output, 'w', encoding='utf-8') as f:
            import json as _json
            _json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved RAG output to {args.save_output}")


if __name__ == '__main__':
    main()
