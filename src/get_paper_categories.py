import argparse
import ast
import logging
import os
import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """
You are a bot designed to categorize machine learning papers given basic information about them. Good categories are
1-2 words long and never more than 3 words long. Here are some examples of good, general paper categories:
- Adversarial Learning
- Attention
- Multi-Armed Bandits
- Bayesian Methods
- Biology
- Causal Learning
- Clustering
- Computer Vision
- Chemistry
- Deep Learning
- Ensembling
- Explainable AI
- Generative Models
- Graph Neural Networks
- Knowledge Retrieval
- Metrics
- Natural Language Processing
- Online Learning
- Physics
- Privacy
- Reinforcement Learning
- Statistics
- Tensors
- Time Series
"""


def main(data_folder):
    raw_data_path = os.path.join(data_folder, config.RAW_DATA_FILENAME)
    df = pd.read_csv(raw_data_path)

    categories = []
    title_groups = np.array_split(df['title'].values, len(df) // config.TITLES_PER_PROMPT + 1)
    highlight_groups = np.array_split(df['highlight'].values, len(df) // config.TITLES_PER_PROMPT + 1)

    for titles, highlights in tqdm(list(zip(title_groups, highlight_groups))):
        paper_info = '\n-----\n'.join('Title: ' + titles + '\nHighlight: ' + highlights)

        paper_prompt = f"""
Consider the following list of {len(titles)} machine learning paper titles and highlights:
-----
{paper_info}
-----
For each machine learning paper in the above list, assign a general category of research to which it belongs.
Return your results as a Python list of categories in the same order as the list above with no additional commentary.
"""
        acc = 0
        while True:
            if acc > 5:
                raise ConnectionRefusedError('Model is overloaded.')
            req = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": paper_prompt},
                ],
                "max_tokens": 256,
                "temperature": 0,
            }
            response = requests.post(config.OPENAI_URL, headers=config.OPENAI_HEADERS, json=req).json()
            try:
                response = response["choices"][0]["message"]["content"]
                break
            except KeyError:  # API is overloaded.
                print('Overloaded. Trying again.')
                time.sleep(3)
                acc += 1

        try:
            response = ast.literal_eval(response)
            categories += response
        except SyntaxError:
            categories += len(titles) * ['ERROR']

    # Simple deduplication.
    clean_categories = []
    for cat in categories:
        cat = cat.replace('-', '').replace(' ', '')
        if cat.endswith('s') and not cat.endswith('ss'):
            cat = cat[:-1]
        clean_categories.append(cat)

    df['category'] = clean_categories
    output_path_all = os.path.join(data_folder, 'categorized_papers.csv')
    df.to_csv(output_path_all, index=False)

    distinct_categories, category_counts = np.unique(df['category'], return_counts=True)
    distinct_df = pd.DataFrame([distinct_categories, category_counts], columns=['category', 'count'])
    output_path_distinct = os.path.join(data_folder, 'distinct_categories.csv')
    distinct_df.to_csv(output_path_distinct, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_folder',
        type=str,
        help='The full path to the folder containing the raw data.'
    )
    args = parser.parse_args()
    main(args.data_folder)
