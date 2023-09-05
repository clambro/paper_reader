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
    logging.info(f'Loading raw data from "{raw_data_path}"')
    df = pd.read_csv(raw_data_path)

    num_splits = len(df) // config.TITLES_PER_PROMPT + 1
    logging.info(
        f'Found {len(df)} papers. Splitting them into {num_splits} groups of at most {config.TITLES_PER_PROMPT}.'
    )

    title_groups = np.array_split(df['title'].values, num_splits)
    highlight_groups = np.array_split(df['highlight'].values, num_splits)

    logging.info(f'Getting categories from each group.')
    categories = []
    for titles, highlights in tqdm(list(zip(title_groups, highlight_groups))):
        paper_info = '\n-----\n'.join('Title: ' + titles + '\nHighlight: ' + highlights)

        user_prompt = f"""
Consider the following list of {len(titles)} machine learning paper titles and highlights:
-----
{paper_info}
-----
For each machine learning paper in the above list, assign a general category of research to which it belongs. Return
your results as a Python list of categories in the same order as the papers above with no additional commentary.
"""
        acc = 0
        while True:
            if acc > 5:
                raise ConnectionRefusedError('Model is overloaded.')
            req = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 256,
                "temperature": 0,
            }
            response = requests.post(config.OPENAI_URL, headers=config.OPENAI_HEADERS, json=req)
            try:
                response = response.json()["choices"][0]["message"]["content"]
                break
            except KeyError:
                if response.status_code == 429:
                    logging.info('OpenAI API is overloaded. Pausing before trying again.')
                    time.sleep(3)
                    acc += 1
                else:
                    logging.exception(f'Got response code {response.status_code} with response {response.json()}')
                    raise ConnectionRefusedError(f'Response code {response.status_code}')

        try:
            response = ast.literal_eval(response)
            categories += response
        except SyntaxError:
            categories += len(titles) * ['ERROR']

    logging.info('Deduplicating categories.')
    # Simple deduplication.
    clean_categories = []
    for cat in categories:
        cat = cat.replace('-', '').replace(' ', '')
        if cat.endswith('s') and not cat.endswith('ss'):
            cat = cat[:-1]
        clean_categories.append(cat)

    df['category'] = clean_categories
    output_path_all = os.path.join(data_folder, 'categorized_papers.csv')
    logging.info('Saving paper dataframe with categories.')
    df.to_csv(output_path_all, index=False)

    logging.info('Calculating distinct categories.')
    distinct_categories, category_counts = np.unique(df['category'], return_counts=True)
    distinct_df = pd.DataFrame({'category': distinct_categories, 'count': category_counts})

    output_path_distinct = os.path.join(data_folder, 'distinct_categories.csv')
    logging.info('Saving distinct category dataframe.')
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
