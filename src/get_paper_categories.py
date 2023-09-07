import argparse
import ast
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import utils

logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """
You are a Python bot designed to categorize machine learning papers given basic information about them. Good categories
are 1-2 words long and never more than 3 words long. Here are some examples of good, general paper categories:
- Adversarial Learning
- Attention
- Bandit Methods
- Bayesian Methods
- Biology
- Causal Analysis
- Clustering
- Computer Vision
- Chemistry
- Deep Learning
- Ensembling
- Explainable AI
- Generative Models
- Graph Neural Networks
- Knowledge Retrieval
- Mathematical Analysis
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

        # The titles returned by GPT are not used in the code, but they improve model performance.
        user_prompt = f"""
Consider the following list of {len(titles)} machine learning paper titles and highlights:
-----
{paper_info}
-----
For each machine learning paper in the above list, assign a general category of research to which it belongs. Your
response must be a Python list of {len(titles)} tuples (title, category) in the same order as the papers above. Sample
response for 2 papers:
[("title 1", "category1"), ("title 2", "category2")]
"""
        response = utils.prompt_chat_gpt(SYSTEM_PROMPT, user_prompt, 400)

        try:
            response = ast.literal_eval(response)
            assert len(response) == len(titles)
            categories += list(zip(*response))[1]  # Ignore the titles.
        except (SyntaxError, AssertionError):
            logging.warning(f'Syntax error. Response was: {response}')
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
    output_path_all = os.path.join(data_folder, config.CATEGORIZED_DATA_FILENAME)
    logging.info('Saving paper dataframe with categories.')
    df.to_csv(output_path_all, index=False)

    logging.info('Calculating distinct categories.')
    distinct_categories, category_counts = np.unique(df['category'], return_counts=True)
    distinct_df = pd.DataFrame({'category': distinct_categories, 'count': category_counts})

    output_path_distinct = os.path.join(data_folder, config.DISTINCT_CATEGORIES_FILENAME)
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
