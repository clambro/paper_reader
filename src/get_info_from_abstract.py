import argparse
import ast
import logging
import os
import re

import numpy as np
import pandas as pd
import requests
from lxml import html
from tqdm import tqdm

import config
import utils

logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """
You are a Python bot designed to categorize machine learning papers and filter out impractical ones, given basic
information about them. 

On categorization: Good categories are 1-2 words long and never more than 3 words long. Here are some examples of good,
general paper categories:
- Adversarial Learning
- Attention
- Bandit Methods
- Bayesian Methods
- Benchmarks
- Biology
- Causal Analysis
- Clustering
- Computer Vision
- Chemistry
- Deep Learning
- Diffusion Models
- Ensembling
- Explainable AI
- Generative Models
- Graph Neural Networks
- Knowledge Retrieval
- Mathematical Analysis
- Metrics
- Model Robustness
- Natural Language Processing
- Online Learning
- Physics
- Privacy
- Prompt Engineering
- Reinforcement Learning
- Statistics
- Tensors
- Time Series

On practicality: A practical paper is one that an experienced machine learning engineer might reasonably implement in
production code at a major tech company. Simple techniques are likely practical. Dense mathematical proofs are
impractical.
"""
USER_PROMPT = """
Consider the following machine learning paper title and abstract:
-----
{title}

{abstract}
-----
Answer the following questions:
1. Summarize the paper in one sentence.
2. Categorize the paper. Use one of the categories listed above if possible.
3. Does this paper describe a practical technique? Answer no if the paper is highly theoretical.
4. Does this paper have a real-world use case in the tech industry? Answer no if the use case is theoretical.
5. Should the content of this paper be implemented/used in a production codebase?

Practical papers answer "yes" to questions 3-5.
Impractical papers answer "no" to questions 3-5.

6. Binarize you answer to question 6 as (1) practical or (0) impractical.

Your response must be in list format, and the final element must be a binary integer. For example:
[
 "one sentence summary",
 "category",
 "brief response to question 3 with reasoning",
 "brief response to question 4 with reasoning",
 "brief response to question 5 with reasoning",
 0
]
"""


def main(data_folder):
    raw_data_path = os.path.join(data_folder, config.RAW_DATA_FILENAME)
    logging.info(f'Loading raw data from "{raw_data_path}"')
    df = pd.read_csv(raw_data_path)

    logging.info(f'Pulling abstracts, categorizing, and estimating practicality.')
    output = []
    for _, row in tqdm(list(df.iterrows())):
        try:
            abstract = get_abstract_from_html(row['abstract_url'])
        except IndexError:
            abstract = 'Failed to load abstract. Use the paper title to answer the questions.'

        user_prompt = USER_PROMPT.format(title=row['title'], abstract=abstract)
        response = utils.prompt_chat_gpt(SYSTEM_PROMPT, user_prompt, 512)
        # Make sure that quotes inside the strings don't crash the evaluation.
        response = response.replace('\n "', '\n """').replace('",\n', '""",\n')

        try:
            response = ast.literal_eval(response)
            assert len(response) == 6
            output.append([abstract] + response)
            # print(row['title'])
            # [print(a) for a in response]
        except (SyntaxError, AssertionError):
            logging.warning(f'Syntax error. Response was: {response}')
            output.append([abstract, 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR', -9999])

    logging.info('Constructing output dataframe.')
    output_df = pd.DataFrame(
        output,
        columns=[
            'abstract', 'summary', 'category', 'abs_practical', 'abs_use_case', 'abs_implement', 'abs_binary'
        ]
    )
    abstract_df = df.copy()
    abstract_df.reset_index(drop=True, inplace=True)
    output_df.reset_index(drop=True, inplace=True)
    abstract_df = pd.concat([abstract_df, output_df], axis=1)

    logging.info(f'Found {len(abstract_df.query("abs_binary == 1"))} practical abstracts in {len(df)} papers.')
    logging.info(f'Found {len(abstract_df.query("abs_binary == -9999"))} errors.')

    logging.info('Deduplicating categories.')
    # Simple deduplication.
    clean_categories = []
    for cat in abstract_df["category"].values:
        cat = cat.replace('-', '').replace(' ', '').lower()
        if cat.endswith('s') and not cat.endswith('ss'):
            cat = cat[:-1]
        clean_categories.append(cat)
    abstract_df["category"] = clean_categories

    output_path_all = os.path.join(data_folder, config.ALL_ABSTRACT_FILENAME)
    logging.info('Saving entire dataframe.')
    abstract_df.to_csv(output_path_all, index=False)

    output_path_practical = os.path.join(data_folder, config.PRACTICAL_ABSTRACT_FILENAME)
    logging.info('Saving practical dataframe.')
    abstract_df.query('abs_binary != 0').to_csv(output_path_practical, index=False)

    logging.info('Calculating distinct categories.')
    distinct_categories, category_counts = np.unique(abstract_df['category'], return_counts=True)
    distinct_df = pd.DataFrame({'category': distinct_categories, 'count': category_counts})
    logging.info(f'Found {len(distinct_df)} distinct categories.')

    output_path_distinct = os.path.join(data_folder, config.DISTINCT_CATEGORIES_FILENAME)
    logging.info('Saving distinct category dataframe.')
    distinct_df.to_csv(output_path_distinct, index=False)


def get_abstract_from_html(abstract_url):
    paper_html = html.fromstring(requests.get(abstract_url).content)
    if '.openreview.net' in abstract_url:
        abstract = paper_html.xpath('//main/div/div/div[4]')[0].text_content()
        abstract = re.search('Abstract: (.+)Submission Number:', abstract)[1]
    elif '.aaai.org' in abstract_url:
        abstract = paper_html.xpath('//section[@class="item abstract"]')[0].text_content()
        abstract = re.search('[\n\t]+Abstract[\n\t]+(.+)[\n\t]+', abstract)[1]
    else:
        raise NotImplementedError(f'URL extraction not implemented for {abstract_url}')
    return abstract


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_folder',
        type=str,
        help='The full path to the folder containing the raw data.'
    )
    args = parser.parse_args()
    main(args.data_folder)
