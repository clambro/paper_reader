import argparse
import ast
import logging
import os
import re

import pandas as pd
import requests
from lxml import html
from tqdm import tqdm

import config
import utils

logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """
You are a Python bot designed to categorize machine learning papers as practical for use in industry or not, given basic
information about them. A practical paper is one that an experienced machine learning engineer might reasonably
implement in production code at a major tech company. Impractical papers are ones that would never be used in industry.
Simple techniques are likely practical. Dense mathematical proofs are likely impractical. You will be asked to reason
out why the paper given to you is practical or not based on some guiding questions, then make your final decision. 
"""


def main(data_folder):
    filtered_paper_path = os.path.join(data_folder, config.FILTERED_CATEGORIES_FILENAME)
    logging.info(f'Loading raw data from "{filtered_paper_path}"')
    df = pd.read_csv(filtered_paper_path)

    logging.info(f'Pulling abstracts and estimating practicality.')
    output = []
    for _, row in tqdm(list(df.iterrows())):
        # TODO: This assumes the paper is on OpenReview, which is not generally the case.
        paper_html = html.fromstring(requests.get(row['url']).content)
        paper_html.xpath('//div/preceding-sibling::strong[text()="Abstract"]')

        abstract = paper_html.xpath('//main/div/div/div[4]')[0].text_content()
        abstract = re.search('Abstract: (.+)Submission Number:', abstract)[1]

        user_prompt = f"""
Consider the following machine learning paper title and abstract:
-----
Title: {row['title']}
Abstract: {abstract}
-----
Answer the following questions:
1. Does this paper describe a practical technique? Answer no if the paper is highly theoretical. Explain your reasoning.
2. Does this paper have a real-world use case in the tech industry? Answer no if the use case is theoretical. Explain
your reasoning.
3. Could an experienced machine learning engineer understand this paper? Explain your reasoning.
4. Could an experienced machine learning engineer implement the content of this paper in production code? Explain your
reasoning.

Practical papers answer "yes" to the above questions.
Impractical papers answer "no" to the above questions.

5. Given your answers to questions 1-4, classify this paper as (1) practical or (0) impractical.

Your response must be in list format, and the final element must be a binary integer. For example:
[
 "response to question 1 with reasoning",
 "response to question 2 with reasoning",
 "response to question 3 with reasoning",
 "response to question 4 with reasoning",
 0
]
"""
        response = utils.prompt_chat_gpt(SYSTEM_PROMPT, user_prompt, 400)

        try:
            response = ast.literal_eval(response)
            assert len(response) == 5
            output.append([abstract] + response)
        except (SyntaxError, AssertionError):
            logging.warning(f'Syntax error. Response was: {response}')
            output.append([abstract, 'ERROR', 'ERROR', 'ERROR', 'ERROR', -9999])

    logging.info('Constructing output dataframe.')
    output_df = pd.DataFrame(
        output,
        columns=[
            'abstract', 'abs_practical', 'abs_use_case', 'abs_understand', 'abs_implement', 'abs_binary'
        ]
    )
    abstract_df = df.copy()
    abstract_df.reset_index(drop=True, inplace=True)
    output_df.reset_index(drop=True, inplace=True)
    abstract_df = pd.concat([abstract_df, output_df], axis=1)

    logging.info(f'Found {sum(abstract_df["abs_binary"]):g} practical abstracts out of {len(df)} papers.')

    output_path = os.path.join(data_folder, config.ABSTRACTS_FILENAME)
    logging.info('Saving paper dataframe with abstracts and practicality.')
    abstract_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_folder',
        type=str,
        help='The full path to the folder containing the raw data.'
    )
    args = parser.parse_args()
    main(args.data_folder)
