import argparse
import ast
import io
import logging
import os

import fitz
import pandas as pd
import requests
from tqdm import tqdm

import config
import utils

logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """
You are a Python bot designed to categorize machine learning papers as practical for use in industry or not, given basic
information about them. A practical paper is one that an experienced machine learning engineer might reasonably
implement in production code at a major tech company. Impractical papers are ones that would never be used in industry.
Simple techniques are likely practical. Dense mathematical proofs are likely impractical. You will be asked to reason
out whether the paper given to you is practical or not based on some guiding questions, then make your final decision. 
"""


def main(data_folder):
    filtered_abstract_path = os.path.join(data_folder, config.FILTERED_CATEGORIES_FILENAME)
    logging.info(f'Loading raw data from "{filtered_abstract_path}"')
    df = pd.read_csv(filtered_abstract_path)

    logging.info(f'Pulling content and estimating practicality.')
    output = []
    for _, row in tqdm(list(df.iterrows())):
        try:
            response = requests.get(row['pdf_url'], timeout=60).content
            pdf = io.BytesIO(response)
            text = []
            with fitz.open(stream=pdf) as doc:
                for page in doc:
                    page_text = page.get_text()
                    text.append(page_text)
                    if 'References' in page_text:
                        break  # Skip everything after the references.
            text = '\n'.join(text[1:])  # Skip the first page. We already have the abstract.
        except Exception as e:
            output.append(['ERROR', 'ERROR', 'ERROR', 'ERROR', -9999])
            print(e)
            continue

        snippets = []
        n = len(text) // 10
        for i in range(10):
            sub_text = text[i * n:i * n + 500]
            snippets.append(sub_text)
        snippets = '\n----------\n'.join(snippets)

        user_prompt = f"""
Consider the following machine learning paper title, abstract, and (non-exhaustive) text snippets:
----------
Title: {row['title']}
Abstract: {row['abstract']}

BEGINNING OF PAPER SNIPPETS

----------
{snippets}
----------

END OF PAPER SNIPPETS

Answer the following questions:
1. Does this paper contain code, algorithms, or repositories?
2. Does this paper contain useful figures and tables?
3. Does this paper describe a practical technique? Answer no if the paper is mostly dense mathematical proofs.
4. Could/should an experienced machine learning engineer implement the content of this paper in production code?

Practical papers answer "yes" to the above questions.
Impractical papers answer "no" to the above questions.

5. Given your answers to questions 1-4, classify this paper as (1) practical or (0) impractical.

Your response must be in list format, and the final element must be a binary integer. For example:
[
 "brief response to question 1 with reasoning",
 "brief response to question 2 with reasoning",
 "brief response to question 3 with reasoning",
 "brief response to question 4 with reasoning",
 0
]
"""
        response = None
        try:
            response = utils.prompt_chat_gpt(SYSTEM_PROMPT, user_prompt, 500)
            # Make sure that quotes inside the strings don't crash the evaluation.
            response = response.replace('\n "', '\n """').replace('",\n', '""",\n')
            response = ast.literal_eval(response)
            assert len(response) == 5
            output.append(response)
        except Exception as e:
            logging.warning(f'Syntax error. Response was: {response}. Error was {e}')
            output.append(['ERROR', 'ERROR', 'ERROR', 'ERROR', -9999])

    logging.info('Constructing output dataframe.')
    output_df = pd.DataFrame(
        output,
        columns=[
            'content_code', 'content_figures', 'content_technique', 'content_implement', 'content_binary'
        ]
    )
    practical_df = df.copy()
    practical_df.reset_index(drop=True, inplace=True)
    output_df.reset_index(drop=True, inplace=True)
    practical_df = pd.concat([practical_df, output_df], axis=1)

    logging.info(f'Found {len(practical_df.query("content_binary == 1"))} practical papers in {len(df)} papers.')
    logging.info(f'Found {len(practical_df.query("content_binary == -9999"))} errors.')

    all_content_path = os.path.join(data_folder, config.ALL_CONTENT_FILENAME)
    logging.info('Saving paper dataframe with content practicality.')
    practical_df.to_csv(all_content_path, index=False)

    filtered_content_path = os.path.join(data_folder, config.PRACTICAL_CONTENT_FILENAME)
    logging.info('Saving final practical paper list.')
    practical_df.query('content_binary != 0').to_csv(filtered_content_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_folder',
        type=str,
        help='The full path to the folder containing the raw data.'
    )
    args = parser.parse_args()
    main(args.data_folder)
