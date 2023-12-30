import argparse
import logging
import os

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO)


def main(data_folder):
    paper_category_path = os.path.join(data_folder, config.PRACTICAL_ABSTRACT_FILENAME)
    logging.info(f'Loading paper category data from "{paper_category_path}"')
    paper_df = pd.read_csv(paper_category_path)
    logging.info(f'Found {len(paper_df)} papers with {paper_df["category"].nunique()} distinct categories.')

    distinct_category_path = os.path.join(data_folder, config.DISTINCT_CATEGORIES_FILENAME)
    logging.info(f'Loading relevant categories from "{distinct_category_path}"')
    cat_df = pd.read_csv(distinct_category_path).dropna(axis=0)
    logging.info(f'Found {len(cat_df)} distinct categories to keep, representing {sum(cat_df["count"]):g} papers.')

    logging.info(f'Filtering papers.')
    paper_df = paper_df[np.isin(paper_df['category'], cat_df['category'])]

    output_path = os.path.join(data_folder, config.FILTERED_CATEGORIES_FILENAME)
    logging.info('Saving filtered paper dataframe.')
    paper_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_folder',
        type=str,
        help='The full path to the folder containing the category data.'
    )
    args = parser.parse_args()
    main(args.data_folder)
