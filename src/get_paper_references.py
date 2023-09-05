import argparse
import logging
import os
import re

import pandas as pd
import requests
from lxml import html
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)


def main(paperdigest_url, output_folder):
    """Download the title, link, and highlight for each paper in the given paperdigest link in CSV format.

    TODO: This assumes the paperdigest links always go to openreview. This is not the case in general.

    Parameters
    ----------
    paperdigest_url : str
        The link to the paperdigest page that lists all the papers for a given conference.
    output_folder : str
        The full path to the folder in which the output will be saved.
    """
    logging.info(f'Calling {paperdigest_url}')
    response = requests.get(paperdigest_url)
    logging.info(f'Response code = {response.status_code}')
    if response.status_code != 200:
        raise ConnectionError(f'Response code from URL was {response.status_code}')

    logging.info(f'Pulling the HTML string from the response.')
    all_paper_html = html.fromstring(response.content)
    rows = all_paper_html.xpath('//div[@class="blog-post__content"]/table/tr/td[2]')

    logging.info(f'Detected {len(rows)} papers. Pulling details.')
    output = []
    for r in tqdm(rows):
        title = r.xpath('a/b/text()')[0]
        paperdigest_url = r.xpath('a/@href')[0]
        url_slug = re.search('forum-id-(.+?)-', paperdigest_url)[1]
        url = 'https://openreview.net/forum?id=' + url_slug
        blurb = r.xpath('small/i/text()')[0][2:]  # Text starts with ": "
        output.append((title, url, blurb))

    df_orig = pd.DataFrame(output, columns=['title', 'url', 'highlight'])
    df_orig = df_orig.sample(frac=1, replace=False, random_state=2112).reset_index(drop=True)  # Shuffle.

    output_path = os.path.join(output_folder, 'raw_paperdigest_data.csv')
    df_orig.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--paperdigest_url',
        type=str,
        help='The link to the paperdigest page that lists all the papers for a given conference.'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        help='The full path to the folder in which the output will be saved.'
    )
    args = parser.parse_args()
    main(args.paperdigest_url, args.output_folder)
