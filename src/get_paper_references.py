import argparse
import logging
import os
import re
import time

import pandas as pd
import requests
from lxml import html
from tqdm import tqdm

import config
from enum import Enum

logging.basicConfig(level=logging.INFO)


class Websites(str, Enum):
    OPENREVIEW = 'openreview'
    AAAI = 'aaai'


def main(paperdigest_url, host_website, output_folder):
    """Download the title, link, and highlight for each paper in the given paperdigest link in CSV format.

    Parameters
    ----------
    paperdigest_url : str
        The link to the paperdigest page that lists all the papers for a given conference.
    host_website : str
        The website name on which the actual papers are hosted.
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
    paper_rows = all_paper_html.xpath('//div[@class="blog-post__content"]/table/tr/td[2]')

    logging.info(f'Detected {len(paper_rows)} papers. Pulling details.')
    output = []
    for row in tqdm(paper_rows):
        title = row.xpath('a/b/text()')[0]
        paperdigest_url = row.xpath('a/@href')[0]
        abstract_url, pdf_url = get_urls_from_website(paperdigest_url, host_website)
        output.append((title, abstract_url, pdf_url))

    df = pd.DataFrame(output, columns=['title', 'abstract_url', 'pdf_url'])

    output_path = os.path.join(output_folder, config.RAW_DATA_FILENAME)
    df.to_csv(output_path, index=False)


def get_urls_from_website(paperdigest_url, host_website):
    if host_website == Websites.OPENREVIEW:
        url_slug = re.search('forum-id-(.+?)-', paperdigest_url)[1]
        abstract_url = 'https://openreview.net/forum?id=' + url_slug
        pdf_url = 'https://openreview.net/pdf?id=' + url_slug
    elif host_website == Websites.AAAI:
        url_slug = re.search('paper_id=aaai-(.+?)-', paperdigest_url)[1]
        abstract_url = 'https://ojs.aaai.org/index.php/AAAI/article/view/' + url_slug
        response = requests.get(abstract_url)
        pdf_url = html.fromstring(response.content).xpath('//a[@class="obj_galley_link pdf"]/@href')[0]
        time.sleep(0.1)
    else:
        raise NotImplementedError(f'URL extraction not implemented for {host_website}')
    return abstract_url, pdf_url


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--paperdigest_url',
        type=str,
        help='The link to the paperdigest page that lists all the papers for a given conference.'
    )
    parser.add_argument(
        '--host_website',
        type=str,
        help='The website name on which the actual papers are hosted..'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        help='The full path to the folder in which the output will be saved.'
    )
    args = parser.parse_args()
    main(args.paperdigest_url, args.host_website, args.output_folder)
