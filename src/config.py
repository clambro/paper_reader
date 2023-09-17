import os

RAW_DATA_FILENAME = '00_raw_paperdigest_papers.csv'
ALL_ABSTRACT_FILENAME = '01_all_abstract_papers.csv'
PRACTICAL_ABSTRACT_FILENAME = '02_practical_abstract_papers.csv'
DISTINCT_CATEGORIES_FILENAME = '03_distinct_categories.csv'
FILTERED_CATEGORIES_FILENAME = '04_filtered_category_papers.csv'
ALL_CONTENT_FILENAME = '05_all_content_papers.csv'
PRACTICAL_CONTENT_FILENAME = '06_practical_content_papers.csv'

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_HEADERS = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
