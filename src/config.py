import os

RAW_DATA_FILENAME = 'raw_paperdigest_data.csv'
CATEGORIZED_DATA_FILENAME = 'categorized_papers.csv'
DISTINCT_CATEGORIES_FILENAME = 'distinct_categories.csv'
FILTERED_CATEGORIES_FILENAME = 'categorized_and_filtered_papers.csv'
ABSTRACTS_FILENAME = 'abstract_practicality.csv'
TITLES_PER_PROMPT = 10  # The model becomes less reliable if you go above 10 papers, even with a 16k context window.

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_HEADERS = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
