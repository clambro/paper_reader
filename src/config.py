import os

RAW_DATA_FILENAME = 'raw_paperdigest_data.csv'
TITLES_PER_PROMPT = 10  # The model becomes less reliable if you go above 10 papers, even with a 16k context window.

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_HEADERS = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
