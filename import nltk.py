from dotenv import load_dotenv
import os

load_dotenv()  # Add this line at the top

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')