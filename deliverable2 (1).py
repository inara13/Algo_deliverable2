import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import random


class URLValidator:
    def rate_url_validity(self, user_query: str, url: str) -> dict:
        """Simulates rating the validity of a URL."""
        content_relevance = random.randint(0, 100)
        bias_score = random.randint(0, 100)
        final_validity_score = (content_relevance + bias_score) // 2

        return {
            "raw_score": {
                "Content Relevance": content_relevance,
                "Bias Score": bias_score,
                "Final Validity Score": final_validity_score
            }
        }

    def __init__(self):
        self.similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.fake_news_classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
        self.sentiment_analyzer = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

    def fetch_page_content(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return " ".join([p.text for p in soup.find_all("p")])
        except requests.RequestException:
            return ""

    def compute_similarity_score(self, user_query: str, content: str) -> int:
        if not content:
            return 0
        return int(util.pytorch_cos_sim(self.similarity_model.encode(user_query), self.similarity_model.encode(content)).item() * 100)

    def detect_bias(self, content: str) -> int:
        if not content:
            return 50
        sentiment_result = self.sentiment_analyzer(content[:512])[0]
        return 100 if sentiment_result["label"] == "POSITIVE" else 50 if sentiment_result["label"] == "NEUTRAL" else 30

    def validate_url(self, user_query, url_to_check):
        try:
            result = self.rate_url_validity(user_query, url_to_check)
            print("Validation Result:", result)  

            if "Validation Error" in result:
                return {"Error": result["Validation Error"]}

            return {
                "Content Relevance Score": f"{result['raw_score']['Content Relevance']} / 100",
                "Bias Score": f"{result['raw_score']['Bias Score']} / 100",
                "Final Validity Score": f"{result['raw_score']['Final Validity Score']} / 100"
            }
        except Exception as e:
            return {"Error": str(e)}

queries_urls = [
    ("How blockchain works", "https://www.ibm.com/topics/what-is-blockchain"),
    ("Climate change effects", "https://www.nationalgeographic.com/environment/article/climate-change-overview"),
    ("COVID-19 vaccine effectiveness", "https://www.cdc.gov/coronavirus/2019-ncov/vaccines/effectiveness.html"),
    ("Latest AI advancements", "https://www.technologyreview.com/topic/artificial-intelligence"),
    ("Stock market trends", "https://www.bloomberg.com/markets"),
    ("Healthy diet tips", "https://www.healthline.com/nutrition/healthy-eating-tips"),
    ("Space exploration missions", "https://www.nasa.gov/missions"),
    ("Electric vehicle benefits", "https://www.tesla.com/benefits"),
    ("History of the internet", "https://www.history.com/topics/inventions/history-of-the-internet"),
    ("Nutritional benefits of a vegan diet", "https://www.hsph.harvard.edu/nutritionsource/healthy-weight/diet-reviews/vegan-diet/"),
    ("Mental health awareness", "https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response")
]

validator = URLValidator()

results = [validator.rate_url_validity(query, url) for query, url in queries_urls]

for result in results:
    print(result)

formatted_output = []

for query, url in queries_urls:
    output_entry = {
        "Query": query,
        "URL": url,
        "Function Rating": random.randint(1, 5), 
        "Custom Rating": random.randint(1, 5)  
    }
    formatted_output.append(output_entry)

formatted_output