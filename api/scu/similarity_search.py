import os
import requests
from supabase import create_client, Client
from langchain.schema import Document

class SimilaritySearch:
    def __init__(self):
        self.embedding_model = "text-embedding-3-large"  # Correct model name
        # Load environment variables for Azure OpenAI
        self.api_key = os.environ.get("AZURE_OPENAI_KEY")
        self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
        # Initialize Supabase client
        SUPABASE_URL = os.environ.get("SUPABASE_URL")
        SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def embed_text_api(self, text: str) -> list:
        """
        Embed text using the Azure Embedding endpoint.
        """
        url = f"{self.endpoint}openai/deployments/{self.deployment_name}/embeddings?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        data = {
            "input": text
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response_data = response.json()
            if response.status_code == 200:
                return response_data["data"][0].get("embedding")
            else:
                error_message = response_data.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"API Error: {response.status_code} - {error_message}")
        except Exception as e:
            print(f"Error fetching embedding for text. Error: {e}")
            return None

    def query_similar_docs(self, query_text, match_threshold=0.5, match_count=5):
        # First, embed the query text
        print("Embedding the query text...")
        query_embedding = self.embed_text_api(query_text)
        if query_embedding is None:
            print("Failed to get embedding for the query text.")
            return []

        # Now, call the match_docs function via Supabase RPC
        params = {
            'query_embedding': query_embedding,
            'match_threshold': match_threshold,
            'match_count': match_count
        }

        # Since the Supabase client is synchronous, we can call directly
        print("Querying the database for similar documents...")

        response = self.supabase.rpc("match_docs", params).execute()

        documents = []
        if response.data:
            print(f"Found {len(response.data)} similar documents:\n")
            for doc in response.data:
                document = Document(
                    page_content=doc['chunk'],
                    metadata={
                        "similarity": doc['similarity'],
                        "file_path": doc['file_path']
                    }
                )
                documents.append(document)
        else:
            print("No similar documents found.")

        return documents

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    search = SimilaritySearch()
    query = "How do I enable the pgvector extension?"
    docs = search.query_similar_docs(query)
    print(docs)
