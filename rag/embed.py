import os
import glob
import aiohttp
import tiktoken
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm
from supabase import create_client, Client
import asyncio

def chunk_text(text, chunk_size=8191, overlap_size=200, encoding_name='cl100k_base'):
    """
    Chunk text into chunks of up to chunk_size tokens, with overlap_size tokens overlapping.
    Returns a list of strings.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    end = chunk_size
    while start < len(tokens):
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap_size
        end = start + chunk_size
    return chunks

class Embedder:
    """
    Used to read mdx files, chunk them, embed them using Azure Embedding endpoint, and insert into Supabase
    """

    def __init__(self):
        self.embedding_model = "text-embedding-3-large"  # Update to your model name if different
        # API call rate limiter based on Azure limits
        self.call_rate_limiter = AsyncLimiter(35, 1)  # 35 requests per second
        # Token rate limiter (Azure default limit is 5,833 tokens per second)
        self.token_rate_limiter = AsyncLimiter(5833, 1)
        # Initialize Supabase client
        SUPABASE_URL = os.environ.get("SUPABASE_URL")
        SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Semaphore to limit concurrency
        self.semaphore = asyncio.Semaphore(35)  # Limit to 35 concurrent requests
        # Initialize aiohttp session
        self.session = aiohttp.ClientSession()

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(self.embedding_model)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    async def embed_text_api(self, text: str, retry_count=0) -> list:
        """
        Embed text using the Azure Embedding endpoint with exponential backoff and max retries
        """
        max_retries = 5
        backoff_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60 seconds
        api_key = os.environ.get("AZURE_OPENAI_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")  # e.g., 'https://your-resource-name.openai.azure.com'
        deployment_name = os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")  # e.g., 'text-embedding-ada-002'
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
        url = f"{endpoint}openai/deployments/{deployment_name}/embeddings?api-version={api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        data = {
            "input": text
        }

        # Tokens
        num_tokens = self.num_tokens_from_string(text)

        # Wait until it's safe to proceed under both rate limits
        await asyncio.gather(
            self.call_rate_limiter.acquire(),
            self.token_rate_limiter.acquire(num_tokens)
        )

        try:
            async with self.session.post(url, headers=headers, json=data) as response:
                response_data = await response.json()
                if response.status == 200:
                    return response_data["data"][0].get("embedding")
                elif response.status == 429:
                    # Handle rate limit error: wait and retry
                    retry_after = int(response.headers.get("Retry-After", str(backoff_time)))
                    print(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                    if retry_count < max_retries:
                        return await self.embed_text_api(text, retry_count + 1)
                    else:
                        print(f"Max retries exceeded for text: {text[:50]}...")
                        return None
                else:
                    raise Exception(f"API Error: {response.status} {response_data.get('error', '')}")
        except Exception as e:
            print(f"Error fetching embedding for text. Error: {e}")
            return None

    async def insert_into_supabase(self, file_path, chunk, embedding):
        data = {
            'file_path': file_path,
            'chunk': chunk,
            'embedding': embedding  # This should be a list of floats
        }
        # Since the Supabase client is synchronous, we can use asyncio.to_thread
        await asyncio.to_thread(self.supabase.table('docs').insert(data).execute)

    async def embed_and_insert(self, chunk_data):
        """
        Embeds the chunk and inserts into Supabase
        """
        async with self.semaphore:
            chunk = chunk_data['chunk']
            file_path = chunk_data['file_path']
            embedding = await self.embed_text_api(chunk)
            if embedding is not None:
                # Insert into Supabase
                await self.insert_into_supabase(file_path, chunk, embedding)

    async def process_files_and_embed(self):
        file_paths = glob.glob('../data/supabase_docs/**/*.mdx', recursive=True)
        print(f"Found {len(file_paths)} files to process.")
        processed_file_paths = []
        # Fetch processed file paths from Supabase
        response = self.supabase.table('docs').select('file_path').execute()
        if response.data:
            processed_file_paths = [row['file_path'] for row in response.data]
        # Filter out already processed files
        file_paths = [file_path for file_path in file_paths if file_path not in processed_file_paths]
        print(f"Found {len(file_paths)} new files to process.")
        print(file_paths)
        all_chunks = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = chunk_text(text, chunk_size=8191, overlap_size=200)  # Updated chunk_size to 8191
            for chunk in chunks:
                all_chunks.append({
                    'file_path': file_path,
                    'chunk': chunk
                })
        print(f"Found {len(all_chunks)} chunks to embed.")
        # Delete all rows in the table DELETE requires a WHERE clause
        # data = self.supabase.table('docs').delete().in_('file_path', [chunk_data['file_path'] for chunk_data in all_chunks]).execute()
        # Create a queue and add all chunk data to it
        queue = asyncio.Queue()
        for chunk_data in all_chunks:
            queue.put_nowait(chunk_data)

        progress = tqdm(total=len(all_chunks), desc="Embedding and inserting into Supabase")

        async def worker():
            while not queue.empty():
                chunk_data = await queue.get()
                await self.embed_and_insert(chunk_data)
                progress.update(1)
                queue.task_done()

        # Start worker tasks
        num_workers = 35  # Number of concurrent workers, adjust as per rate limit
        tasks = []
        for _ in range(num_workers):
            task = asyncio.create_task(worker())
            tasks.append(task)

        await queue.join()

        for task in tasks:
            task.cancel()

        progress.close()
        await self.session.close()  # Close the aiohttp session

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        embedder = Embedder()
        await embedder.process_files_and_embed()

    asyncio.run(main())
