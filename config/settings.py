from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # SEC EDGAR
    sec_user_agent: str = "Somaya Jain somayaj@andrew.cmu.edu"
    sec_rate_limit: float = 0.11  # seconds between requests (max 10 req/sec)

    # Financial Modeling Prep
    fmp_api_key: str = ""
    fmp_base_url: str = "https://financialmodelingprep.com/api/v3"

    # OpenAI via CMU AI Gateway (LiteLLM proxy)
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o"
    openai_base_url: str = "https://ai-gateway.andrew.cmu.edu"

    # AWS Bedrock (Titan Embeddings)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dim: int = 1024

    # Chunking
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    tokenizer_name: str = "cl100k_base"

    # FAISS
    faiss_index_type: str = "FlatIP"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
