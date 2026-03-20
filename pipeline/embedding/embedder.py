"""Embed text chunks using Amazon Titan Embeddings V2 via AWS Bedrock."""

import json
import logging
import time

import boto3
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class TitanEmbedder:
    """Embeds text using Amazon Titan Text Embeddings V2 via Bedrock."""

    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_default_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
        self.model_id = settings.embedding_model_id
        self.dim = settings.embedding_dim

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns a normalized 1-D vector."""
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        vec = np.array(result["embedding"], dtype=np.float32)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts. Returns an (N, dim) normalized array.

        Titan doesn't support batch embedding, so we call one at a time
        with rate limiting and progress tracking.
        """
        n = len(texts)
        vectors = np.zeros((n, self.dim), dtype=np.float32)
        failed = []

        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 100 == 0:
                logger.info(f"Embedding progress: {i + 1}/{n}")

            try:
                vectors[i] = self.embed_single(text)
            except Exception as e:
                logger.warning(f"Failed to embed chunk {i}: {e}")
                failed.append(i)
                # Brief pause on error to avoid rate limit cascades
                time.sleep(1)

        if failed:
            logger.warning(
                f"Retrying {len(failed)} failed embeddings..."
            )
            for i in failed:
                try:
                    time.sleep(0.5)
                    vectors[i] = self.embed_single(texts[i])
                except Exception as e:
                    logger.error(f"Retry failed for chunk {i}: {e}")

        logger.info(f"Embedded {n} chunks ({n - len(failed)} succeeded on first pass)")
        return vectors
