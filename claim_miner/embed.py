"""
Copyright Society Library and Conversence 2022-2024
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import shutil
import tarfile
from typing import Dict, Type, List, Union
from abc import abstractclassmethod, ABC

import requests
import numpy as np
from numpy.typing import NDArray
from asyncio import sleep
from . import config, run_sync


def normalization(embeds):
    norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    return embeds / norms


embedder_registry: Dict[str, Type[AbstractEmbedder]] = {}
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class AbstractEmbedder(ABC):
    dimensionality: int
    name: str
    display_name: str
    batch_size: int = 1

    @abstractclassmethod
    async def setup(cls):
        raise NotImplementedError()

    @abstractclassmethod
    async def embed(cls, text: Union[str, List[str]]) -> NDArray[np.float32]:
        raise NotImplementedError()


class Use4Embedder(AbstractEmbedder):
    dimensionality = 512
    name = "universal_sentence_encoder_4"
    display_name = "Google USE 4"
    model = None

    @classmethod
    async def setup(cls):
        if cls.model is None:
            import tensorflow_hub as hub

            cls.model = hub.load(
                "https://tfhub.dev/google/universal-sentence-encoder/4"
            )

    @classmethod
    def do_embed(cls, text: Union[str, List[str]]):
        import tensorflow as tf

        if isinstance(text, list):
            result = []
            for t in text:
                result.append([cls.do_embed(t)])
            return np.concatenate(result)

        return normalization(cls.model(tf.constant([text])))[0].numpy().astype(float)

    @classmethod
    async def embed(cls, text: Union[str, List[str]]) -> NDArray[np.float32]:
        await cls.setup()
        return await run_sync(cls.do_embed)(text)


class OpenAIEmbedder(AbstractEmbedder):
    client = None
    openai_name: str
    batch_size = 10
    omit_dimensions = False

    @classmethod
    async def setup(cls):
        if cls.client is None:
            from openai import AsyncOpenAI

            cls.client = AsyncOpenAI(
                api_key=config.get("openai", "api_key"),
                organization=config.get("openai", "organization"),
            )

    @classmethod
    async def embed(cls, text: Union[str, List[str]]) -> NDArray[np.float32]:
        from openai import RateLimitError

        await cls.setup()
        if is_single := not isinstance(text, list):
            text = [text]
        # TODO: Batch by size, handle oversize...
        while True:
            try:
                kwargs = dict(
                    model=cls.openai_name, input=text, dimensions=cls.dimensionality
                )
                if cls.omit_dimensions:
                    kwargs.pop("dimensions")
                results = await cls.client.embeddings.create(**kwargs)
                break
            except RateLimitError:
                await sleep(20)
        results = [r.embedding for r in results.data]
        if is_single:
            results = results[0]
        return results


class Ada2Embedder(OpenAIEmbedder):
    dimensionality = 1536
    omit_dimensions = True
    name = "txt_embed_ada_2"
    openai_name = "text-embedding-ada-002"
    display_name = "OpenAI ADA2"
    pricing = 0.0001


class Txt3SmallAbstract(OpenAIEmbedder):
    openai_name = "text-embedding-3-small"
    pricing = 0.00002


class Txt3Small512(Txt3SmallAbstract):
    name = "txt3_small_512"
    display_name = "OpenAI txt-3-embedding small 512"
    dimensionality = 512


class Txt3Small1536(Txt3SmallAbstract):
    name = "txt3_small_1536"
    display_name = "OpenAI txt-3-embedding small 1536"
    dimensionality = 1536


class Txt3LargeAbstract(OpenAIEmbedder):
    openai_name = "text-embedding-3-large"
    pricing = 0.00013


class Txt3Large256(Txt3LargeAbstract):
    name = "txt3_large_256"
    display_name = "OpenAI txt-3-embedding large 256"
    dimensionality = 256


class Txt3Large1024(Txt3LargeAbstract):
    name = "txt3_large_1024"
    display_name = "OpenAI txt-3-embedding large 1024"
    dimensionality = 1024


class OnnxEmbedder(AbstractEmbedder):
    model = None
    tokenizer = None
    dimensionality = 384
    batch_size = 32
    name = "all_minilm_l6_v2"
    display_name = "Onnx Minilm6.2"

    @classmethod
    async def setup(cls):
        if cls.model is None:
            from tokenizers import Tokenizer
            import onnxruntime

            fname = "onnx.tar.gz"
            download_path = Path.home() / ".cache" / "onnx_models"
            model_archive = download_path / fname
            model_path = download_path / "onnx"
            model_url = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
            if not model_archive.exists():
                # TODO: protect with an atomic lock
                download_path.mkdir()
                # TODO: httpx
                response = requests.get(model_url, stream=True)
                with model_archive.open("wb") as out_file:
                    shutil.copyfileobj(response.raw, out_file)
            if not model_path.exists():
                with tarfile.open(download_path / fname, "r:gz") as tar:
                    tar.extractall(download_path)
            tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
            tokenizer.enable_truncation(max_length=256)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
            cls.model = onnxruntime.InferenceSession(
                str(model_path / "model.onnx"), providers=["CPUExecutionProvider"]
            )
            cls.tokenizer = tokenizer

    @classmethod
    async def embed(cls, text: Union[str, List[str]]) -> NDArray[np.float32]:
        await cls.setup()
        if is_single := not isinstance(text, list):
            text = [text]
        results = []
        batch_size = 32
        for i in range(0, len(text), batch_size):
            batch = text[i : i + batch_size]
            encoded = [cls.tokenizer.encode(d) for d in batch]
            input_ids = np.array([e.ids for e in encoded])
            attention_mask = np.array([e.attention_mask for e in encoded])
            onnx_input = {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "token_type_ids": np.array(
                    [np.zeros(len(e), dtype=np.int64) for e in input_ids],
                    dtype=np.int64,
                ),
            }

            model_output = await run_sync(cls.model.run)(None, onnx_input)
            last_hidden_state = model_output[0]
            # Perform mean pooling with attention weighting
            input_mask_expanded = np.broadcast_to(
                np.expand_dims(attention_mask, -1), last_hidden_state.shape
            )
            embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(
                input_mask_expanded.sum(1), a_min=1e-9, a_max=None
            )
            embeddings = normalization(embeddings.astype(np.float32))
            results.append(embeddings)
        results = np.concatenate(results)
        if is_single:
            results = results[0]
        return results


class AngleEmbedder(AbstractEmbedder):
    name = "uae_l_v1"
    display_name = "UAE large v1"
    dimensionality = 1024
    model = None

    @classmethod
    async def setup(cls):
        if cls.model is None:
            if sys.platform == "darwin":
                # tensorflow_hub breaks if loaded after angle_emb
                import tensorflow_hub
            from angle_emb import AnglE

            cls.model = AnglE.from_pretrained(
                "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
            ).cuda()

    @classmethod
    def do_embed(cls, text: Union[str, List[str]]) -> NDArray[np.float32]:
        if isinstance(text, list):
            result = []
            for t in text:
                result.append([cls.do_embed(t)])
            return np.concatenate(result)
        return cls.model.encode(text, to_numpy=True)[0]

    @classmethod
    async def embed(cls, text: Union[str, List[str]]) -> NDArray[np.float32]:
        await cls.setup()
        return await run_sync(cls.do_embed)(text)


BASE_EMBED_MODEL_NAME = OnnxEmbedder.name

for cls in (
    OnnxEmbedder,
    AngleEmbedder,
    Ada2Embedder,
    Use4Embedder,
    Txt3Small512,
    Txt3Small1536,
    Txt3Large256,
    Txt3Large1024,
):
    embedder_registry[cls.name] = cls


async def tf_embed(text, model_name: str = BASE_EMBED_MODEL_NAME):
    embedder = embedder_registry[model_name]
    return await embedder.embed(text)


def batch(iterator, batch_size):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    yield batch


async def batch_async(iterator, batch_size):
    batch = []
    async for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    yield batch
