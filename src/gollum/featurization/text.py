from dataclasses import dataclass
import os
from typing import Optional
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

import numpy as np

from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer,
    AutoModel,
)
from functools import partial
import torch.nn.functional as F


from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

# from openai import OpenAI

# client = OpenAI()

from transformers import AutoTokenizer
from gollum.featurization.utils.pooling import average_pool, last_token_pool, weighted_average_pool





@lru_cache(maxsize=None)
def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return (
        _get_openai_client().embeddings.create(input=[text], model=model).data[0].embedding
    )


def ada_embeddings(texts, model="text-embedding-ada-002"):
    """
    Get ADA embeddings for a list of texts.

    :param texts: List of texts to be embedded
    :type texts: list of str
    :param model: Model name to use for embedding (default is "text-embedding-ada-002")
    :type model: str
    :return: NumPy array of ADA embeddings
    """
    get_embedding_with_model = partial(get_embedding, model=model)

    # Cap workers to avoid spawning N processes for N texts (2D fix)
    n_workers = min(8, max(1, len(texts)))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        embeddings = list(
            tqdm(
                executor.map(get_embedding_with_model, texts),
                total=len(texts),
                desc="Getting Embeddings",
            )
        )
    return np.array(embeddings)


def ada_embeddings_3(texts, model="text-embedding-3-small"):
    return ada_embeddings(texts, model=model)


from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5EncoderModel, T5Config
from transformers import LlamaModel, LlamaConfig



@dataclass
class ModelConfig:
    name: str
    config_class: Optional[any] = None
    model_class: Optional[any] = None
    dropout_field: str = "dropout_rate"


MODEL_CONFIGS = {
    "t5-base": ModelConfig("t5-base", T5Config, T5EncoderModel),
    "GT4SD/multitask-text-and-chemistry-t5-base-augm": ModelConfig(
        "GT4SD/multitask-text-and-chemistry-t5-base-augm",
        T5Config,
        T5EncoderModel,
    ),
    "Rostlab/prot_t5_xl_uniref50": ModelConfig(
        "Rostlab/prot_t5_xl_uniref50",
        T5Config,
        T5EncoderModel,
    ),
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp": ModelConfig(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        LlamaConfig,
        LlamaModel,
        "attn_dropout",
    ),
}


def _is_esmc_model(model_name: str) -> bool:
    name = model_name.lower()
    return "esmc" in name or "evolutionaryscale/esmc" in name


def _normalize_esmc_model_name(model_name: str) -> str:
    normalized = model_name.lower()
    if "600m" in normalized:
        return "esmc_600m"
    if "300m" in normalized:
        return "esmc_300m"
    return "esmc_600m"


def get_model_and_tokenizer(model_name: str, device: str='cuda'):

    if _is_esmc_model(model_name):
        from esm.models.esmc import ESMC

        esmc_name = _normalize_esmc_model_name(model_name)
        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = ESMC.from_pretrained(esmc_name, device=torch_device).to(torch_device)
        tokenizer = model.tokenizer
        return model, tokenizer

    if "prot_t5" in model_name.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if model_config := MODEL_CONFIGS.get(model_name):
        # Known small models (e.g. T5) stay in float32
        config = model_config.config_class.from_pretrained(model_name)
        setattr(config, model_config.dropout_field, 0)
        torch_dtype = torch.bfloat16 if "prot_t5" in model_name.lower() else torch.float32
        model = model_config.model_class.from_pretrained(
            model_name, config=config, torch_dtype=torch_dtype
        ).to(device)
    else:
        # bfloat16 halves VRAM for large models (1B fix)
        _dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModel.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=_dtype,
        )

    # _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def get_tokens(
    texts,
    model_name="WhereIsAI/UAE-Large-V1",
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print(model_name, "for get tokens")
    if _is_esmc_model(model_name):
        from esm.tokenization import get_esmc_model_tokenizers

        tokenizer = get_esmc_model_tokenizers()
        token_batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            ids = encoded.get("input_ids")
            if ids is None:
                ids = encoded["sequence_tokens"]
            masks = encoded.get("attention_mask")
            if masks is None:
                masks = (ids != tokenizer.pad_token_id).long()

            pad_len = 512 - ids.size(1)
            if pad_len > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_len), value=tokenizer.pad_token_id)
                masks = torch.nn.functional.pad(masks, (0, pad_len), value=0)

            token_batches.append(torch.cat([ids, masks], dim=1))

        all_encoded_inputs = torch.cat(token_batches, dim=0)
        return all_encoded_inputs.cpu().numpy()

    if "prot_t5" in model_name.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
        # ProtT5 requires space-separated amino acids
        texts = [" ".join(list(seq)) for seq in texts]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize in batches to avoid allocating 3 full-dataset copies at once
    # (1D fix). Collect [input_ids | attn_mask] rows; concatenate once at end.
    encoded_batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        # Pad to a fixed width (512) so all batches have consistent columns
        ids = encoded.input_ids  # (B, seq_len)
        masks = encoded.attention_mask  # (B, seq_len)
        # Right-pad each batch to max_length=512 for consistent stacking
        pad_len = 512 - ids.size(1)
        if pad_len > 0:
            ids = torch.nn.functional.pad(ids, (0, pad_len), value=tokenizer.pad_token_id)
            masks = torch.nn.functional.pad(masks, (0, pad_len), value=0)
        encoded_batches.append(torch.cat([ids, masks], dim=1))

    all_encoded_inputs = torch.cat(encoded_batches, dim=0)
    return all_encoded_inputs.cpu().numpy()


def get_huggingface_embeddings(
    texts,
    model_name="tiiuae/falcon-7b",
    max_length=512,
    batch_size=8,
    pooling_method="cls",
    prefix=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    normalize_embeddings=False,
):
    """
    General function to get embeddings from a HuggingFace transformer model.
    """
    print(f"featurizing with {model_name}")
    model, tokenizer = get_model_and_tokenizer(model_name, device)
    left_padding = tokenizer.padding_side == "left"
    model.eval()

    # ProtT5 requires space-separated amino acids
    if "prot_t5" in model_name.lower():
        texts = [" ".join(list(seq)) for seq in texts]

    # optionally add prefix to each text
    if prefix:
        texts = [prefix + text for text in texts]

    pooling_functions = {
        "average": average_pool,
        "cls": lambda x, _: x[:, 0],
        "last_token_pool": partial(last_token_pool, left_padding=left_padding),
        "weighted_average": weighted_average_pool,
    }

    # Pre-allocate output array after first batch to avoid repeated np.concatenate
    # copies (3A fix).
    output = None
    write_idx = 0
    for i in tqdm(
        range(0, len(texts), batch_size), desc=f"Processing with {model_name}"
    ):
        batch_texts = texts[i : i + batch_size]
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded_input)
            pooled = pooling_functions[pooling_method](
                outputs.last_hidden_state, encoded_input["attention_mask"]
            )

            if normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)
            batch_np = pooled.cpu().numpy()

        if output is None:
            output = np.empty((len(texts), batch_np.shape[1]), dtype=batch_np.dtype)
        batch_len = batch_np.shape[0]
        output[write_idx : write_idx + batch_len] = batch_np
        write_idx += batch_len

        torch.cuda.empty_cache()

    return output


def get_sentence_transformer_embeddings(
    texts, model_name="bigscience/sgpt-bloom-7b1-msmarco", batch_size=32
):
    model = SentenceTransformer(model_name)
    embeddings_list = []
    for i in tqdm(
        range(0, len(texts), batch_size), desc=f"Processing with {model_name}"
    ):
        batch_texts = texts[i : i + batch_size]
        embeddings = model.encode(batch_texts)
        embeddings_list.append(embeddings)

    return np.concatenate(embeddings_list, axis=0)


def instructor_embeddings(
    texts,
    model_name="hkunlp/instructor-xl",
    instruction="Represent the chemistry procedure: ",
    normalize=False,
):
    """
    Get Instructor embeddings for a list of texts.

    :param texts: List of texts to be embedded
    :type texts: list of str
    :param model_name: Pretrained model name to use for embedding
    :type model_name: str
    :param instruction: Instruction string for the embedding task
    :type instruction: str
    :return: NumPy array of Instructor embeddings
    """
    # Load the INSTRUCTOR model
    model = INSTRUCTOR(model_name).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 32
    sentence_embeddings_list = []
    paired_texts = [[instruction, text] for text in texts]

    for i in tqdm(range(0, len(paired_texts), batch_size)):
        batch_embeddings = model.encode(
            paired_texts[i : i + batch_size], normalize_embeddings=normalize
        )
        sentence_embeddings_list.append(batch_embeddings)


    return np.concatenate(sentence_embeddings_list, axis=0)
















