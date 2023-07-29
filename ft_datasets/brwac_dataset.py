import os
import warnings
from typing import Dict, Iterable, Union

from functools import partial

from datasets import load_dataset, Dataset
import datasets

from transformers import AutoTokenizer

import numpy as np
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase


def concatenate_paragraphs(examples):
    outputs = []
    for exam in examples["text"]:
        full_text = ""
        paragraphs = exam["paragraphs"]
        for para in paragraphs:
            if len(para[0].split()) > 20:
                full_text += para[0] + " "
        outputs.append(full_text)

    return {"text": outputs}


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors, so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(
        self,
        hf_dataset: Union[datasets.IterableDataset, datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap

        self.bos_tokens = self.tokenizer(
            self.bos_text, truncation=False, padding=False, add_special_tokens=False
        )["input_ids"]
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f"You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error."
            )

        self.eos_tokens = self.tokenizer(
            self.eos_text, truncation=False, padding=False, add_special_tokens=False
        )["input_ids"]
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f"You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error."
            )

        eos_text_provided = self.eos_text != ""
        bos_text_provided = self.bos_text != ""
        test_text = self.tokenizer("")
        if len(test_text["input_ids"]) > 0 and (eos_text_provided or bos_text_provided):
            message = (
                "both eos and bos"
                if eos_text_provided and bos_text_provided
                else ("eos_text" if eos_text_provided else "bos_text")
            )
            warnings.warn(
                f"The provided tokenizer adds special tokens, but you also specified {message}. This may result "
                "in duplicated special tokens. Please be sure this is what you intend."
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        buffer = []
        for sample in self.hf_dataset:
            encoded = self.tokenizer(sample["text"], truncation=False, padding=False)
            iids = encoded["input_ids"]
            buffer = buffer + self.bos_tokens + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[: self.max_length]
                buffer = buffer[self.max_length :] if self.should_wrap else []
                yield {"tokens": np.asarray(concat_sample)}  # .tobytes()


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
hf_dataset = load_dataset("brwac", data_dir="../llm-foundry/data")
hf_dataset = hf_dataset["train"].map(concatenate_paragraphs, batched=True)
hf_dataset = hf_dataset.remove_columns(["doc_id", "title", "uri"])


max_length = 4096
bos_text = "<s>"
eos_text = "</s>"

train_dataset = ConcatTokensDataset(
    hf_dataset=hf_dataset,
    tokenizer=tokenizer,
    max_length=max_length,
    bos_text=bos_text,
    eos_text=eos_text,
    no_wrap=False,
)


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


ds = Dataset.from_generator(partial(gen_from_iterable_dataset, train_dataset))

ds.save_to_disk("data/data_processed")
