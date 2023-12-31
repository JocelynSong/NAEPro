# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
import os
import torch
import torch.nn.functional as F

from omegaconf import MISSING, II, OmegaConf

import numpy as np
from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
    indexed_dataset
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.indexed_dataset import get_available_dataset_impl

from .language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES


logger = logging.getLogger(__name__)


@dataclass
class MutationBertConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    dataset_impl: ChoiceEnum(get_available_dataset_impl()) = field(
        default="binarize",
        metadata={"help": "data saving format"}
    )
    source_lang: str = field(
        default="viral",
        metadata={"help": "source side language"}
    )
    target_lang: str = field(
        default="mutation",
        metadata={"help": "target side language"}
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"},
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"},
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    seed: int = II("common.seed")


@register_task("mutation_bert_task", dataclass=MutationBertConfig)
class MutationBertTask(FairseqTask):

    cfg: MutationBertConfig

    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, cfg: MutationBertConfig, dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary

        # add mask token
        self.mask_idx = dictionary.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, cfg: MutationBertConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(cfg, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, "{}.{}-{}.{}.pth".format(split, src, tgt, lang))
            return indexed_dataset.dataset_exists(filename, impl=self.cfg.dataset_impl)

        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        src, tgt = self.cfg.source_lang, self.cfg.target_lang
        if split_exists(split, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split, src, tgt))
        elif split_exists(split, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split, tgt, src))
        else:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, data_path)
            )

        src_split_path = prefix + src + ".pth"
        tgt_split_path = prefix + tgt + ".pth"

        src_dataset = data_utils.load_indexed_dataset(src_split_path, self.dictionary, self.cfg.dataset_impl)
        tgt_dataset = data_utils.load_indexed_dataset(tgt_split_path, self.dictionary, self.cfg.dataset_impl)

        if src_dataset is None or tgt_dataset is None:
            raise FileNotFoundError("Dataset not found: {} ({})".format(split, src_split_path))

        assert len(src_dataset) == len(tgt_dataset) or len(tgt_dataset) == 0

        src_dataset = PrependTokenDataset(src_dataset, self.dictionary.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, self.dictionary.bos())

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": {
                        "src_tokens": RightPadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                    },
                    "target": RightPadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = RightPadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.cfg.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        hypos = []
        masked_tokens = None
        with torch.no_grad():
            outputs = models[0](**sample["net_input"], masked_tokens=masked_tokens)

            # logits = F.log_softmax(outputs[0], dim=-1)  # [B, T, C]
            logits = torch.log(outputs[0])
            targets = models[0].get_targets(sample, [logits])   # [B, T]
            sources = models[0].get_sources(sample)
            targets = torch.cat((targets[:, 0].unsqueeze(1), targets[:, 2:]), 1)  # remove the label tag
            sources = torch.cat((sources[:, 0].unsqueeze(1), sources[:, 2:]), 1)  # remove the label tag
            masked_tokens = targets.ne(self.dictionary.pad_index)

            if masked_tokens is not None:
                targets = targets[masked_tokens].reshape(targets.size(0), -1)
                sources = sources[masked_tokens].reshape(targets.size(0), -1)
                logits = logits[masked_tokens].reshape(logits.size(0), -1, logits.size(-1))

            mutated_pos = (sources != targets)  # True for mutation, and False for not
            true_pos = ~mutated_pos
            true_pos = true_pos.long()
            change_pos = mutated_pos.long()

            mutated_pos = mutated_pos.long().unsqueeze(-1)  # [B, T, 1]
            scores = logits * mutated_pos
            sorted_scores, indices = torch.sort(scores, dim=-1, descending=True)  # [B, T, C]
            for i in range(generator.beam_size):
                hypo = sources * true_pos + indices[:, :, i].squeeze(-1) * change_pos
                hypos.append(hypo)

            # pred_scores = F.softmax(outputs[0], dim=-1)
            pred_scores = outputs[0]
            sorted_scores, indices = torch.sort(pred_scores, dim=-1, descending=True)  # [B, T, C]
            pred_scores = sorted_scores[:, :, 0]
            pred_scores = pred_scores * change_pos
            for i in range(change_pos.size(0)):
                for j in range(change_pos.size(1)):
                    if change_pos[i][j] == 1 and (targets[i][j] != indices[i][j][0]):
                        change_pos[i][j] = 0
        pred_scores = pred_scores.reshape(-1).float()
        change_pos = change_pos.reshape(-1)
        print(len(pred_scores))
        assert len(pred_scores) == len(change_pos)
        pred_probs, labels = [], []
        for i in range(len(pred_scores)):
            if pred_scores[i] == 0:
                continue
            pred_probs.append(pred_scores[i].item())
            labels.append(change_pos[i].item())
        print(len(pred_probs))
        print(len(labels))
        return hypos, pred_probs, labels
