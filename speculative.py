"""Learned draft callback for llama-cpp-python's target-verified decoding."""
import hashlib
import struct
import numpy as np


def tokenizer_signature(model):
    digest = hashlib.sha256()
    digest.update(struct.pack('!iii', model.n_vocab(), model.token_bos(), model.token_eos()))
    for token in range(model.n_vocab()):
        piece = model.detokenize([token], special=True)
        digest.update(struct.pack('!I', len(piece)))
        digest.update(piece)
    return digest.digest()


class SmallModelDraft:
    """Propose greedy tokens. The target model alone accepts/rejects proposals."""
    def __init__(self, model, num_tokens=8):
        self.model = model
        self.num_tokens = num_tokens

    def verify(self, target):
        # Compare tokenizer metadata (including merges/normalization) when exposed;
        # chat templates affect prompting, not token IDs, and may legitimately differ.
        def metadata(model):
            return {k: v for k, v in getattr(model, 'metadata', {}).items()
                    if k.startswith('tokenizer.') and 'chat_template' not in k}
        if metadata(target) != metadata(self.model):
            raise ValueError('Draft and target tokenizer metadata differs')
        if tokenizer_signature(target) != tokenizer_signature(self.model):
            raise ValueError('Draft and target token vocabularies/special IDs differ')
        # Check behavior on representative text as well as every vocabulary ID.
        for sample in (b'hello world', b'def add(a, b):\n    return a + b', 'café 中文'.encode()):
            if target.tokenize(sample) != self.model.tokenize(sample):
                raise ValueError('Draft and target tokenization differs')

    def __call__(self, input_ids, **kwargs):
        room = min(self.num_tokens, self.model.n_ctx() - len(input_ids))
        if room <= 0:
            return np.array([], dtype=np.intc)
        tokens = []
        generator = self.model.generate(input_ids.tolist(), temp=0, top_k=1, top_p=1, reset=True)
        try:
            for token in generator:
                tokens.append(token)
                if token == self.model.token_eos() or len(tokens) >= room:
                    break
        finally:
            generator.close()
        return np.array(tokens, dtype=np.intc)

    def close(self):
        self.model.close()
