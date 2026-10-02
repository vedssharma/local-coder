import numpy as np
import pytest
from speculative import SmallModelDraft
from model_backend import EmbeddedModel, OpenAIModel


class Fake:
    def n_vocab(self): return 3
    def n_ctx(self): return 16
    def token_bos(self): return 0
    def token_eos(self): return 2
    def detokenize(self, ids, special=True): return bytes(ids)
    def tokenize(self, text): return [0, 1]
    def generate(self, ids, **kwargs):
        yield 1
        yield 2
        yield 1


def test_draft_stops_at_eos_and_checks_vocab_and_tokenization():
    target, draft_model = Fake(), Fake()
    draft = SmallModelDraft(draft_model, 8)
    draft.verify(target)
    assert draft(np.array([0, 1])).tolist() == [1, 2]
    assert draft(np.zeros(16, dtype=np.intc)).size == 0
    class Mismatch(Fake):
        def detokenize(self, ids, special=True): return b'other'
    with pytest.raises(ValueError, match='vocabularies'):
        SmallModelDraft(Mismatch()).verify(target)
    class EncodingMismatch(Fake):
        def tokenize(self, text): return [1, 0]
    with pytest.raises(ValueError, match='tokenization'):
        SmallModelDraft(EncodingMismatch()).verify(target)


def test_speculation_is_explicit_and_backend_checked():
    EmbeddedModel({'speculative_mode': 'off'})
    with pytest.raises(ValueError):
        EmbeddedModel({'speculative_mode': 'draft-model'})
    with pytest.raises(ValueError):
        OpenAIModel({'backend': 'openai', 'speculative_mode': 'prompt-lookup'})


def test_native_loader_supplies_prompt_lookup_callback(monkeypatch, tmp_path):
    (tmp_path / 'target.gguf').touch()
    monkeypatch.chdir(tmp_path)
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    import llama_cpp
    constructor = MagicMock()
    callback = MagicMock()
    monkeypatch.setattr(llama_cpp, 'Llama', constructor)
    monkeypatch.setitem(sys.modules, 'llama_cpp.llama_speculative', SimpleNamespace(LlamaPromptLookupDecoding=callback))
    profile = {'model_path': 'target.gguf', 'n_ctx': 1024, 'n_gpu_layers': 0,
               'speculative_mode': 'prompt-lookup', 'draft_tokens': 4}
    EmbeddedModel(profile).load()
    callback.assert_called_once_with(max_ngram_size=2, num_pred_tokens=4)
    assert constructor.call_args.kwargs['draft_model'] is callback.return_value


def test_incompatible_learned_draft_releases_both_models(monkeypatch, tmp_path):
    (tmp_path / 'target.gguf').touch()
    monkeypatch.chdir(tmp_path)
    import llama_cpp
    class Model(Fake):
        def __init__(self, mismatch=False):
            self.closed = False
            self.metadata = {'tokenizer.ggml.pre': 'different' if mismatch else 'same'}
        def close(self): self.closed = True
    draft, target = Model(), Model(mismatch=True)
    monkeypatch.setattr(llama_cpp, 'Llama', lambda **kw: draft if kw['model_path'] == 'draft.gguf' else target)
    model = EmbeddedModel({'model_path': 'target.gguf', 'draft_model_path': 'draft.gguf',
                           'n_ctx': 1024, 'n_gpu_layers': 0, 'speculative_mode': 'draft-model'})
    with pytest.raises(ValueError, match='metadata'):
        model.load()
    assert draft.closed and target.closed
    assert model._model is None and model._draft is None
