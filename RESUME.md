# Session Resume: Chat Format LLM Training

## What We're Doing

Redesigning LLM training from flat next-token-prediction on Neuroglot → instruct chat format so the model learns to **interpret** Neuroglot into English rather than speak it.

**Root problem**: LoRA rank=64 flat LM on Neuroglot overrides base model's English generation entirely. Model speaks Neuroglot instead of translating from it.

**Solution**: Format corpus as multi-turn chat, loss only on assistant turns. Neuroglot goes in user role → model continues in assistant role using base model's English capacity. Cross-lingual transfer (same mechanism as Chinese→English) bridges the gap at inference.

## Corpus Format

The corpus (`data/translator/dialogue_corpus_v7_natural.txt`) uses `paragraph_format=True`:
- Each line is one document: 4 sentences joined as natural paragraph
- Format: `"Sentence0. Sentence1. Sentence2. Sentence3."`
- Text is romanized IPA (isomorphic `romanize_iso()` mapping, not lossy)
- Documents are single lines (no blank-line separators in paragraph_format)

The training format we want:
```
[system] You are a Neuroglot language expert...
[user]   T0
[asst]   T1     ← loss here
[user]   T2
[asst]   T3     ← loss here
```

## Files To Modify / Create

### 1. `src/lfm/translator/tokenized_dataset.py`
Add `ChatTokenizedH5Dataset(TokenizedH5Dataset)`:
- `from_corpus(corpus_path, tokenizer, max_len=768, h5_path, val_fraction, system_prompt, chunk_size)`
- `_parse_turns(line) -> list[str] | None`: split on `'. '`, return 4 turns
- `_build_chat_example(turns, tokenizer, max_len, system_prompt, pad_id)`:
  - Use prefix-length approach to find assistant token spans:
    ```python
    prefix1_ids = tokenizer.apply_chat_template([sys, user0], tokenize=True, add_generation_prompt=True)
    prefix1_with_T1_ids = tokenizer.apply_chat_template([sys, user0, asst1], tokenize=True, add_generation_prompt=False)
    prefix2_ids = tokenizer.apply_chat_template([sys, user0, asst1, user2], tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(all_msgs, tokenize=True, add_generation_prompt=False)
    # T1 tokens: [len(prefix1_ids) : len(prefix1_with_T1_ids)]
    # T3 tokens: [len(prefix2_ids) : len(full_ids)]
    ```
  - Labels = -100 everywhere except T1 and T3 spans
  - Pad to max_len, attention_mask=1 for real tokens
- `_chat_tokenize_to_h5(lines, tokenizer, max_len, h5_path, system_prompt, chunk_size)`:
  - Process one line at a time (can't batch due to per-doc offset computation)
  - Write to HDF5 same structure as parent (input_ids, attention_mask, labels)
  - Skip/placeholder for unparseable lines, log skipped count

### 2. `src/lfm/translator/pretrain.py`
In `_build_dataloaders()`, add branch for `cfg.use_chat_format`:
```python
if cfg.use_chat_format:
    from lfm.translator.tokenized_dataset import ChatTokenizedH5Dataset
    train_ds, val_ds = ChatTokenizedH5Dataset.from_corpus(
        cfg.corpus_path, tokenizer,
        max_len=cfg.max_len,
        system_prompt=cfg.chat_system_prompt,
    )
else:
    train_ds, val_ds = TokenizedH5Dataset.from_corpus(...)
```

### 3. `configs/pretrain_lora_instruct_v7_chat.yaml` (NEW)
```yaml
model_name: "Qwen/Qwen2.5-0.5B"
corpus_path: "data/translator/dialogue_corpus_v7_natural.txt"
use_chat_format: true
chat_system_prompt: "You are a Neuroglot language expert. Neuroglot is an artificial language that expresses machine perceptions of the world."
use_lora: true
lora_r: 16          # reduced from 64 — enough to adapt, not enough to clobber English
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
max_len: 768        # fits 4-turn conversation
epochs: 5
lr: 2e-4
batch_size: 4
gradient_accumulation_steps: 8
use_amp: true
warmup_fraction: 0.05
output_dir: "data/translator_lora_chat_v7"
device: "cuda"
seed: 42
```

### 4. `scripts/interpret_samples.py`
Update system prompt for new model (which interprets rather than speaks Neuroglot):
```python
"You are an expert interpreter of Neuroglot, a natural language "
"with its own vocabulary, grammar, and meaning. It is not a "
"phonetic transcription of any known human language. "
"When given a Neuroglot passage, interpret what it is about in English."
```
This is already correct in the current version. No change needed unless model changes.

## Current State

### Already Done (from previous sessions)
- `src/lfm/translator/config.py`: `PretrainConfig` has `use_chat_format: bool = False` and `chat_system_prompt: str`
- `scripts/interpret_samples.py`: Fixed to use `romanize_iso()`, natural paragraph format, chat template, LoRA adapter detection
- `src/lfm/agents/diffusion.py`: `length_distribution_loss` uses sparsity on optional slots only (position 0 always free)
- `configs/dialogue_v7_phase1_vent2.yaml`: Updated to use sparsity-driven phrase count (no target_phrases)

### Not Yet Done
- `ChatTokenizedH5Dataset` class (main remaining task)
- `_build_dataloaders` branch in `pretrain.py`  
- `configs/pretrain_lora_instruct_v7_chat.yaml`
- Stop vast.ai current run (`translator_lora_instruct_v7`) and restart with new config

## Vast.ai Instance
- SSH: `ssh -o StrictHostKeyChecking=no -p 21066 -i ~/.ssh/id_ed25519 root@ssh6.vast.ai`
- Current run log: `/workspace/lfm/data/translator_lora_instruct_v7/pretrain.log`
- This run uses WRONG format (flat LM on romanized) — should be stopped and restarted with chat format

## Local vent2 Training
- Config: `configs/dialogue_v7_phase1_vent2.yaml`
- Was stopped during this session for evaluation; should be restarted
- Code changes are correct (sparsity on optional slots, no target_phrases)
- Restart: `poetry run lfm agent dialogue configs/dialogue_v7_phase1_vent2.yaml --resume data/dialogue_game_v7_vent2/latest.pt`

## Why This Design

1. **No explicit translation pairs**: The model never sees (Neuroglot, English_meaning) pairs. Instead it learns Neuroglot via next-token prediction within the chat format.
2. **Cross-lingual transfer**: At inference, presenting Neuroglot + English question to a model that understands Neuroglot (LoRA) but generates English (base model) activates zero-shot cross-lingual transfer — same mechanism Qwen uses for Chinese→English.
3. **LoRA rank=16**: Strong enough to shift Neuroglot distribution, not strong enough to override English generation entirely. Rank=64 (current run) is too strong.
4. **Loss on assistant tokens only**: Preserves base model's English generation capacity in the assistant role. T0/T2 in user role are context, not prediction targets.
