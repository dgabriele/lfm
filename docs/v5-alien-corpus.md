# v5 Alien Corpus

**Status:** Generation in progress (3 passes × 1M paragraphs, seeds 1/2/3, on Vast 3090).
**Source:** `data/synth_qwen_multisent_v5/` artifacts + `data/synth_contrastive_multisent_v5/latest.pt` (Phase 2 step ~5500).
**Final output:** `data/synth_qwen_multisent_v5/corpus_3M.txt` (~3M paragraphs, ~750M alien BPE tokens).

## What this corpus is

A self-supervised pretraining corpus for downstream UNMT — alien-language paragraphs only, no parallel English. Each paragraph is the alien-surface expression of one source embedding, produced by:

1. **Source embedding** drawn from the 1M-passage Qwen-2.5-0.5B contextual store (`data/embeddings_qwen/`). Currently Qwen-of-English, but the pipeline is source-agnostic — any continuous representation works.
2. **PrefixProjector (zero-init)** maps the source vector into 8 prefix tokens for the body.
3. **Phase 1 body** — Qwen-2.5-0.5B body fine-tuned on 897K cipher-encoded paragraphs from a multi-sentence English corpus (wikitext-103 + CC-news, NER-normalised, `@-@`-escape-stripped).
4. **v5 hierarchical cipher** assigns each English word a depth-3 vocabulary-tree cluster code `(c₁, c₂, c₃) ∈ [8]³`, mapping syllable positions to nested syllable-vocab slices. Semantically related source words share leading alien syllables; this is the central architectural difference vs v4's pure-hash cipher.
5. **Generation** — autoregressive sampling from the body conditioned on the projector's prefix.

## Filter pipeline (per generated paragraph)

Each candidate is salvaged + checked by `dump_phase2_corpus_filtered.py::normalize_paragraph`:

- **Salvage:** trim any partial trailing sentence after the last `. ? !`.
- **Reject if:** no terminal punctuation anywhere; <2 sentences after trim; <6 tokens; ≥5 identical tokens in a row; ≥6 consecutive ≤1-char tokens (broken BPE fragmentation).
- **Naturalise:** collapse whitespace before punctuation; capitalise paragraph-initial + post-`. ? !` characters.

Rejected paragraphs trigger up to **2 retries** with different RNG seeds for the same anchor; if still unsalvageable, dropped (loses one entry from the 1M; expected drop rate ≈ 0.06% per pass at observed 97.5% first-attempt pass rate).

## Sample paragraphs (from `corpus_preview.txt`, naturalised)

The samples below cover the length distribution from p10 to p90.

### Short (≈100 tokens, 640 chars)

> Lîvkòk zej sômsíj héthàjháhnih ' rër víwwädwen, hëlhênhëdfëk fä wge v fûh hómhéh tòbsãksàr jäwhêthëbnôm ” rùs pür gîthärhëhgûg mïgmân pãn gàz lásjõgjòd gîthärhëhgûg källevlâz, wïfvõdvíbhôp râh pawpêgpãrkez. Rìv hûngûthánlûn lunnufnõfjõ tòtsàt dùlfèp rum lãslòd tãksìd sômsíj wimwïh dùlfèp rùs sômsíj v ij zêd zèj dän zej pawpêgpãrkez, nofnöbnòl püg " pür lâwjüd zej jêphõm faldob géwgàjfutfï põl sômsíj wikwât vèp zèf zâp kãnkiskîz sômsíj vupwöl. Pawpêgpãrkez mïjmàlmâz râh pür zàtzagwüd rófpúh nísnàv pawpêgpãrkez, wã dwa g võr pür dï d fûm ful fûf fûs. Sômsíj dêt z õ ke ' jâp, gí p hêb hèh hùfhij jìnháfgüjfel võ sv õ s rùs gùfhâwgüdzãd.

### Medium (≈153 tokens, 840 chars)

> Död jr â s põl sômsíj nös lúv lôg vòz ( guwgûrhájrís tîgsín, pawpêgpãrkez ). Sû w sù s fôv sômsíj vófvûs rùs sômsíj jìnháfgüjfel sutsâf nar nub nöw b b ìv. Lunnufnõfjõ rügpöz, püg sùssàm põgpúj pür guwgûrhájrís râh sijsáz wòz kîk kïl lèn këd këb kõv jüt júp gèp. Sômsíj gökhékhêvwát sùssàm wã zv ïb víj rùs sômsíj gíkhâtgüglòm lèzkîf, lunnufnõfjõ sômsíj juglêhléb tîgsín vohvíbvír râh kul kòv wík dìn dïs ta m, kúmléklés sômsíj kú s hïljäshüfvïh, géh tîgsín là h kis kéw gìg fìdgäggáf zej sômsíj konjüh kek jút jús kèl mîb. Rün gùh põgpúj guwhíw rãbpús zej " gõbhéthêzkâg ", pulpõz gûb séfsãf sùlsáz pïd jêphõm vîvwóg sólsãt pèmpúg góghökhút vab sîz patpès tò w sà k sèvsëj sãpsëf sijsáz põtpâw nòz râh rãm. Sômsíj jon kív kêv sâvsàn rúl rùs héthàjháhnih pöl jeghâzgümmöj pêf sômsíj jâhhathàgjäv nè b nâ k nös mùh lôg mïz mòf patpès líjjón.

### Long (≈208 tokens, 1141 chars)

> Lík tìmsawsàd pawpêgpãrkez tîgsín gîthärhëhgûg héthàjháhnih jänhanhazdãl tîgsín sômsíj tâw ví vw èt döb nãf pünpõb sômsíj kùbkëzkìd zej gökhékhêvwát juglêhléb pawpêgpãrkez hùfhij dëd dêp rùs rêr tefsêb gìbhug gíkhâtgüglòm rùs gùfhâwgüdzãd. " püg sùssàm míf nöb tefsêb tãksìd püg dùlfèp nôp nàp nùzpüv fópfúbfun júm. Rùs sômsíj gát fáz dür tîgsín tõvsim vûfv rùs fàg kësjód dùlfèp mag nóv nõj ned mâz máf rùs jeghâzgümmöj tãksìd jarhanhatwök rë d pê h ' jïp jîb sutsâf wòb zèn záz z ò. Sõzság — pür nè n mõw mûd ta p sômsíj wafvünvõj tîgsín gí p hêb hèh rùs sômsíj jôj gar tîgsín sî ne v mà j lusnanmùl rùs sômsíj jûj dîs zej sômsíj gîthärhëhgûg nõg nõl tòd sídsãt jeghâzgümmöj ' fê bdä d dèf. Sùpsâr dùlfèp mèm nõp nòv vak sômsíj kàjkêtkir zej sômsíj wimwïh tõvsim fëb dön dur dîh tîgsín püg mô b nöj rìv sômsíj wáz wãg wáv gèh gän gek duk patpès sômsíj loh jïk jôg düsdîkdòsrew dïpgan. Puj sïb lüj jômlëb dùlfèp vibwur zèmzevzãw rùs jâhhathàgjäv, püg dugféf mëwnuknõm rùs sômsíj nejmïv kòs fë gf é s fín kôz júm tãksìd jêphõm tëjsìz pür wíf zãl zêf patpès nab mük müj pür sössäw vég fùg gêbfòz rùs lö h mow mïw gîrhávgüdzéb hïfhàphâpwém....

## Surface-structure notes

- **Multi-sentence:** mean 5.4 sentences/paragraph (median 5).
- **Length:** mean 193 tokens/paragraph (p10=100, p90=256, max=256).
- **Vocab use:** ~77% of the 8K alien-BPE vocabulary appears in any 5K-paragraph slice.
- **Function words** (top 5): `sômsíj` (≈ "the"), `zej`, `tîgsín`, `rùs`, `râh`. These have natural Zipfian dominance, mirroring source-corpus function-word frequency.
- **NER placeholder tokens** (`pawpêgpãrkez`, `gökhékhêvwát`, `gîthärhëhgûg`, `héthàjháhnih`) account for ~8% of all tokens. This reflects the source corpus's NER normalisation (every entity surface → typed slot like `personnamea`/`organizationnamea`).
- **Cluster-mate prefixes:** runs like `mïn mîd môf mé v` or `dïf gâf gãs` (multiple alien tokens with shared leading-character morphology) are direct surface evidence that the hierarchical cipher's cluster slices are being used coherently within paragraphs.
- **BPE fragmentation runs** (e.g. `tâ jt ej`) appear in roughly 0.2% of paragraphs and reflect within-leaf hash variability that the BPE didn't merge. Not filtered out — Qwen's downstream tokenizer will re-merge.

## Pipeline reproduction

```bash
# 1. Cluster-aware cipher artifact
poetry run python scripts/build_semantic_cipher_clusters.py \
    --corpus data/multisent_corpus/passages_normalized_clean.jsonl \
    --output-dir data/synth_qwen_multisent_v5 --mode hierarchical --sweep

# 2. Encode corpus + train BPE on alien output
poetry run lfm synth build-vocab configs/synth_multisent_local_v5.yaml

# 3. Build n-gram reference distributions
for N in 2 3 4; do
  poetry run python scripts/build_synth_ngram.py \
      configs/synth_multisent_local_v5.yaml --n $N
done

# 4. Phase 1 body fine-tune (Qwen 2.5 0.5B body, 25K steps)
poetry run lfm synth train-phase1 configs/synth_multisent_local_v5.yaml

# 5. Phase 2 contrastive game (15K steps)
poetry run lfm agent synth configs/synth_contrastive_multisent_v5_vast.yaml

# 6. Generate filtered 3-pass corpus
for SEED in 1 2 3; do
  poetry run python scripts/dump_phase2_corpus_filtered.py \
      --config configs/synth_contrastive_multisent_v5_vast.yaml \
      --p2-checkpoint data/synth_contrastive_multisent_v5/latest.pt \
      --n 1000000 --batch 128 --seed $SEED --max-retries 2 \
      --out data/synth_qwen_multisent_v5/corpus_pass${SEED}.txt
done
cat data/synth_qwen_multisent_v5/corpus_pass{1,2,3}.txt > data/synth_qwen_multisent_v5/corpus_3M.txt
```

## Downstream use

This corpus is the **input to UNMT pretraining of Qwen-2.5 7B Instruct**, fully unsupervised — alien tokens only, next-token prediction, no English alignment. Translation capability is then evaluated separately via few-shot prompting on a held-out LLM.
