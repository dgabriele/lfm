# LFM Project Status

Last updated: 2026-04-06

## Trained Models

| Component | Key Metric |
|-----------|------------|
| PhraseDecoder v7 | Val CE = 0.0082, 11.6M full constituency phrases, 12 languages |
| Dialogue game | 98.3% accuracy at 100% hard negatives (16-way, 4 turns × 3 phrases) |
| Expression game | 95.4% accuracy, 99.997% surface diversity (300K expressions) |

## In Progress

- **Dialogue corpus generation**: 900K syllable-hyphenated IPA documents (4 turns each)
- **LLM pretraining**: Qwen 2.5 0.5B on dialogue corpus (pending corpus completion)

## Next

1. Few-shot translation evaluation
2. Reconstruction training (inverse decoder, alternative to games)
3. Multi-target discrimination
4. LIGO gravitational wave analysis
