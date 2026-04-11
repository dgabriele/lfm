"""Unsupervised neural machine translation for Neuroglot ↔ English.

This package implements the Lample/Artetxe 2018 recipe for learning a
translator between two languages using only monolingual corpora — no
paired examples are ever used during training.

The design assumption is that the source side (Neuroglot) will
eventually come from agents perceiving arbitrary non-linguistic inputs
(LIGO waveforms, physical sensor streams, etc.), so no step in the
pipeline may depend on the Neuroglot having an aligned English source.
The current sentence-embedding experiment is a development harness; the
method must generalize.

Pipeline::

    monolingual Neuroglot         monolingual English
           │                             │
           └─────── shared SPM ──────────┘
                        │
               monolingual fasttext (both languages)
                        │
                 MUSE Procrustes alignment
                        │
                 shared-weight seq2seq
              (denoising autoencoder +
               iterative backtranslation)
                        │
                    translator
"""
