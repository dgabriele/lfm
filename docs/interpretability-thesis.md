# LFM as Perceptual Interpretability

## The Core Claim

LFM provides a method for neural systems to express their learned representations as natural language, enabling a new form of interpretability through LLM-mediated dialogue.

Existing interpretability methods ask "does the model represent X?" where X is a human-predefined category. Probing classifiers, mechanistic interpretability, concept bottleneck models, network dissection — they all impose human ontology on the model and test whether it conforms.

LFM inverts this. Instead of probing the model with human hypotheses, the model expresses its perception in its own emergent language, and a human discovers the model's categories through an LLM interpreter. The model's ontology is the output, not the input to the analysis.

## How It Differs from Existing Approaches

| Method | Direction | Limitation |
|--------|-----------|------------|
| Probing classifiers | Human → Model ("do you represent X?") | Only tests predefined categories |
| Mechanistic interpretability | Human → Model ("what does this circuit do?") | Static component analysis, not holistic perception |
| Concept bottleneck models | Human → Model ("use these concepts") | Concepts are predefined, not emergent |
| Network dissection | Human → Model ("which neurons match label Y?") | Human categories imposed |
| **LFM** | **Model → Human ("this is what I perceive")** | **Model expresses its own ontology** |

The fundamental difference is the direction of information flow. Current methods project human understanding onto the model. LFM lets the model project its understanding onto human language.

## Perceptual Feedback

Current feedback mechanisms (RLHF, DPO, fine-tuning) operate on model outputs: "this answer is wrong, here's the right one." They correct behavior without understanding why the model erred, treating internal perception as a black box.

If a model can express how it perceives an input — through its emergent language, interpreted by an LLM — you can diagnose the perceptual error, not just the output error. The feedback becomes "you're not wrong about the answer, you're wrong about what you're looking at."

A vision model misclassifies an image. Current approach: "that's not a dog, it's a wolf" — the model adjusts its decision boundary. LFM approach: the model expresses its perception, the interpreter renders it as "a furry quadruped with domestic posture near human structures." The perceptual gap is visible — the model is attending to posture and context but missing morphological differences. Feedback can target the root cause: "look at the snout shape and ear angle."

This extends to alignment. Harmful behavior may stem from bad perception — misreading a situation — not bad values. Seeing how the model perceives the situation enables correcting the perception rather than just punishing the output.

## Relationship to Machine Consciousness Research

LFM does not claim to detect consciousness. But it provides something that did not previously exist: a channel through which a system can express its subjective perceptual state in a form that humans can interrogate through dialogue.

The emergent language is the system's own. The frozen decoder imposes linguistic structure but not linguistic content. The content comes from the agent's learned representation — what it chose to encode, what distinctions it made, what it collapsed. When an LLM interprets a passage as "a spiraling tension between two rigid structures," that reflects something the system itself encoded, not something humans told it to say.

This does not prove the system "experiences" spiraling tension. But it opens a new empirical surface for the debate. Instead of arguing whether architectures are sufficient for consciousness in the abstract, researchers can examine what specific systems express about specific inputs and ask whether those expressions show properties associated with perceptual experience: consistency, differentiation, integration, context-sensitivity.

The contribution is moving the question from metaphysics to empirics, without overclaiming.

## What Faithful Interpretation Means

The LLM's interpretation of a Neuroglot passage is not a translation of the original input to the source encoder. It is a linguistic interpretation of how the agent — the system trained via the communication game — learned to perceive the input.

If the source encoder (e.g., a sentence-transformer) encodes "China's economy grew" and "India's economy grew" with similar embeddings, and the agent treats them identically (same region of its representation space), the LLM should produce one interpretation for both. That is not a failure — it is a faithful report of the agent's perception.

Faithfulness is measured topologically:

1. **Consistency**: Similar inputs to the agent produce similar interpretations
2. **Differentiation**: Different inputs produce different interpretations
3. **Stability**: The same input produces semantically consistent interpretations across runs
4. **Structure preservation**: The cosine similarity structure of the interpretations (re-embedded) correlates with the similarity structure of the agent's internal representations

None of these require knowing the "correct" interpretation. They measure whether the interpretation preserves the structure of the agent's learned representation space.

## Source Agnosticism

The approach is not specific to any particular source encoder or domain. The frozen decoder, the communication game, the LLM interpreter — none of these depend on the source of the input embeddings. The current proof of concept uses sentence-transformer embeddings, but the same pipeline applies to:

- Vision encoders (how does the model perceive this image?)
- Audio encoders (what does it hear in this signal?)
- Scientific instruments (how does a trained model perceive a gravitational wave?)
- Multi-agent systems (what is this agent's model of the environment?)
- Protein structure encoders (how does the model see this molecular configuration?)

In each case, the LLM's interpretation reveals the source system's learned ontology — the categories, distinctions, and relationships it has discovered — in human-readable natural language.
