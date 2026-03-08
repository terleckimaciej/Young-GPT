# Young-GTP
### From Scratch to Pretrained: A Study in Training Small Language Models

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat-square&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=flat-square&logo=huggingface&logoColor=white) ![Google Colab](https://img.shields.io/badge/Google%20Colab-Computational%20Power-F9AB00?style=flat-square&logo=googlecolab&logoColor=white) ![OpenAI](https://img.shields.io/badge/OpenAI-Tokenizer-green?style=flat-square&logo=openai&logoColor=white)

## Introduction
**Young-GTP** is a project focused on building a simple Generative Pre-trained Transformer (GPT) from scratch, featuring a manual implementation of the **self-attention** mechanism. The experiment utilizes a self-collected dataset containing the complete discography of Polish rapper **Tede**, concatenated into a single text file.

The primary objective is to investigate whether a model built entirely from the ground up can learn to generate text resembling the Polish language in a song-like manner. Since the model starts with no prior knowledge and operates on individual characters, it must effectively learn how to assemble letters into valid words and syntax. 

Next, we experiment with leveraging different tokenizers by shifting from character-level to BPE-level encoding to understand the challenges that arise regarding data, computation, model size, and pre-training. This provides key insights into how LLMs "learn to speak" and what was required to evolve the self-attention mechanism (introduced in the 2017 [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) paper) into the ChatGPT-like models we take for granted today.

These baseline results are subsequently compared against two evolved approaches:
1.  A model utilizing BPE (Byte Pair Encoding) tokenizer to process sub-word units instead of characters. Multiple applied approaches reveal its challenges and properties.
2.  A **pre-trained model** where existing weights are recalibrated (fine-tuned) on the same Tede dataset, leveraging Transfer Learning. The idea is to try to overcome the dataset and computation limitations of the from-scratch-BPE model.

## The Dataset
The models are trained on a corpus comprising the entire discography of Polish rapper **Tede**. It consists of ~1M characters.
The data was self-acquired via the Genius API. Users can seamlessly collect an equivalent dataset for their artist of choice by changing the `ARTIST_NAME` parameter and running the [`data_collection.py`](data_collection.py) script. 

## The Three Approaches: Comparative Results

This project represents an evolution of understanding—from learning the shapes of letters to mastering the style of a specific artist. The following breakdown illustrates the specific goals, trade-offs, and results of each architectural decision.

> **Note:** The text samples below are short excerpts. You are encouraged to check the full generated output files, which are linked in each section under **Output Example**.

### Stage 1: From Scratch (The Visual Mimic)
*   **Model:** [`gpt.py`](gpt.py)
*   **Architecture:** Custom Transformer (Pytorch)
*   **Tokenization:** Character-level (~100 unique tokens)
*   **Concept & Goal:** The model starts with zero knowledge of language. It reads text character-by-character, aiming to learn how to assemble letters into valid words and syntax purely from statistical probability.

**Output Example:** ([`assets/char_model/output/output1.txt`](assets/char_model/output/output1.txt))
```text
Chłopaki z klałki, taki za mną
Lalk dawno punkt ci, towarli mno
Patrzę nowa w nas weekend wapno
A llub Cicho, liczą się za mną...
```

**Observations:**
*   **Visual Structure:** This model processes text almost like an image. It perfectly mimics the *visual density* of lyrics—short, punchy lines, frequent line breaks, and consistent stanza groupings (often resembling a "4-bar loop").
*   **Phonetic Hallucination:** It synthesizes non-existent but phonetically plausible words (*"klałki"*, *"towarli"*). It implicitly learned that the average Polish word is 2-3 syllables long, creating a jagged but physically pronounceable rhythm.
*   **"Child-like" Rhymes:** The rhyming strategy is purely morphological (suffix matching). It sees that lines often end with similar characters even if it doesn't understand the words (e.g., matching *za mną* with *wapno*). It’s like a child equating verbs with verbs.
*   **The "Density" Advantage:** With ~1M characters and only ~100 tokens (letters), the model has **~10,000 examples per token**. This massive signal density allows it to master local dependencies (spelling, line endings) far better than the BPE models below.
*   **Problem:** It has mastering the *texture* of the language but has zero semantic understanding. To fix this, we need to move from characters to words.

---

### Stage 2: The Abstraction (The Data Density Problem)
*   **Model:** [`gpt_tiktoken.py`](gpt_tiktoken.py)
*   **Architecture:** Custom Transformer
*   **Tokenization:** Byte Pair Encoding (BPE)
*   **Concept & Goal:** We shift from predicting letters to predicting "tokens" (sub-words or whole words). The goal is to allow the model to process information more efficiently and produce coherent sentences by taking "words for granted."

**A. Standard OpenAI Tokenizer (50k vocab)**
*   **Why this choice:** We began with the industry standard (`gpt2` tokenizer) to establish a baseline. As the default tokenization method for many modern LLMs, it was the natural first step to see if simply adopting "professional" tokenization could elevate the model from spelling letters to forming sentences.
*   **Output Example:** ([`assets/tiktoken_model/output/output2.txt`](assets/tiktoken_model/output/output2.txt))
    ```text
    Pij robię swój w to i rap grzejmij
    Ona ma, rozprowadz mi, jest mi i prawdzmat, a ja jestem
    I widzę jestem dziś mówię to za rzeczy zapórz
    ```
*   **Observations:** 
    *   **The "Sparsity" Crash:** By increasing the vocabulary to 50,000 tokens while keeping the dataset small (~1MB), the average **examples per token dropped drastically**. The model has too many "slots" to fill and not enough data to learn the relationships between them.
    *   **Word Salad:** It jumps between topics every token. Real words exist (*"robię"*, *"jestem"*), but grammatical connection is lost.
    *   **Fragmentation:** Because the OpenAI tokenizer is universal (not Polish-optimized), it often splits Polish words into unintuitive sub-tokens (thanks to polish grammar), occasionally creating invalid neologisms (*"prawdzmat"*) when those sub-tokens are reassembled incorrectly.

**B. Custom Small-Vocab Tokenizer (2-5k vocab)**
*   **Why this choice:** To combat the "sparsity crash," we trained a custom tokenizer specifically on our dataset with a drastically reduced vocabulary (2-5k tokens). The hypothesis was that eliminating unused English tokens and shrinking the dictionary would ensure each token appears frequently enough for the model to learn it.
*   **Output Example:** ([`assets/tiktoken_model/output/output4.txt`](assets/tiktoken_model/output/output4.txt))
    ```text
    ka chęć twoje ruchają tam 1600! Tak to lecę ogólnie go zamimieli spe ały czas
    ```
*   **Observations:** 
    *   **The Paradox of Compression:** In BPE, a smaller vocabulary forces the tokenizer to be more aggressive. It can only "afford" to store the most frequent sub-units. Since full words (like *"samochód"*) are rarer than their building blocks (letters/syllables like *"sa"*, *"mo"*), the tokenizer discards full words to save space.
    *   **Regression to Characters:** Paradoxically, by trying to simplify the vocabulary, we forced the model back towards character-level processing. The token stream became a sequence of syllables (`ka`, `chę`, `ć`), requiring the model to relearn how to glue them together—a task it failed at due to the lack of dense training data.
    *   **Un-learning Spelling:** The model was effectively forced back to "Stage 1" (learning to spell), but with worse resolution than characters. Detailed words like *"specjalnie"* became sequences like `s` `pe` `cjal` `nie`, which the model failed to reassemble consistently.

**C. Polish-Specific Tokenizer**
*   **Why this choice:** Both previous attempts failed at "word integrity." The standard tokenizer chopped Polish words because it didn't know them; the custom tokenizer chopped them because it had no room for them. We concluded that, while sticking to the "from scratch" approach and using the same dataset, one more avenue to explore was using a tokenizer specifically trained on Polish corpora to see if it handles the task better than a universal one.
*   **Output Example:** ([`assets/tiktoken_model/output/output6.txt`](assets/tiktoken_model/output/output6.txt))
    ```text
    Mam pierwszy patrz, na płyty uwieja w tejjedno baj noga zobowiązani, przekaz
    Gdzie możesz mieć podwójewska Operacyjnego to kwestia rachunek marnkę
    ```
*   **Observations:** 
    *   **Structural Rigidity:** This tokenizer successfully kept complex Polish words intact (*"zobowiązani"*, *"operacyjnego"*), and they seem to occur more frequently than when using OpenAI's universal tokenizer. 
    *   **Lack of Flow:** While the words are mostly valid, the model still lacks the data volume to weave them into a flow. The syntax is rigid, resembling a list of dictionary words rather than a song.
    *   **Conclusion:** We cannot solve the data-density problem by just changing the tokenizer. To get semantic meaning and style we need a massive dataset... or a model that *already knows* Polish.

**Conclusion: BPE Failure:**
Ultimately, shifting to BPE (regardless of the specific tokenizer) failed to produce a better "song writer" than the simple character-level model. We faced a wall that engineering couldn't climb: **Data Volume**.
*   **Char-Level:** ~100 tokens, 1M characters dataset = High density learning (model sees each token thousands of times).
*   **BPE-Level:** ~2k-50k tokens, 1M characters dataset = Extreme sparsity (model sees many tokens only once or twice).
To make BPE work effectively, we would need a dataset 100x larger (hundreds of megabytes, not 1MB). Since we cannot generate more songs by the artist, the only way to scale up without changing the dataset is to change the brain: **Transfer Learning**. We need a model that *already* read the entire Polish internet, so we only have to teach it *style* and *rhyme*, not the entire language from scratch.
---

### Stage 3: Transfer Learning (The Style Transfer)
*   **Model:** [`gpt_hf.py`](gpt_hf.py)
*   **Architecture:** Pre-trained GPT-2 (`flax-community/papuGaPT2`)
*   **Tokenization:** Pre-trained GPT-2 Tokenizer
*   **Concept & Goal:** Leverage a model pre-trained on massive Polish corpora (Wikipedia, etc.) to handle grammar and meaning, then fine-tune it to "un-learn" its formal tone and adopt the artist's specific style (slang, flow, entities).

**Why this choice:**
We hit a wall with data volume. Our 1MB dataset was enough to learn letters (Stage 1) but too small to learn a language (Stage 2). 
Instead of teaching a baby to speak from scratch using only rap lyrics, we take an educated adult (a pre-trained model that already knows Polish grammar, vocabulary) and "taught it to rap." 
This way, the model uses our small dataset **solely to learn style, slang, flow and form of a lyric**.


**Output Example (Early Epochs):** ([`assets/hf_model/output/output1.txt`](assets/hf_model/output/output1.txt))
*   **Context:** We first tried running just a few epochs (4) to see the early result.
```text
Ej joł, ziom
Jestem Tuzin Gibka co to za mafia ta i inne bauns'y na eBayu! Buhhh- buuuuhahaha to nie teges z tej strony pozdrawiam ciebie koleś ze stalowowolskiego osiedla ten koleżka wiesz tak mnie znają wiem dawno ich poznałem...
```
**Observations:**
*   **Wiki-Bias (Prose vs. Verse):** At this early stage, the model speaks in prose. Since its base training comes from encyclopedic texts (Wikipedia), it hasn't yet learned the concept of a "verse" or "line break," producing a continuous block of text instead.
*   **The Prompt Effect:** We prompted the model with *"Ej joł"* (a slang term frequent in Tede's lyrics but absent in Wikipedia). Interestingly, the model generated a single line break immediately after this prompt (likely recognizing the token from fine-tuning) — **a behavior consistent across all 3 generated samples** — but then immediately reverted to its "default" prose mode for the rest of the output.
*   **Conclusion:** 4 epochs are not enough to overwrite the strong "encyclopedic memory" of the base model. To force it to adopt the visual structure of a song, we need significantly more training steps (overfitting) to "break" its formal habits.

**Output Example (Overfitted - "The Stylist"):** ([`assets/hf_model/output/output2.txt`](assets/hf_model/output/output2.txt))
*   **Context:** Here's the effect after 50 epochs. The loss also fell drastically from 3.36 to 0.02.
```text
Wiecie rap ma przede mną szerokie pole do popisu
Wiem kto jest Kto lepszy od kogo o kim aktualnie piszę
Do pokoju wchodzi fajna dziewczyna z działu panowie
Dziewczyny mi nie urwą więc monitor mam wypchnięty
Kolesie są nieco umęczeni wczorajszym wieczorem
Panowie co ich hartuje to co słabe to co co dobre
Z klubu wychodzi facet w białym płaszczyku popite już Drinkiem
Drinki mu dają po dupie pije dalej przez tydzień
```

**Observations:**
*   **Semantic Coherence:** Unlike previous models, these sentences have logical meaning. The model describes a scene (club, people, evening) rather than just vomiting words, but shifts between them chaotically like a dream. Although it occasionally loses context, this model links concepts across lines. Line 7 ends with *"Drinkiem"*, and Line 8 immediately picks up with *"Drinki"* (relating back to the person from the previous line), demonstrating an emerging ability to maintain context.
*   **The "Hip-Hop Cadence":** 
    It manipulates larger rhythmic units than the char-model:
    * **Slant rhyme** (assonance) of *Drinkiem* (drink-yem) vs *Tydzień* (ti-jen). This is a sophisticated imperfect rhyme which a character model looking for exact text matches would never attempt.
    * **Consonance:** *Popisu* vs *Piszę*. Though the endings differ, the phonetic skeleton (P-P-S vs P-Sz) is nearly identical.
    * **Alliteration:** *"Wiem **k**to jest **K**to lepszy od **k**ogo o **k**im..."* — a flow technique used to build rhythm within a line.
*   **Style Transfer & Overfitting:**
    *   Initially (output1), the model suffered from "Wiki-Bias", struggling to stop being formal.
    *   By aggressively **overfitting** (50 epochs) with high weight decay, we forced a complete persona switch. The model successfully adopted entities, effectively "forgetting" its encyclopedic nature to become a rapper.
*   **Trade-off (Visuals):** While semantically superior, this model struggles slightly with the *visual discipline* of the char-model. It sometimes drifts into long, run-on sentences or prose-like structures ("stream of consciousness") because its base training (Wikipedia) wasn't formatted as poetry. It prioritizes *saying something* over *looking like a song*.

## Conclusions
The project highlights the critical relationship between **vocabulary size** and **dataset volume** in training Language Models.

1.  **Form over Content (Char-level):** The character-level model (Stage 1) demonstrated that with a small dataset, it is easier to mimic the *visual and phonetic structure* of language (rhymes, line breaks) than its meaning. The high frequency of character occurrence allowed the model to master the "texture" of the text despite having zero semantic understanding.

2.  **The Curse of Dimensionality (BPE):** Attempts to introduce semantic awareness via BPE tokenization (Stage 2) failed not due to architecture, but due to **data sparsity**. Increasing the vocabulary size from ~100 (chars) to ~50k (tokens) without increasing the dataset size diluted the signal, preventing the model from learning distinct token embeddings. This confirmed that **model complexity cannot outpace available data**.

3.  **The Power of Transfer Learning:** The final approach (Stage 3) proved that for niche tasks with limited data, **fine-tuning is indispensable**. By leveraging a model that already understood Polish grammar (PapuGaPT2), we could repurpose the small dataset from "learning to speak" to "learning a style." This resulted in the only model capable of generating coherently structured, stylistically accurate lyrics that maintained semantic context.
## Project Structure

The repository is organized to separate source code, data, and trained model artifacts.

```plaintext
Mlody-GTP/
├── data_collection.py       # Script to fetch lyrics from Genius API
├── gpt.py                   # Stage 1: Character-level GPT implementation (from scratch)
├── gpt_tiktoken.py          # Stage 2: BPE-level GPT implementation (from scratch)
├── gpt_hf.py                # Stage 3: Fine-tuning script for pre-trained Hugging Face models
├── train_tokenizer.py       # Utility to train custom BPE tokenizers on the dataset
├── prompt.txt               # Sample prompts for testing generation
├── requirements.txt         # Python dependencies
│
└── assets/                  # Main directory for all data and artifacts
    ├── input/               # Raw training datasets (text files).
    │   └── input2.txt       # The primary dataset used (Tede discography)
    │
    ├── char_model/          # Stage 1 artifacts
    │   ├── model_best.pt    # Saved weights for the character-level model
    │   └── output/          # Generated text samples and training parameters
    │
    ├── tiktoken_model/      # Stage 2 artifacts
    │   ├── polish_gpt2/     # Custom trained Polish BPE tokenizer files
    │   ├── model_best.pt    # Saved weights for BPE models
    │   └── output/          # Results for different BPE tokenizer approaches 
    │
    └── hf_model/            # Stage 3 artifacts (Fine-tuned PapuGaPT2)
        ├── model.safetensors # Fine-tuned model weights
        ├── tokenizer.json   # Tokenizer configuration
        └── output/          # Samples from different training epochs (4 vs 50) demonstrating the progression
```

## Usage

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Experiments

**Level 1: Train & Generate (Character-Level)**
```bash
python gpt.py --input assets/input/input2.txt \
              --batch_size 64 \
              --block_size 256 \
              --max_iters 5000 \
              --learning_rate 3e-4 \
              --output assets/char_model/output.txt
```

**Level 2: Train & Generate (BPE / Tiktoken)**
```bash
# pick a --tokenizer yourself
python gpt_tiktoken.py --tokenizer tiktoken \
                       --input assets/input/input2.txt \
                       --max_iters 2000 \
                       --batch_size 32
```

**Level 3: Fine-Tune (Hugging Face / PapuGaPT2)**
```bash
python gpt_hf.py --input assets/input/input2.txt \
                 --model_name flax-community/papuGaPT2 \
                 --epochs 50 \
                 --block_size 256 \
                 --learning_rate 5e-5 \
                 --prompt "Ej joł"
```
## Acknowledgements
The architecture of the base character-level model (`gpt.py`) is derived from the model built during **Andrej Karpathy's** [Let's build GPT: from scratch, in code, spelled out](https://github.com/karpathy/nn-zero-to-hero) lecture, with additional modifications implemented for this project. The following models were developed by myself.

The pre-trained model used for the Hugging Face transfer learning experiment is [**PapuGaPT2**](https://huggingface.co/flax-community/papuGaPT2), developed by the **Flax Community**.

I would also like to thank my university lecturer, dr. **Maciej Switała**, for the opportunity to discuss the project results. 