# Mlody-GTP
### From Scratch to Pretrained: A Study in Training Small Language Models

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat-square&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=flat-square&logo=huggingface&logoColor=white) ![Google Colab](https://img.shields.io/badge/Google%20Colab-Computational%20Power-F9AB00?style=flat-square&logo=googlecolab&logoColor=white) ![OpenAI](https://img.shields.io/badge/OpenAI-Tokenizer-green?style=flat-square&logo=openai&logoColor=white)

## Introduction
**Mlody-GTP** is a project focused on building a simple Generative Pre-trained Transformer (GPT) from scratch, featuring a manual implementation of the **self-attention** mechanism. The experiment utilizes a self-collected dataset containing the complete discography of Polish rapper **Tede**, concatenated into a single text file.

The primary objective is to investigate whether a model built entirely from the ground up can learn to generate text resembling the Polish language in a song-like manner. Since the model starts with no prior knowledge and operates on individual characters, it must effectively learn how to assemble letters into valid words and syntax. 

Next, we experiment with leveraging different tokenizers by shifting from character-level to BPE-level encoding to understand the challenges that arise regarding data, computation, model size, and pre-training. This provides key insights into how LLMs "learn to speak" and what was required to evolve the self-attention mechanism (introduced in the 2017 [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) paper) into the ChatGPT-like models we take for granted today.

These baseline results are subsequently compared against two evolved approaches:
1.  A model utilizing BPE (Byte Pair Encoding) tokenizer to process sub-word units instead of characters. Multiple applied approaches reveal it's challenges and properties.
2.  A **pre-trained model** where existing weights are recalibrated (fine-tuned) on the same Tede dataset, leveraging Transfer Learning. The idea is to try to overcome dataset and computation limitations of from-scratch-BPE model.

## The Dataset
The models are trained on a corpus of polish rapper **Tede** entire discography lyrics. It consists of ~1M characters.
The data was self acquired via Genius API, one can seamlessly collect its artist of choice eqivalent dataset by changing the ARTIST_NAME parameter and running the [`data_collection.py`](data_collection.py) script. 

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
*   **Why this choice:** Both previous attempts failed at "word integrity." The standard tokenizer chopped Polish words because it didn't know them; the custom tokenizer chopped them because it had no room for them. We concluded that while sticking to "from scratch" approach and using the same dataset one more thing to try is using tokenizer specifically trained on polish corpora to see if it handles the task better than an universal one. 
*   **Output Example:** ([`assets/tiktoken_model/output/output6.txt`](assets/tiktoken_model/output/output6.txt))
    ```text
    Mam pierwszy patrz, na płyty uwieja w tejjedno baj noga zobowiązani, przekaz
    Gdzie możesz mieć podwójewska Operacyjnego to kwestia rachunek marnkę
    ```
*   **Observations:** 
    *   **Structural Stiff:** This tokenizer successfully kept complex Polish words entire (*"zobowiązani"*, *"operacyjnego"*), they seem to occur more frequently than when using OpenAI's universal tokenizer. 
    *   **Lack of Flow:** While the words are mostly valid, the model still lacks the data volume to weave them into a flow. The syntax is rigid, resembling a list of dictionary words rather than a song.
    *   **Conclusion:** We cannot solve the data-density problem by just changing the tokenizer. To get semantic meaning and style we need a massive dataset... or a model that *already knows* Polish.

**Conclusion: BPE Failure:**
Ultimately, shifting to BPE (regardless of the specific tokenizer) failed to produce a better "song writer" than the simple character-level model. We faced a wall that engineering couldn't climb: **Data Volume**.
*   **Char-Level:** ~100 tokens, 1M characters dataset = High density learning (model sees each token thousands of times).
*   **BPE-Level:** ~2k-50k tokens, 1M characters dataset = Extreme sparsity (model sees many tokens only once or twice).
To make BPE work effectively, we would need a dataset 100x larger (hundreds of megabytes, not 1MB). Since we cannot generate more songs by the artist, the only way to scale up without changing the dataset is to change the brain: **Transfer Learning**. We need a model that *already* read the entire Polish internet, so we only have to teach it *style* and *rhyme*, not the entire language from scratch.
---

TO DO: w ponizszej sekcji oprocz poprawek dodac opis 1st run.
### Stage 3: Transfer Learning (The Style Transfer)
*   **Model:** [`gpt_hf.py`](gpt_hf.py)
*   **Architecture:** Pre-trained GPT-2 (`flax-community/papuGaPT2`)
*   **Tokenization:** Pre-trained GPT-2 Tokenizer
*   **Concept & Goal:** Leverage a model pre-trained on massive Polish corpora (Wikipedia, etc.) to handle grammar and meaning, then fine-tune it to "un-learn" its formal tone and adopt the artist's specific style (slang, flow, entities).

**Output Example:** ([`assets/hf_model/output/output2.txt`](assets/hf_model/output/output2.txt))
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
*   **Semantic Coherence:** Unlike previous models, these sentences have logical meaning. The model describes a scene (club, people, evening) rather than just vomiting words.
*   **The "Hip-Hop Cadence":** It manipulates larger rhythmic units than the char-model. Notice the **slant rhyme** (assonance) of *Drinkiem* (drink-yem) vs *Tydzień* (ti-jen). This is a sophisticated imperfect rhyme typical of modern rap flow, which a character model looking for exact text matches would never attempt.
*   **Style Transfer & Overfitting:**
    *   Initially (output1), the model suffered from "Wiki-Bias", struggling to stop being formal.
    *   By aggressively **overfitting** (50 epochs) with high weight decay, we forced a complete persona switch. The model successfully adopted specific slang (*"bauns"*, *"grrryatem"*) and entities, effectively "forgetting" its encyclopedic nature to become a rapper.
*   **Trade-off (Visuals):** While semantically superior, this model struggles slightly with the *visual discipline* of the char-model. It sometimes drifts into long, run-on sentences or prose-like structures ("stream of consciousness") because its base training (Wikipedia) wasn't formatted as poetry. It prioritizes *saying something* over *looking like a song*.

## Usage

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Experiments

**Level 1: Train & Generate (Character-Level)**
```bash
python gpt.py --input assets/input/input2.txt --max_iters 5000
```

**Level 2: Train & Generate (Tiktoken)**
```bash
python gpt_tiktoken.py --input assets/input/input2.txt --max_iters 5000
```

**Level 3: Fine-Tune (Hugging Face)**
```bash
python gpt_hf.py --input assets/input/input2.txt --epochs 3
```
TO DO: dodac contributions dla Karpathy'ego