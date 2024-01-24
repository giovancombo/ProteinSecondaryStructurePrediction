# DL2023 | Using Transformers for Protein Secondary Structure Prediction

https://en.wikipedia.org/wiki/Protein_secondary_structure?wprov=sfla1

**Protein Structure Prediction** (PSP) has always been one of the most popular applications of Deep Learning, and one of the most important fields of application of bioinformatics and computational biology, as it is useful for *drug design* and *novel enzymes design*.

Historical methods for determining/predicting the Secondary Structure of a protein from its Primary Structure were the **Chou-Fasman** method, the **GOR** method, then replaced by algorithms like **DEFINE**, **STRIDE**, **ScrewFit**, **SST**, and **DSSP**, which formally established a dictionary of all the currently known types of Secondary Structure.

Algorithms for *predicting* Secondary Structure are **PSIpred**, **SAM**, **PORTER**, **PROF** and **SABLE**.

**Guarda meglio gli altri algoritmi da menzionare**

**!! Nota: Crea una tabella "benchmark" con le accuracies dei vari algoritmi storici, e poi più avanti anche una tabella benchmark riguardo i modelli usati sul dataset specifico CullPDB, per fare un confronto.**

## Background and Related Works
This work focuses on replicating part of the results achieved in the following paper:
> [(1) Zhou, J., & Troyanskaya, O. G., Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction, 2014](https://arxiv.org/abs/1403.1347)

Instead of using a ConvNet, though, a Transformer will be implemented, following the architecture presented in the original paper, [*Attention Is All You Need* (2)](https://arxiv.org/abs/1706.03762).

The Primary Structure of a protein can be seen as a sequence of **characters** (instead of words) drawn from a **vocabulary of size 20**. "Translating" from the Primary to the Secondary Structure means converting the input sequence to another sequence of **characters** drawn from a **vocabulary of size 8** (i.e. the 8 possible classes of Secondary Structure), or **size 3**.
The size of the vocabulary only depends on the level of prediction accuracy we want to achieve, as the size-3-vocabulary merges the 8 classes into 3 macroclasses. Thus, prediction on the 3-classes problem is easier.

### Some biological facts
Proteins are **chains of proteinogenic amino acids**.

Amino acids are, basically, **biomolecules**. In nature, there are 20 genetically encoded, "standard" types of them, plus 2 "special" ones:

|Abbr|Amino Acid|Frequency in Proteins|
|:-:|:-:|:-:|
| A | *Alanine*| **8,76%**|
| C | *Cysteine*| *1,38%*|
| D | *Aspartic Acid*| 5,49%|
| E | *Glutamic Acid*| 6,32%|
| F | *Phenylalanine*| 3,87%|
| G | *Glycine*| 7,03%|
| H | *Histidine*| 2,26%|
| I | *Isoleucine*| 5,49%|
| K | *Lysine*| 5,19%|
| L | *Leucine*| **9,68%**|
| M | *Methionine*| 2,32%|
| N | *Asparagine*| 3,93%|
| O | *Pyrrolisine*| special|
| P | *Proline*| 5,02%|
| Q | *Glutamine*| 3,90%|
| R | *Arginine*| 5,78%|
| S | *Serine*| 7,14%|
| T | *Threonine*| 5,53%|
| U | *Selenocysteine*| special|
| V | *Valine*| 6,73%|
| W | *Tryptophan*| *1,25%*|
| Y | *Tyrosine*| 2,91%|

**Note:** The 2 special amino acids (O - Selenocysteine and U - Pyrrolysine) are not present in the datasets used for this exercise.

The raw chain of amino acids linked together via the amminic-carbossilic groups determines the **Primary Structure** of a protein.

---

The **Secondary Structure** is the local spatial conformation of the sequence of amino acids. It is determined according to the way in which amino acids link together through hydrogen bonds. This means that **every single amino acid in the protein sequence is linked to another of the same sequence**, building what it could be interpreted as an end-to-end relationship.

There are **8** types of Secondary Structure for a protein:

| Type     | DSSP Abbr  |
| :----: | :----: |
| α-helix (4-helix) | 'H' |
| β-strand | 'E' |
| loop / irregular (coil) | 'L' or 'C' |
| β-turn | 'T' |
| bend | 'S' |
| $3_10$-helix (3-helix) | 'G' |
| β-bridge | 'B' |
| π-helix (5-helix) | 'I' |

It's curious to note that specific amino acids are more prone to be found in specific types of structure:
+ M, A, L, E, K prefer *helices*
+ F, I, T, V, W, Y prefer *strands*
+ G, P are known as *helix breakers*

**!! AGGIUNGERE ALTRA TEORIA RIGUARDO SPECIFICI TIPI DI STRUTTURA**

### Some insights about Data
*Position-Specific Scoring Matrix* (PSSM) values can be interpreted as **word vectors** for the input characters. As these values are not obvious to calculate, I will use *CullPDB* and *CB513* datasets, provided by the authors of the [(1)](https://arxiv.org/abs/1403.1347) paper (available at [this link](https://www.princeton.edu/~jzthree/datasets/ICML2014)), which contain a bunch of ready-to-use PSSMs.

This dataset was originally hosted at This link. Since the original URL is no longer available and the dataset is still used by many, the dataset has been moved [here](https://zenodo.org/records/7764556#.ZByi1ezMJvI) or mirrored [here](https://mega.nz/folder/xct0XSpA#SKz72JtnSAaX61QLMC_JNg).
I actually downloaded the datasets from chinese Baidu AI Studio platform (https://aistudio.baidu.com/datasetdetail/79771) because of the Princeton link not working.

Proteins that show extra characters not included in the Primary Structure Vocabulary (e.g. 'X', indicating an unknown amino acid) are removed from the datasets.

Some proteins are composed of a very long sequence of amino acids: this could bring some computational problems, due to the $O(n^2)$ complexity of the Transformer, hence these particular proteins are divided in many shorter fragments. The longest proteins are segmented into blocks of length $N <<$ n_max, with N chosen accordingly with the available computational resources (at least, $N > 30$).

As the length of most protein sequences is less than 700, the one-hot coding of residue sequences and the size of the PSSM are generally unified into 700 × 21. That is, the sequences whose length is greater than 700 will be divided into two overlapping sequences, while the sequences whose length is less than 700 will be augmented by filling in zeros. Thus, the input feature of the prediction model is a matrix with the size of 700 × 42.

### Some insights about the Architecture
One thing to note is that PSSMs are *NOT* absolute embedding values: they are **relative** to their mutual positions and in the protein sequence. Thus, while implementing the Transformer architecture, it can result more efficient to prefer a **relative positional encoding** strategy (as explained in [this (3)](https://arxiv.org/abs/1803.02155) paper) over the absolute one employed in the [(2)](https://arxiv.org/abs/1706.03762) paper.

Encoder-Decoder Transformers were originally presented in [(2)](https://arxiv.org/abs/1706.03762) as an Attention-based tool for Sequence-to-Sequence tasks such as Machine Translation.

But the task addressed here is actually easier, as it is a **Sequence Classification** task: at every input *character* (from the Primary Structure Vocabulary, of size 20) corresponds **one** output *character* (from the Secondary Structure Vocabulary, of size 8). In fact, with no surprises, this task can be solved using a ConvNet.
This allows to guess that I could implement only an Encoder Transformer, getting rid of the Decoder, and replacing it with a simple classification head (e.g. an MLP) over every vector from the last layer of the remaining Transformer.

---

**COSE TEORICHE SU OGNI CLASSE DI SS**



---

**COSE AGGIUNTE - DA INTEGRARE ALLE COSE SOPRA**

https://github.com/amckenna41/DCBLSTM_PSP
The training datasets used in this project are taken from the ICML 2014 Deep Supervised and Convolutional Generative Stochastic Network paper [1]. The datasets in this paper were created using the PISCES protein culling server, that is used to cull protein sequences from the protein data bank (PDB) [14]. As of Oct 2018, an updated dataset, with any of the duplicated in the previous __6133 dataset removed, has been release called cullpdb+profile_5926. Both of the datasets contain a filtered and unfiltered version. The filtered version is filtered to remove any redundancy with the CB513 test dataset. The unfiltered datasets have the train/test/val split whilst for filtered, all proteins can be used for training and test on CB513 dataset. Both filtered and unfiltered dataset were trained and evaluated on the models.

The publicly available training dataset used in our models was the CullPDB dataset – CullPDB 6133. The dataset was produced by PISCES CullPDB [2] and contains 6128 protein sequences, their constituent amino acids and the associated 57 features of each residue. The first 22 features are the amino acid residues, followed by a ‘NoSeq’ which just marks the end of the protein sequence. The next 9 features are the secondary structure labels (L, B, E, G, I, H, S, T), similarly followed by a ‘NoSeq’. The next 4 features are the N and C terminals, followed by the relative and absolute solvent accessibility. The relative accessibility is normalized by the largest accessibility value in a protein and thresholded at 0.15 and the absolute accessibility is thresholded at 15. The final 22 features represent the protein sequence profile. The secondary structure label features and relative and solvent accessibilities are hidden during testing. There exists a filtered and unfiltered version of this dataset, with the filtered data having redundancy with the CB513 test dataset removed.
--> DATASET USED = CullPDB 6133 FILTERED

Test: Three datasets were used for evaluating the models created throughout this project:
+ CB513
+ CASP10
+ CASP11
The CB513 dataset is available at: https://www.princeton.edu/~jzthree/datasets/ICML2014/


https://github.com/LucaAngioloni/ProteinSecondaryStructure-CNN?tab=readme-ov-file
Instead of using the primary structure as a simple indicator for the presence of one of the amino acids, a more powerful primary structure representation has been used: Protein Profiles. These are used to take into account evolutionary neighborhoods and are used to model protein families and domains. They are built by converting multiple sequence alignments into position-specific scoring systems (PSSMs). Amino acids at each position in the alignment are scored according to the frequency with which they occur at that position.
A protein’s polypeptide chain typically consist of around 200-300 amino acids, but it can consist of far less or far more. The amino acids can occure at any position in a chain, meaning that even for a chain consisting of 4 amino acids, there are 204 possible distinct combinations. In the used dataset the average protein chain consists of 208 amino acids.

Proteins’ secondary structure determines structural states of local segments of amino acid residues in the protein. The alpha-helix state for instance forms a coiled up shape and the beta-strand forms a zig-zag like shape etc. The secondary structure of the protein is interesting because it, as mentioned in the introduction, reveals important chemical properties of the protein and because it can be used for further predicting it’s tertiary structure. When predicting protein's secondary structure we distinguish between 3-state SS prediction and 8-state SS prediction.

For 3-state prediction the goal is to classify each amino acid into either:

alpha-helix, which is a regular state denoted by an ’H’.
beta-strand, which is a regular state denoted by an ’E’.
coil region, which is an irregular state denoted by a ’C’.
The letters which denotes the above secondary structures are not to be confused with those which denotes the amino acids.

For 8-state prediction, Alpha-helix is further sub-divided into three states: alpha-helix (’H’), 310 helix (’G’) and pi-helix (’I’). Beta-strand is sub-divided into: beta-strand (’E’) and beta-bride (’B’) and coil region is sub-divided into: high curvature loop (’S’), beta-turn (’T’) and irregular (’L’).

The dataset used is CullPDB data set, consisting of 6133 proteins each of 39900 features. The 6133 proteins × 39900 features can be reshaped into 6133 proteins × 700 amino acids × 57 features.

The amino acid chains are described by a 700 × 57 matrix to keep the data size consistent. The 700 denotes the peptide chain and the 57 denotes the number of features in each amino acid. When the end of a chain is reached the rest of the vector will simply be labeled as ’No Seq’ (a padding is applied).

Among the 57 features, 22 represent the primary structure (20 amino acids, 1 unknown or any amino acid, 1 'No Seq' -padding-), 22 the Protein Profiles (same as primary structure) and 9 are the secondary structure (8 possible states, 1 'No Seq' -padding-).

The Protein profiles where used instead of the amino acids residues.

In a first phase of research the whole amino acid sequence was used as an example (700 x 22) to predict the whole secondary structure (label) (700 x 9).

In the second phase, local windows of a limited number of elements, shifted along the sequence, were used as examples (cnn_width x 21) to predict the secondary structure (8 classes) in a single location in the center of each window (The 'No Seq' and padding were removed and ignored in this phase because it wasn't necessary anymore for the sequences to be of the same length).

The Dataset (of 6133 proteins) was divided randomly into training (5600), validation (256) and testing (272) sets.

The dataset is currently in numpy format as a (N protein x k features) matrix. You can reshape it to (N protein x 700 amino acids x 57 features) first. 

The 57 features are:
[0,22): amino acid residues, with the order of 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq'
[22,31): Secondary structure labels, with the sequence of 'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'
[31,33): N- and C- terminals;
[33,35): relative and absolute solvent accessibility, used only for training. (absolute accessibility is thresholded at 15; relative accessibility is normalized by the largest accessibility value in a protein and thresholded at 0.15; original solvent accessibility is computed by DSSP)
[35,57): sequence profile. Note the order of amino acid residues is ACDEFGHIKLMNPQRSTVWXY and it is different from the order for amino acid residues

The last feature of both amino acid residues and secondary structure labels just mark end of the protein sequence. 
[22,31) and [33,35) are hidden during testing.

---

A position weight matrix (PWM), also known as a position-specific weight matrix (PSWM) or position-specific scoring matrix (PSSM), is a commonly used representation of motifs (patterns) in biological sequences.

PWMs are often derived from a set of aligned sequences that are thought to be functionally related and have become an important part of many software tools for computational motif discovery.

---

The PSSMs were generated from PSI-BLAST profiles, which contain important evolution information.

The fundamental elements of the secondary structure of proteins are -helices, -sheets, coils, and turns. Some methods have been developed for defining various protein secondary structure elements from the atomic coordinates in the Protein Data Bank (PDB), such as DSSP,4 STRIDE,5 and DEFINE.6 According to DSSP, 8 types of protein secondary structure elements were classified and denoted by letters: H (-helix), E (extended -strand), G (310 helix), I (-helix), B (isolated -strand), T (turn), S (bend) and “_” (coil). The 8 classes are usually reduced to three states, helix (H), sheet (E), and coil (C) by different reduction methods.7 Thus, the secondary structure prediction can be analyzed as a typical three-state pattern recognition or classification problem, where the secondary structure class of a given amino acid residue in a protein is predicted based on its sequence features.

 each residue is coded as a 21-dimensional vector,
where the first 20 elements of the vector are the corresponding elements in PSI-BLAST matrix.

---

Using a DSSP algorithm [14], the three general states were transformed and expanded into eight fine-grained states. They are 310 helix (G), α-helix (H), π-helix (I), β-strand (E), bridge (B), turn (T), high curvature loop (S), and others (L).

Prediction accuracy is
greatly enhanced by the inclusion of sequence evolutionary
profiles obtained from Multiple Sequence Alignments
(MSA), known as Position Specific Scoring Matrices
(PSSM).

In DeepACLSTM [18], the benefits of the CNN and
LSTM networks are integrated. They use CNN to obtain the
protein's local features and BLSTM to get information on
long distance dependency. The model known as
MUFOLD-SS [19] is a Google Inception network [20] based
Deep3I network, also referred to as a deep inception-insideinception
network.The improvised model of MULFOLD-SS,
known as SAINT [21] integrates the self-attention
mechanism and Deep3I, and increases the effectiveness of
gathering long-distance information on contacts in
sequences.
The OPUS-TASS approach integrated the architectures
of CNN, BLSTM, and Transformer [22]. In the work
proposed in [23], the MLPRNN architecture consists of two
layer stacked BGRU(Bidirectional Gated Recurrent Unit)
block.

In both the proposed models, because an α-helix typically
has eleven residues and a β-strand typically has six, the
window size was selected to exceed the value of 11. Based
on the trade-off between performance and training, a window
size of 17 was chosen.

---

OPUS-TASS PAPER

Determining protein structure with experimental approaches such as
X-ray crystallography, Cryo-EM and NMR are usually timeconsuming.
To tackle this issue, many computational prediction
methods have been proposed.

In recent years, a new architecture named Transformer (Vaswani
et al., 2017) has been proposed, achieving better performance comparing
to the traditional bidirectional recurrent neural networks in
most Natural Language Processing (NLP) tasks. Transformer is a
self-attention model, which is able to capture the interactions between
the two items theoretically with arbitrary distance.

---

> PAPER SHUFFLENET_SS: Protein secondary structure prediction using a lightweight convolutional network and label distribution aware margin loss

3-state prediction:
+ Helices (H,G,I)
+ Strands (B,E)
+ Coils (S,T,L)

8-state prediction can provide more valuable local structure information, and is more challenging because eight states have an extremely imbalanced distribution in protein structures (metti tabella con numeri e percentuali sul dataset di frequenza di ogni classe).

When training a deep PSSP model, a specific number of protein chains needs to be randomly selected from the training set to form a minibatch. The length of protein chains in minibatches is usually not equal due to the variable length distribution of protein chains. Therefore, it is necessary to perform zero padding on shorter protein chains so that all protein chains in the minibatch have the same length.

The input of our network model includes two parts: a feature data tensor and a binary mask matrix. The shape of the feature data tensor is (N, C, L), where N represents the number of protein chains in a minibatch, C represents the length of the feature vector corresponding to each amino acid residue, and L represents the maximum protein chain length in a minibatch. The size of the binary mask matrix is (N, L), where 1 indicates a nonpadded position and 0 indicates a padded position.

In the current PSSP models, all secondary structure categories are treated equally during training without considering that their distribution is extremely imbalanced.

Considering this, we adopt the **label distribution aware margin loss**, that encourages rare classes to have larger margins to train the proposed network. This can enhance the network's ability to learn rare classes without sacrificing the network's ability to fit frequent classes.

When the standard cross-entropy loss without considering class imbalance is used for training, the rare classes can only obtain extremely low classification accuracy due to being overwhelmed by larger classes during training.

in addition to the LDAM loss, there are other recently introduced losses, such as the class-balanced loss [43], focal loss [44], seesaw loss [45], equalization loss [46] and logit adjustment loss [47], that can effectively address the imbalanced classification problem with a long-tail distribution. In particular, the application scenarios of these losses for classification problems usually assume that the training set is class imbalanced while the validation and test sets are class balanced. However, all datasets used in eight-state PSSP are class imbalanced. In practice, we observe that only the LDAM loss can improve the prediction performance of the protein secondary structure;

## Model Architecture

### Encoder

#### Attention Mechanism

#### Relative Positional Encoding

Since self-attention networks are inherently position agnostic, we need to explicitly encode frame positions in the model. The original paper used sinusoidal position encodings for this purpose.

While absolute positional encodings work reasonably well, there have also been efforts to exploit pairwise, relative positional information. In Self-Attention with Relative Position Representations, Shaw et al. introduced a way of using pairwise distances as a way of creating positional encodings.

There are a number of reasons why we might want to use relative positional encodings instead of absolute ones. First, using absolute positional information necessarily means that there is a limit to the number of tokens a model can process. Say a language model can only encode up to 1024 positions. This necessarily means that any sequence longer than 1024 tokens cannot be processed by the model. Using relative pairwise distances can more gracefully solve this problem, though not without limitations. Relative positional encodings can generalize to sequences of unseen lengths, since theoretically the only information it encodes is the relative pairwise distance between two tokens.

Relative positional information is supplied to the model on two levels: values and keys. This becomes apparent in the two modified self-attention equations.

instead of simply combining semantic embeddings with absolute positional ones, relative positional information is added to keys and values on the fly during attention calculation.

calculating relative positional encodings as introduced in Shaw et al. requires O(L^2 * D) memory due to the introduction of an additional relative positional encoding matrix. Here, L denotes the length of the sequence, and D, the hidden state dimension used by the model.

**The key concept is that Distance between two characters is more important than their absolute positions in the sequence**

### Classification Head

### Decoder

## Training and Results
Batch size = 4 **proteins**

### Ablation
Si può provare a ripetere il training aggiungendo i vettori one hot o togliendoli

## Conclusion

## References
For Data, Features, Architecture:
https://cs.rice.edu/~ogilvie/comp571/pssm/
+ https://arxiv.org/abs/1403.1347
+ https://arxiv.org/abs/1512.03385
+ https://arxiv.org/abs/1607.06450
+ https://arxiv.org/abs/1412.6980
+ https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
+ https://www.princeton.edu/~jzthree/datasets/ICML2014/
+ https://aistudio.baidu.com/datasetdetail/79771
+ https://arxiv.org/abs/1706.03762
+ https://en.wikipedia.org/wiki/Position_weight_matrix
+ https://github.com/amckenna41/DCBLSTM_PSP
+ https://github.com/LucaAngioloni/ProteinSecondaryStructure-CNN
+ https://www.sciencedirect.com/science/article/pii/S2001037022005062
+ https://www.frontiersin.org/articles/10.3389/fbioe.2022.901018/full
+ https://arxiv.org/abs/1702.03865
+ https://ieeexplore.ieee.org/document/10080387
+ https://www.baskent.edu.tr/~hogul/secondary.pdf

For Relative Positional Encoding:
+ https://arxiv.org/abs/1803.02155
+ https://arxiv.org/abs/1811.07143
+ https://arxiv.org/abs/2101.11605
+ https://github.com/The-AI-Summer/self-attention-cv
+ https://jaketae.github.io/study/relative-positional-encoding/
+ https://arxiv.org/pdf/1911.00203.pdf

---

IDEE DI ALTRE COSE DA FARE:
+ File di testo requirements.txt con tutti i moduli e pacchetti usati, versione minima necessaria (pip install -r requirements.txt nello script)
+ Provare training con absolute e relative positional encoding, e senza nulla
+ Provare training con anche features onehot e con solo PSSM
+ Provare training con MLP e Decoder
