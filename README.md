# DL2023 | Protein Secondary Structure Prediction using Transformers

<img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/report/introbanner.png">

The field of protein secondary structure prediction has evolved significantly over the past few decades, with methods ranging from simple statistical approaches to sophisticated machine learning models. This progression reflects both our growing understanding of protein structures and the increasing computational power available to researchers.

## 1 - Introduction

This work focuses on replicating part of the results achieved in the following paper:
> [(1) Zhou, J., & Troyanskaya, O. G., Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction, 2014](https://arxiv.org/abs/1403.1347)

Instead of using a ConvNet, though, a Transformer will be implemented, following the architecture presented in the original paper, [*Attention Is All You Need* (2)](https://arxiv.org/abs/1706.03762).

The Primary Structure of a protein can be seen as a sequence of **characters** (instead of words) drawn from a **vocabulary of size 20**. "Translating" from the Primary to the Secondary Structure means converting the input sequence to another sequence of **characters** drawn from a **vocabulary of size 8** (i.e. the 8 possible classes of Secondary Structure), or **size 3**.
The size of the vocabulary only depends on the level of prediction accuracy we want to achieve, as the size-3-vocabulary merges the 8 classes into 3 macroclasses. Thus, prediction on the 3-classes problem is easier.

### Biological context

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

The **Secondary Structure** is the local spatial conformation of the sequence of amino acids. It is determined according to the way in which amino acids link together through hydrogen bonds. This means that **every single amino acid in the protein sequence is linked to another of the same sequence**, building what it could be interpreted as an end-to-end relationship.

There are **8** types of Secondary Structure for a protein:

| Type     | DSSP Abbr  |
| :----: | :----: |
| α-helix (4-helix) | 'H' |
| β-strand | 'E' |
| loop / irregular (coil) | 'L' or 'C' |
| β-turn | 'T' |
| bend | 'S' |
| $3_{10}$-helix (3-helix) | 'G' |
| β-bridge | 'B' |
| π-helix (5-helix) | 'I' |

It's curious to note that specific amino acids are more prone to be found in specific types of structure:
+ M, A, L, E, K prefer *helices*
+ F, I, T, V, W, Y prefer *strands*
+ G, P are known as *helix breakers*

### Data interpretation
*Position-Specific Scoring Matrix* (PSSM) values can be interpreted as **word vectors** for the input characters. As these values are not obvious to calculate, I will use *CullPDB* and *CB513* datasets, provided by the authors of the [(1)](https://arxiv.org/abs/1403.1347) paper (available at [this link](https://www.princeton.edu/~jzthree/datasets/ICML2014)), which contain a bunch of ready-to-use PSSMs.

This dataset was originally hosted at This link. Since the original URL is no longer available and the dataset is still used by many, the dataset has been moved [here](https://zenodo.org/records/7764556#.ZByi1ezMJvI) or mirrored [here](https://mega.nz/folder/xct0XSpA#SKz72JtnSAaX61QLMC_JNg).
I actually downloaded the datasets from chinese Baidu AI Studio platform (https://aistudio.baidu.com/datasetdetail/79771) because of the Princeton link not working.

Proteins that show extra characters not included in the Primary Structure Vocabulary (e.g. 'X', indicating an unknown amino acid) are removed from the datasets.

Some proteins are composed of a very long sequence of amino acids: this could bring some computational problems, due to the $O(n^2)$ complexity of the Transformer, hence these particular proteins are divided in many shorter fragments. The longest proteins are segmented into blocks of length $N <<$ n_max, with N chosen accordingly with the available computational resources (at least, $N > 30$).

As the length of most protein sequences is less than 700, the one-hot coding of residue sequences and the size of the PSSM are generally unified into 700 × 21. That is, the sequences whose length is greater than 700 will be divided into two overlapping sequences, while the sequences whose length is less than 700 will be augmented by filling in zeros. Thus, the input feature of the prediction model is a matrix with the size of 700 × 42.

### Insights about the architecture
One thing to note is that PSSMs are *NOT* absolute embedding values: they are **relative** to their mutual positions and in the protein sequence. Thus, while implementing the Transformer architecture, it can result more efficient to prefer a **relative positional encoding** strategy (as explained in [this (3)](https://arxiv.org/abs/1803.02155) paper) over the absolute one employed in the [(2)](https://arxiv.org/abs/1706.03762) paper.

Encoder-Decoder Transformers were originally presented in [(2)](https://arxiv.org/abs/1706.03762) as an Attention-based tool for Sequence-to-Sequence tasks such as Machine Translation.

But the task addressed here is actually easier, as it is a **Sequence Classification** task: at every input *character* (from the Primary Structure Vocabulary, of size 20) corresponds **one** output *character* (from the Secondary Structure Vocabulary, of size 8). In fact, with no surprises, this task can be solved using a ConvNet.
This allows to guess that I could implement only an Encoder Transformer, getting rid of the Decoder, and replacing it with a simple classification head (e.g. an MLP) over every vector from the last layer of the remaining Transformer.

---

## 2 - Related Works

Early approaches like Chou-Fasman and GOR algorithms relied on simple statistical propensities of amino acids, achieving modest accuracies of 50-60%.

A major breakthrough came with the introduction of machine learning techniques. Rost and Sander's PHD method (1993) employed neural networks and evolutionary information, pushing accuracies above 70%. The use of Position-Specific Scoring Matrices (PSSMs) became crucial, providing context beyond the amino acid sequence alone.

PSIPRED, developed by Jones in 1999, set a new benchmark with its two-stage neural network approach, achieving around 80% accuracy for three-state prediction. This marked the beginning of modern, high-accuracy prediction methods.

The deep learning era brought further advancements. Zhou and Troyanskaya's 2014 work with stacked sparse autoencoders demonstrated the power of deep architectures in capturing both local and global structural information. Recurrent Neural Networks, particularly LSTMs, proved effective in modeling long-range dependencies, as shown in works like DeepCNF (Sønderby and Winther, 2014) and DCRNN (Li and Yu, 2016).

Convolutional Neural Networks also found success, especially in capturing local structural motifs. Wang et al.'s DeepCNF (2016) combined CNNs with Conditional Random Fields, achieving state-of-the-art performance by modeling both local and global sequence-structure relationships.

Recent years have seen the emergence of attention mechanisms and transformer architectures. While primarily developed for natural language processing, these models have shown promise in understanding complex protein structures, as demonstrated by AlphaFold (Jumper et al., 2021) in tertiary structure prediction.

The current trend is towards hybrid models that combine different architectures. OPUS-TASS (Xu et al., 2020) integrates CNNs, bidirectional LSTMs, and transformer layers, achieving high accuracy in both 3-state and 8-state predictions.

---

## 3 - Method

### Raw data
The datasets utilized in this project were generated using the PISCES protein culling server, a tool designed to extract protein sequences from the Protein Data Bank (PDB).

Multiple versions of the CullPDB datasets are available, varying in size. For this project, I selected the "cullpdb+profile_6133-filtered" dataset, in ordet to have access to more training data. It's worth noting that *filtered* versions of the CullPDB datasets are particularly suitable for use in conjunction with the CB513 test dataset, as it eliminates all redundant data.

Moreover, while unfiltered datasets come pre-divided into train/test/validation splits, the filtered version allows for more flexibility, as **all proteins in the filtered dataset can be used for training**, with no need to split into training and validation sets, and with testing conducted on the separate CB513 dataset.

The CullPDB-6133-filtered dataset that I will use for training comprises 6128 protein sequences, each containing a maximum of 700 amino acids. Every element in these sequences is associated with 57 features, distributed as follows:

- Features [0,22): One-hot encoded amino acid residues, followed by a 'NoSeq' marker indicating the end of the protein sequence.
- Features [22,31): One-hot Secondary Structure labels **[L, B, E, G, I, H, S, T]**, also followed by a 'NoSeq' marker.
- Features [31,35): N and C terminals, followed by relative and absolute solvent accessibility measures.
- Features [35,57): Position-Specific Scoring Matrix (PSSM) values, representing the protein sequence profile.

For per-epoch validation and final testing, we employ the CB513 dataset. This dataset contains 513 protein sequences specifically designed for testing when filtered CullPDB datasets are used for training. Importantly, CB513 maintains the same features as CullPDB, ensuring consistency in the data structure across training and testing phases.

All the datasets are officially available at [this Princeton link](https://www.princeton.edu/~jzthree/datasets/ICML2014/). Yet, I encountered access issues with this source.

### Data pre-processing
When downloaded, the raw *CullPDB-6133-filtered* dataset is structured as a (5534, 3990) *numpy* matrix, necessitating a reshaping process to extract individual protein sequences.

To clarify the actual data structure, I reshaped the dataset into a (5534, 700, 57) *numpy* tensor.

Some proteins in the dataset contain an unknown 'X' amino acid, that represents particular or very rare amino acids. To mitigate potential issues, I made the decision to remove all proteins containing at least one 'X' element. This refinement process resulted in the creation of the **cullpdb+profile_6133_FINAL.npy** dataset, a (3880, 700, 57) *numpy* tensor, which I promptly converted to a PyTorch tensor for my experiments.

I also performed feature processing on this dataset. First, I separated the 9 secondary structure targets from the other features. Then, I removed the features related to N and C terminals and solvent accessibility, as they were not relevant to the scope of my project. Additionally, I eliminated the two features associated with the previously removed 'X' amino acid.

This processing left me with a final dataset configuration consisting of:
- 42 features describing the primary structure: 21 amino acid residues + 21 PSSMs.
- 9 one-hot targets: 8 secondary structures + the NoSeq class, which is crucial for generating the padding mask.

---

## 4 - Experimental setup

---

## 5 - Results

---

## 6 - Ablation studies

---

## 7 - Conclusion

