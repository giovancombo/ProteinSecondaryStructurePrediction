# DL2023 | Using Transformers for Protein Secundary Structure Prediction

https://en.wikipedia.org/wiki/Protein_secondary_structure?wprov=sfla1

**Protein Structure Prediction** (PSP) has always been one of the most popular applications of Deep Learning, and one of the most important fields of application of bioinformatics and computational biology, as it is useful for *drug design* and *novel enzymes design*.

Historical methods for determining/predicting the Secundary Structure of a protein from its Primary Structure were the **Chou-Fasman** method, the **GOR** method, then replaced by algorithms like **DEFINE**, **STRIDE**, **ScrewFit**, **SST**, and **DSSP**, which formally established a dictionary of all the currently known types of Secundary Structure.

Algorithms for *predicting* Secundary Structure are **PSIpred**, **SAM**, **PORTER**, **PROF** and **SABLE**.

**Guarda meglio gli altri algoritmi**

**!! Nota: Crea una tabella "benchmark" con le accuracies dei vari algoritmi storici, e poi più avanti anche una tabella benchmark riguardo i modelli usati sul dataset specifico CullPDB, per fare un confronto.**

## Background and Related Works
This work focuses on replicating part of the results achieved in the following paper:
> [(1) Zhou, J., & Troyanskaya, O. G., Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction, 2014](https://arxiv.org/abs/1403.1347)

Instead of using a ConvNet, though, a Transformer will be implemented, following the architecture presented in the original paper, [*Attention Is All You Need* (2)](https://arxiv.org/abs/1706.03762).

The Primary Structure of a protein can be seen as a sequence of **characters** (instead of words) drawn from a **vocabulary of size 20**. "Translating" from the Primary to the Secundary Structure means converting the input sequence to another sequence of **characters** drawn from a **vocabulary of size 8** (i.e. the 8 possible classes of Secundary Structure), or **size 3**.
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

The **Secundary Structure** is the local spatial conformation of the sequence of amino acids. It is determined according to the way in which amino acids link together through hydrogen bonds. This means that **every single amino acid in the protein sequence is linked to another of the same sequence**, building what it could be interpreted as an end-to-end relationship.

There are **8** types of Secundary Structure for a protein:

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

I actually downloaded the datasets from chinese Baidu AI Studio platform (https://aistudio.baidu.com/datasetdetail/79771) because of the Princeton link not working.

Proteins that show extra characters not included in the Primary Structure Vocabulary (e.g. 'X', indicating an unknown amino acid) are removed from the datasets.

Some proteins are composed of a very long sequence of amino acids: this could bring some computational problems, due to the $O(n^2)$ complexity of the Transformer, hence these particular proteins are divided in many shorter fragments. The longest proteins are segmented into blocks of length $N <<$ n_max, with N chosen accordingly with the available computational resources (at least, $N > 30$).

### Some insights about the Architecture
One thing to note is that PSSMs are *NOT* absolute embedding values: they are **relative** to their mutual positions and in the protein sequence. Thus, while implementing the Transformer architecture, it can result more efficient to prefer a **relative positional encoding** strategy (as explained in [this (3)](https://arxiv.org/abs/1803.02155) paper) over the absolute one employed in the [(2)](https://arxiv.org/abs/1706.03762) paper.

Encoder-Decoder Transformers were originally presented in [(2)](https://arxiv.org/abs/1706.03762) as an Attention-based tool for Sequence-to-Sequence tasks such as Machine Translation.

But the task addressed here is actually easier, as it is a **Sequence Classification** task: at every input *character* (from the Primary Structure Vocabulary, of size 20) corresponds **one** output *character* (from the Secundary Structure Vocabulary, of size 8). In fact, with no surprises, this task can be solved using a ConvNet.
This allows to guess that I could implement only an Encoder Transformer, getting rid of the Decoder, and replacing it with a simple classification head (e.g. an MLP) over every vector from the last layer of the remaining Transformer.

---

**COSE AGGIUNTE - DA INTEGRARE ALLE COSE SOPRA**

The training datasets used in this project are taken from the ICML 2014 Deep Supervised and Convolutional Generative Stochastic Network paper [1]. The datasets in this paper were created using the PISCES protein culling server, that is used to cull protein sequences from the protein data bank (PDB) [14]. As of Oct 2018, an updated dataset, with any of the duplicated in the previous __6133 dataset removed, has been release called cullpdb+profile_5926. Both of the datasets contain a filtered and unfiltered version. The filtered version is filtered to remove any redundancy with the CB513 test dataset. The unfiltered datasets have the train/test/val split whilst for filtered, all proteins can be used for training and test on CB513 dataset. Both filtered and unfiltered dataset were trained and evaluated on the models.

The publicly available training dataset used in our models was the CullPDB dataset – CullPDB 6133. The dataset was produced by PISCES CullPDB [2] and contains 6128 protein sequences, their constituent amino acids and the associated 57 features of each residue. The first 22 features are the amino acid residues, followed by a ‘NoSeq’ which just marks the end of the protein sequence. The next 9 features are the secondary structure labels (L, B, E, G, I, H, S, T), similarly followed by a ‘NoSeq’. The next 4 features are the N and C terminals, followed by the relative and absolute solvent accessibility. The relative accessibility is normalized by the largest accessibility value in a protein and thresholded at 0.15 and the absolute accessibility is thresholded at 15. The final 22 features represent the protein sequence profile. The secondary structure label features and relative and solvent accessibilities are hidden during testing. There exists a filtered and unfiltered version of this dataset, with the filtered data having redundancy with the CB513 test dataset removed.
--> DATASET USED = CullPDB 6133 FILTERED

Test: Three datasets were used for evaluating the models created throughout this project:
+ CB513
+ CASP10
+ CASP11
The CB513 dataset is available at: https://www.princeton.edu/~jzthree/datasets/ICML2014/

## Model Architecture

### Encoder

#### Attention Mechanism

#### Relative Positional Encoding

### Classification Head

### Decoder

## Training and Results

## Conclusion

## References

---

IDEE DI ALTRE COSE DA FARE:
+ File di testo requirements.txt con tutti i moduli e pacchetti usati, versione minima necessaria (pip install -r requirements.txt nello script)
