# DL2023 | Protein Secondary Structure Prediction using Transformers

<img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/report/introbanner.png">

The field of protein secondary structure prediction has evolved significantly over the past few decades, with methods ranging from simple statistical approaches to sophisticated machine learning models. This progression reflects both our growing understanding of protein structures and the increasing computational power available to researchers.

## 1 - Introduction

This work focuses on replicating part of the results achieved in the following paper:
> [Zhou, J., & Troyanskaya, O. G., Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction, 2014](https://arxiv.org/abs/1403.1347)

Instead of using a ConvNet, though, a Transformer will be implemented, following the architecture presented in the original paper, [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).

The Primary Structure of a protein can be seen as a sequence of **characters** (instead of words) drawn from a **vocabulary of size 20**. "Translating" from the Primary to the Secondary Structure means converting the input sequence to another sequence of **characters** drawn from a **vocabulary of size 8** (i.e. the 8 possible classes of Secondary Structure), or **size 3**.
The size of the vocabulary only depends on the level of prediction accuracy we want to achieve, as the size-3-vocabulary merges the 8 classes into 3 macroclasses. Thus, prediction on the 3-classes problem is easier.

### Biology

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
*Position-Specific Scoring Matrix* (PSSM) values can be interpreted as **word vectors** for the input characters. As these values are not obvious to calculate, I will use *CullPDB* and *CB513* datasets, available at [this link](https://www.princeton.edu/~jzthree/datasets/ICML2014), which contain a bunch of ready-to-use PSSMs.

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

More recent architectures specialized in Secondary Structure Prediction are MUFOLD-SS (2018), DeepACLSTM (2019), Contextnet (2019) and MCNN-PSSP (2022), which outperformed previous approaches with testing accuracies around 70-71% on the CB513 dataset.

Recent years have seen the emergence of attention mechanisms and transformer architectures. While primarily developed for natural language processing, these models have shown promise in understanding complex protein structures, as demonstrated by AlphaFold (Jumper et al., 2021) in tertiary structure prediction.

The current trend is towards hybrid models that combine different architectures. OPUS-TASS (Xu et al., 2020) integrates CNNs, bidirectional LSTMs, and transformer layers, achieving high accuracy in both 3-state and 8-state predictions.

| Year     | Method  | Q8 Accuracy on CullPDB (%) | Q8 Accuracy on CB513 (%) |
| :----: | :----: | :----: | :----: |
| 2011 | RaptorX-SS [(1)](http://raptorx6.uchicago.edu/StructurePropertyPred/predict/) | 69.7 | 64.9 |
| 2014 | **SC-GSN** [(2)](https://arxiv.org/abs/1403.1347) | 72.1 | 66.4 |
| 2016 | DeepCNF [(3)](https://arxiv.org/abs/1512.00843) | 75.2 | 68.3 |
| 2016 | DCRNN [(4)](https://arxiv.org/abs/1604.07176) | - | 70.4 |
| 2016 | MUST-CNN [(5)](https://arxiv.org/pdf/1605.03004) | - | 68.4 |
| 2016 | SSREDN [(6)](https://www.sciencedirect.com/science/article/abs/pii/S0950705116304713) | 73.1 | 68.2 |
| 2018 | CNNH_PSS [(7)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2067-8) | 74.0 | 70.3 |
| 2018 | MUFOLD-SS [(8)](https://arxiv.org/abs/1709.06165) | - | 73.4 |
| 2018 | CRRNN [(9)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2280-5) | - | 71.4 |
| 2019 | Contextnet [(10)](https://pubs.rsc.org/en/content/articlelanding/2019/ra/c9ra05218f) | - | 71.9 |
| 2019 | DeepACLSTM [(11)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2940-0) | - | 70.5 |
| 2020 | F1DCNN-SS [(12)](https://www.eurekaselect.com/article/103754) | 74.1 | 70.5 |
| 2022 | MCNN-PSSP [(13)](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2022.901018/full) | 74.2 | 70.6 |
| 2024 | **DL23-PSSP** | **72.1** | **69.2** |

This table presents an overview of major contributions to the field of Protein Secondary Structure Prediction (PSSP) over the past decade. Results achieved by these works reveal an apparent performance ceiling that has been challenging to surpass. Specifically, the Q8 accuracy on the CB513 dataset seems to plateau around 70-71%, while performance on the CullPDB dataset reaches up to about 75%.

A particular configuration of my model, **DL23-PSSP**, which I will present in the next section, achieves a Q8 accuracy of 69.2% on CB513, resulting competitive among other recent approaches. In my evaluation, I tried to maintain methodological consistency with previous studies, particularly in using the entire CullPDB dataset for training and the entire CB513 for validation and testing. This approach ensures a fair comparison within the established benchmarking framework of the field.

---

## 3 - Method

The Protein Secondary Structure Prediction model employed in this study is an adaptation of the **Transformer** architecture, drawing inspiration from the [work of *Vaswani et al. (2017)*](https://arxiv.org/abs/1706.03762). This modified Transformer is specifically tailored to address the unique challenges of protein sequence analysis while exploiting the architecture's ability to capture long-range dependencies between elements.

The model utilizes an *Encoder-only* structure. This design choice is motivated by the nature of Secondary Structure Prediction, which doesn't require the generation of new sequences but rather the **classification** of existing ones. Thus, the model substitutes the Decoder with a straightforward *Multi-Layer Perceptron*. This MLP serves as the classification layer, performing element-wise categorization to assign each amino acid to one of **eight** secondary structure classes.

The model's input layer is designed to process sequences of 21 amino acid residues, including a special 'NoSeq' token that indicates the end of a protein sequence. An *Embedding* layer transforms these one-hot amino acid residues inputs into dense vector representations, with the 'NoSeq' token strategically set as the *padding_index* to prevent learning on these placeholder elements.

A key innovation in this model is the implementation of **Relative Positional Encoding** within the attention mechanism. This approach is crucial for capturing local semantic relationships and patterns within the protein sequence that simple Absolute Positional Encoding techniques might fail to address. The model gains the ability to consider the spatial relationships between amino acids at different scales, a critical factor in understanding protein structure.

The Encoder stack consists of four identical layers, each incorporating a multi-head attention block with eight attention heads, followed by a position-wise Feed-Forward network. This structure allows the model to process the protein sequence at multiple levels of abstraction.

Adhering to the original Transformer design, the model employs *post-LayerNormalization*, applying normalization to the outputs of both the Attention and Feed-Forward blocks in every Encoder layer. Additionally, Dropout is integrated throughout the network as a regularization technique, enhancing the model's ability to generalize.

---

## 4 - Experimental setup

### Raw data
The datasets utilized in this project were generated using the PISCES protein culling server, a tool designed to extract protein sequences from the Protein Data Bank (PDB).

Multiple versions of the CullPDB datasets are available, varying in size. For this project, I selected the "cullpdb+profile_6133-filtered" dataset, in ordet to have access to more training data. It's worth noting that *filtered* versions of the CullPDB datasets are particularly suitable for use in conjunction with the CB513 test dataset, as it eliminates all redundant data.

Moreover, while unfiltered datasets come pre-divided into train/test/validation splits, the filtered version allows for more flexibility, as **all proteins in the filtered dataset can be used for training**, with no need to split into training and validation sets, and with testing conducted on the separate CB513 dataset.

The CullPDB-6133-filtered dataset that I will use for training comprises 6128 protein sequences, each containing a maximum of 700 amino acids. Every element in these sequences is associated with 57 features, distributed as follows:

- Features [0,22): One-hot encoded amino acid residues, with the order of 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq'. The 'NoSeq' marker indicates the end of the protein sequence.
- Features [22,31): One-hot Secondary Structure labels, with the order of 'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'.
- Features [31,33): N- and C- terminals.
- Features [33,35): Relative and absolute solvent accessibility measures.
- Features [35,57): Position-Specific Scoring Matrix (PSSM) values, representing the protein sequence profile, with the order of 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y','NoSeq'.

For per-epoch validation and final testing, we employ the CB513 dataset. This dataset contains 513 protein sequences specifically designed for testing when filtered CullPDB datasets are used for training. Importantly, CB513 maintains the same features as CullPDB, ensuring consistency in the data structure across training and testing phases.

All the datasets are officially available at [this Princeton link](https://www.princeton.edu/~jzthree/datasets/ICML2014/). Since the original URL is no longer available and the dataset is still used by many, the dataset has been moved [here](https://zenodo.org/records/7764556#.ZByi1ezMJvI) and mirrored [here](https://mega.nz/folder/xct0XSpA#SKz72JtnSAaX61QLMC_JNg).

### Data pre-processing
When downloaded, the raw *CullPDB-6133-filtered* dataset is structured as a (5534, 3990) *numpy* matrix, necessitating a reshaping process to extract individual protein sequences.

To clarify the actual data structure, I reshaped the dataset into a (5534, 700, 57) *numpy* tensor.

Some proteins in the dataset contain an unknown 'X' amino acid, that represents particular or very rare amino acids. To mitigate potential issues, I made the decision to remove all proteins containing at least one 'X' element. This refinement process resulted in the creation of the **cullpdb+profile_6133_FINAL.npy** dataset, a (3880, 700, 57) *numpy* tensor, which I promptly converted to a PyTorch tensor for my experiments.

I also performed feature processing on this dataset. First, I separated the 9 secondary structure targets from the other features. Then, I removed the features related to N and C terminals and solvent accessibility, as they were not relevant to the scope of my project. Additionally, I eliminated the two features associated with the previously removed 'X' amino acid.

This processing left me with a final dataset configuration consisting of:
- 42 features describing the primary structure: 21 amino acid residues + 21 PSSMs.
- 9 one-hot targets: 8 secondary structures + the NoSeq class, which is crucial for generating the padding mask.

### More

- Riproducibility
- Optimizer, Loss, Scheduler, HPs
- Epochs, batch size, device
- Index of github scripts w requirements

---

## 5 - Results

---

## 6 - Ablation studies

### Data structure

The first aspect I had to consider was the sizing of the dataset, integrating the necessary features in the best format for the Protein Prediction task. Aware of the Transformer architecture, understanding the advantages of using embeddings, and recognizing that PSSMs can be viewed as embedding vectors related to the position in the protein, I initially thought that PSSP could be conducted using only PSSMs.

<p float="left", align="center">
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/pssm_only.png" width="60%" />
</p>

The model's performance using only the sequence profile is promising, revealing that the the model is able to understand many structural semantic characteristics of proteins are well represented within the PSSMs. However, when I tried to include the raw one-hot sequence of residues, I noticed a further improvement. This indicates that, although much less informative than PSSMs, residues can provide useful information for better generalization of the problem and capture relationships between amino acids.

### One-Hot vs Embedded Residues

The *CullPDB* and *CB513* datasets present amino acid sequences as one-hot encoded vectors, utilizing a sparse representation. I decided to incorporate residue features alongside PSSMs in the dataset, resulting in a total of 42 features. Aware of the fact that dense representations generally yield superior results compared to sparse representations, I opted to modify the structure of my Transformer model.

The revised model accepts PSSMs and raw residues as input, with the latter converted from one-hot to integer format. As an initial operation, I implemented an Embedding layer for amino acid residues, projecting them into a dense, higher-dimensional space compared to the original 21-dimensional one-hot encoding. Given that the 21 amino acids include the fictitious 'NoSeq' token, which solely indicates the protein sequence termination, I designated this amino acid as the *padding_index*. This ensures its initialization as a null vector, preventing parameter updates during the learning process.

Embedding dimensions are typically powers of 2. Then, I defined the residue-specific embedding dimension as $embd_{dim} = d_{model} - 21$, where $d_{model}$ represents the total embedding dimension entering the Transformer, and $21$ corresponds to the number of PSSM features (which are inherently relative embedding vectors). These PSSM vectors are concatenated with the $d_{model} - 21$ embedding vectors generated from the residues, creating the tensor that is given in input to the Transformer Encoder. This approach maintains consistency with other Transformer parameters I will tune later, such as the number of attention heads or the head size.

<p float="left", align="center">
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/embedding_vs_onehot.png" width="60%" />
</p>

The performance analysis, as illustrated in the graph, reveals that the use of one-hot vectors (which are basically a very particular set of 42-dimensional embeddings) can achieve comparable performance to other embeddings (32), only being outperformed by embedding configurations with dimensions exceeding the total dimension of the one-hot version (96, 128, 192, 256). The relatively modest difference in performance can be attributed to the limited vocabulary size (21 elements) and the potentially dominant effect of PSSMs, which may overshadow the impact of the residues themselves.

This observed phenomenon underscores the robustness of the Transformer architecture in extracting relevant information from various input representations, while also highlighting the significant contribution of PSSM features in the protein secondary structure prediction task.

### Positional Encoding

**Relative Positional Encoding** proves to be vital in executing this task. Proteins exhibit highly variable lengths, and secondary structure is predominantly dependent on local relationships rather than the absolute position of an amino acid within the protein. *Absolute Positional Encoding*, which is usually employed in common tasks with Transformers and other Attention models, here does not scale well with sequences of variable length, where examination of local relationships is crucial. Relative Positional Encoding is more suitable because it is invariant to protein length.

<p float="left", align="center">
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/rel_abs%20%2B%20focal_ce_combined.png" width="49%" />
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/max_rel_position.png" width="49%" />
</p>

Adopting Relative Positional Encoding, I introduce a hyperparameter, **max_rel_position**, which determines the radius of the window within which local relationships are considered for each element $x$, fixed to the interval $[x - max_rel_position : x + max_rel_position]$. The optimal window size is found to be 80, but it is evident that proteins in the dataset can generally be classified well using short-range relationships, as performance remains comparable even with a window radius of 10. Conversely, an excessively small window significantly degrades performance. This suggests that long-range relationships are either few in number or are not the primary contributors to improved performance.

The observation that increasing the window size beyond 80 does not yield significant improvements suggests a saturation point in the useful range of positional information for this task. This could be attributed to the nature of protein secondary structures, which are often determined by interactions within a limited spatial range.
These findings not only validate the choice of Relative Positional Encoding for this task but also provide insights into the spatial scale of relationships that are most informative for protein secondary structure prediction.

### Loss Function

Analyzing the distribution of secondary structure classes within the datasets, I observed a significant imbalance among the classes. In particular, classes **B** and **I** are severely underrepresented. So, after an initial cycle of runs using **CrossEntropyLoss**, I considered introducing a new loss function, **FocalLoss**, to address this imbalance and increase sensitivity towards the classification of rarer classes.

<p float="left", align="center">
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/rel_abs%20%2B%20focal_ce_combined.png" width="49%" />
</p>

However, FocalLoss not only failed to correctly classify elements of rare classes but, due to its nature of reducing confidence in classifications of more frequent classes, it introduced more errors, significantly lowering the overall accuracy.

As a compromise, I attempted to implement a **CombinedLoss**, in which CrossEntropyLoss and FocalLoss contribute to the learning process with adjustable weights. Nevertheless, I recognized that this approach requires numerous experimental runs to determine the optimal setting, thus leaving the investigation of its effectiveness as future work.

### Softmax Temperature

Based on its frequent application in various other deep learning techniques, I decided to introduce a *Temperature* parameter applied to the logits output from the Classification Head (MLP), implemented as the actual output of the Transformer is $logit / temperature$. The purpose of this parameter is to scale the output logits from the Transformer, artificially increasing the uncertainty of the classification and thereby giving less probable classes a higher chance of being selected.

<p float="left", align="center">
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/softmax_temperature.png" width="60%" />
</p>

Higher Temperature values lead to more "uncertain" classifications. A value of 5 proved to be promising, yielding better results compared to the default value of 1.

The Temperature parameter effectively serves as a smoothing factor for the softmax function that follows the logit computation. By dividing the logits by a temperature value greater than 1, we're reducing the magnitude of differences between logits before they're passed through the softmax. This has the effect of producing a more uniform probability distribution across classes.

### Gradient Clipping

<p float="left", align="center">
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/grad_clipping.png" width="60%" />
</p>

### Layer Normalization

<p float="left", align="center">
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/layernorm.png" width="60%" />
</p>

### Optimizer

<p float="left", align="center">
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/optimizer.png" width="60%" />
</p>

### Weight Decay

<p float="left", align="center">
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/weightdecay_train.png" width="49%" />
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/weightdecay_validation.png" width="49%" />
</p>

### Dropout

<p float="left", align="center">
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/dropout.png" width="60%" />
</p>

### Protein filtering

<p float="left", align="center">
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/truncated.png" width="49%" />
  <img src="https://github.com/giovancombo/ProteinSecondaryStructurePrediction/blob/main/images/results/plots/removed_gradclip.png" width="49%" />
</p>

---

## 7 - Conclusion

