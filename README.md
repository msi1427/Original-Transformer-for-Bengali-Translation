# Original Transformer for Bengali Translation [Work in Progress]

I come from a developing country named Bangladesh. Our mother tongue is Bengali. In this project, my plan is to translate different languages to Bengali. I am fascinated by the transformer architecture and here I also plan to implement the whole architecture from scratch using PyTorch. I also have plans to deploy the model after the project. 

## Transformer

Transformer architecture was first introduced by Vaswani et. al. in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper. If you're a beginner, you might be wondering why the transformer architecture brought down all the previous SotA Neural Machine Translation approaches and why is it so popular among AI practitioners and scientists. I will try to list down some of the reasons.<br/>

- Better long-range dependency modeling.
- Gets rid of recurrent nets which makes the model highly parallelizable and enables GPU computation.
- Multi-head Attention.

However, I would state here that even if the paper title says **attention** is all you need. There had to use positional encodings, feed-forwarding layers and label smoothing to make the whole thing work.

## Understanding Transformer from theoretical perspective

 I used the following resources to understand the ins and outs of transformer: <br/>

- [Paper explanation by Yannic Kilcher](https://www.youtube.com/watch?v=iDulhoQ2pro)
- [Visually intuitive blog and video explanation by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- [Illustrated step by step explanation by Micheal Phi](https://www.youtube.com/watch?v=4Bdc55j80l8) 
- [Blog post and lecture by Peter Bloem](http://peterbloem.nl/blog/transformers)

I broke down the blocks of the architecture into **5 distinguishing parts** to understand as a whole.

The transformer architecture looks like this

<img src = "images/Transformer Architecture.png">

### 1. Input Embeddings

### 2. Positional Encoding

### 3. Encoder Layer

#### 3.1. Multi-head Attention

##### 3.1.1. Self-Attention

#### 3.2. Residual Connection

#### 3.3. Layer Normalization

#### 3.4. Pointwise Feed Forward

### 4. Output Embedding & Positional Encoding

### 5. Decoder Layer

#### 5.1 Multi-head Attention Layer 1

##### 5.1.1. Self-Attention

##### 5.1.2. Look-Ahead Mask

#### 5.2 Multi-head Attention Layer 2 (Cross-attention)

#### 5.3 Residual Connection, Layer Normalization & Pointwise Feed Forward

#### 5.4 Linear Classifier

## Implementing transformer from Scratch

I am implementing the original paper from scratch using PyTorch. I am using the following resources to understand the implementation:

- [The Annotated Transformer by Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch official implementation](https://github.com/pytorch/pytorch/blob/187e23397c075ec2f6e89ea75d24371e3fbf9efa/torch/nn/modules/transformer.py) 
- [Implementation from Scratch by Gordic Aleksa](https://github.com/gordicaleksa/pytorch-original-transformer#hardware-requirements)
- [Implementation Walkthrough by Aladdin Persson](https://www.youtube.com/playlist?list=PLhhyoLH6Ijfyl_VMCsi54UqGQafGkNOQH)

