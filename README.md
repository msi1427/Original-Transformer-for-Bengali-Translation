# Original Transformer for Bengali Translation [Work in Progress]

I come from a developing country named Bangladesh. Our mother tongue is Bengali. In this project, my plan is to translate different languages to Bengali. I am fascinated by the transformer architecture and here I also plan to implement the whole architecture from scratch using PyTorch. I also have plans to deploy the model after the project. 

## Transformer

Transformer architecture was first introduced by Vaswani et. al. in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper. If you're a beginner, you might be wondering why the transformer architecture brought down all the previous SotA Neural Machine Translation approaches and why is it so popular among AI practitioners and scientists. I will try to list down some of the reasons.<br/>

- Better long-range dependency modeling.
- Gets rid of recurrent nets which makes the model highly parallelizable and enables GPU computation.
- Multihead Attention and Cross Attention.

However, I would state here that even if the paper title says **attention** is all you need. There had to use positional encodings, feed-forwarding layers and label smoothing to make the whole thing work.

## Understanding Transformer

 I used the following resources to understand the paper: <br/>

- [Paper explanation by Yannic Kilcher](https://www.youtube.com/watch?v=iDulhoQ2pro)
- [Visually intuitive blog and video explanation by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- [Illustrated step by step explanation by Micheal Phi](https://www.youtube.com/watch?v=4Bdc55j80l8) 

The transformer architecture looks like this

<img src = "images/Transformer Architecture.png" width="500" height="600">