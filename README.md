# HRRs

Adapted from http://github.com/stephantul/plate.git. The basic algorithms are identical, but I found that Magnitude gives much better performance than Reach, since it uses a preprocessed form of the embedding dataset. This version also allows arbitrarily long sets of key, value pairs to be stuffed into a single encoded vector, providing an easy way to see what happens when this scheme is pushed to its limits.

This is a simple implementation of holographic reduced representation, as described by Plate (1993). The idea behind it is that you can recursively encode vector representations through circular convolution. A nice property of circular convolution is that it is invertible by involution, as long as you know one of the constituents of the representation.

Original paper: Holographic Reduced Representations: Convolution Algebra for Compositional Distributed Representations",
 IEEE Transactions on Neural Networks 6:3:623-641 1995.

# Setup

1. Download embeddings. The embeddings used in this example are GloVe vector embeddings from the Common Crawl corpus, using the "medium" flavor Magnitude encoding, from http://magnitude.plasticity.ai/fasttext/medium/crawl-300d-2M.magnitude
2. Run, as shown in the examples below

# Usage



# Example

```bash
# download embeddings
> curl -O http://magnitude.plasticity.ai/fasttext/medium/crawl-300d-2M.magnitude

# now run

# this one works as expected
> python3 example.py --embeddings glove.840B.300d.magnitude -s 'dog:subject chase:verb cat:object' 
Loading embeddings from glove.840B.300d.magnitude

subject:  dog
verb:     chase
object:   cat

subject-> dog,   score = 1.5568918287666396
verb->    chase, score = 1.152669560199552
object->  cat,   score = 1.519399347639514

# this one, not so much
> python3 example.py --embeddings glove.840B.300d.magnitude -s 'whale:subject eat:verb squid:object but:connective why:reason' 
Loading embeddings from glove.840B.300d.magnitude

subject:     whale
verb:        eat
object:      squid
connective:  but
reason:      why

subject->    whale, score = 1.7913796329070313
verb->       eat,   score = 1.7029636007098716
object->     squid, score = 1.6767798134144103
connective-> but,   score = 1.639495820476182
reason->     that,  score = 2.33219373058271

```
