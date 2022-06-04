"""Test with word embeddings."""
from pymagnitude import Magnitude
from hrr_ops import circular_convolution, decode
import numpy as np
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog='example.py', add_help=False,
                                     description='An example implementation of Holographic Reduced Representations, '
                                                 'as described in Plate(1995).',
                                     epilog='''
Example:

 > python3 -e glove.840B.300d.magnitude -s "dog:subject chase:verb cat:object"
 
 should produce something similar to:
 
Loading embeddings from glove.840B.300d.magnitude

subject:  dog
verb:     chase
object:   cat

subject-> dog,   score = 1.5568918287666396
verb->    chase, score = 1.152669560199552
object->  cat,   score = 1.519399347639514

Original paper: "Holographic Reduced Representations: Convolution Algebra for Compositional Distributed Representations",
 IEEE Transactions on Neural Networks 6:3:623-641 1995.
''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-e', '--embeddings', type=str, required=True, help='Path for a file of embeddings. '
                                                                              'word2vec and GloVe embeddings work '
                                                                              'equally well.')
    required.add_argument('-s', '--sentence', type=str, required=True, help='A "sentence", really just a set of '
                                                                            'key/value pairs in the format "val1:key1 '
                                                                            'val2:key2 ... valN:keyN". Both keys and '
                                                                            'values must be strings that are '
                                                                            'present in the embeddings dataset.')
    if not {'-h', '--help'}.isdisjoint(sys.argv[1:]):
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    s = []

    print(f'Loading embeddings from {args.embeddings}\n')
    mag = Magnitude(args.embeddings)

    pairs = [pair.split(':') for pair in args.sentence.split(' ')]
    wmax = max(len(x) for x, y in pairs) + 1
    pmax = max(len(y) for x, y in pairs) + 2
    for word, pos in pairs:
        print(f'{pos + ":":<{pmax}} {word}')
        pvec = mag.query(pos)
        wvec = mag.query(word)
        enc = circular_convolution(pvec, wvec)
        s.append(enc)
    print()
    aggregate = np.sum(np.array(s), axis=0)
    for word, pos in pairs:
        vec = decode(mag.query(pos), aggregate)
        result = mag.most_similar(vec[0], topn=10)
        print(f'{pos + "->":<{pmax}} {result[0][0] + ",":<{wmax}} score = {result[0][1]}')


if __name__ == '__main__':
    main()
