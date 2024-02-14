# SIMD Parsing Examples / Experiments

This repo contains some number of examples of using SIMD to parse
textual and binary data.

The goal of this work is to build familiarity with where and how SIMD can be
utilized for parsing data encoding formats, as well as experiment with some
ideas as they develop.

I've implemented everything in this repo using C++ (targetting C++17). This
is not meant to be any commentary on the language itself, it was just the
mood I was in.


## Examples

### Text

* [commanum](commanum/) - A parser for a comma-separated integer grammar.

### Binary

* [binnums](binnums/) - A parser for a binary integer encoding.
