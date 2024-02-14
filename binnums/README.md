# Binary Nums
What would the [commanum/](../commanum) example look like if we moved it over to a binary format?

We're going to implement an encoding scheme that allows us to store multiple integers, in a list,
just like our text version. But in this case, we're going to build the encoding so that we do not
have any overlapping values, or a need to escape bytes. We can worry about escapes in another
experiment.

## Encoding

The basic idea here is to provide a list of integers. This particular example does not try to support
negative integers, since that was not part of the original commanum example.

The goal here, is to separate the structural bytes from the data bytes. We need a clear way to identify
where each value starts and ends, without actually processing them.

For this particular grammar, we need a single structural value to indicate that a number follows. We'll
use `0xFF` to denote this. The thought process in choosing this value is not too involved; we need a way
distinguish this value from the integer data and one easy way to do that is to put "all" structural
bytes into the set of bytes with their most significant bit set. This would allow us to use any byte within
the range 0x80-0xFF for opcodes, so 0xFF is an easy one to pick.
 
### Integer Encodings
Since we've identified 0xFF as our structural byte, we've reduced the value space for our data bytes to
0x00-0x7F. So we'll be encoding our integers by taking the 32-bit values and re-packing them into 7bit
sequences and storing them in bytes with the most significant bit, not set.

Given a number, say 135 = `0b1000_0111`, we're going to split this up into 2 bytes. Each byte
has the potential to contain 7 bits of data from the original value. The most significant bit
of each of these bytes will be 0, followed by 7 bits of the original value.

So, 0x87 = `0b1000_0111` would encode as `0b0000_0001 0b0000_0111`.

We can take any 32-bit value and identify how many bits we'll need to represent it by subtracting
the number of leading zeros from 32. This ceiling of that value divided by 7 will then tell us how
many bytes the value will encode to.
