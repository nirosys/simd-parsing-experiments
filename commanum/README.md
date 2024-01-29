# Commanum
This example implements a parser for comma separated integers.
The grammar is pretty simple:
- Integers are comprised of one or more decimal digits
- Each integer is separated by a comma, on a single line
- Spaces can exist before and after each comma, as well as at the start and end of the line.

An example:
```
12, 35, 1235,   4  ,205
```

The example takes a single argument, a string containing a numbers separated by commas, and
maybe some spaces:
```
$ commanum/commanum "1,  232, 30000"
Numbers:
   1
   232
   30000
$  
```

## Implementation
This implementation models itself in a way to reflect how a more "serious" implementation
could be constructed, in order to more easily facilitate supporting multiple architectures.

The `Deserializer` template class implemented in the global namespace is parameterized by a
template parameter `Input` that is expected to provide the platform-specific implementations
of a set of functions which provide the underlying SIMD capabilities.
