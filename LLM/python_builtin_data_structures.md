# Python Built-in Data Structures Reference

## Collections Module

### 1. defaultdict
**Purpose**: Dictionary with default values for missing keys
```python
from collections import defaultdict

# Auto-creates missing keys with default value
dd = defaultdict(int)  # Missing keys get 0
dd['missing'] += 1     # Works! Creates key with 0, then increments to 1

# Common factory functions
defaultdict(list)      # Missing keys get []
defaultdict(set)       # Missing keys get set()
defaultdict(str)       # Missing keys get ""
defaultdict(lambda: "N/A")  # Custom default
```

### 2. Counter
**Purpose**: Count hashable objects
```python
from collections import Counter

# Count characters
Counter("hello")  # Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})

# Count list items
Counter([1, 2, 2, 3, 3, 3])  # Counter({3: 3, 2: 2, 1: 1})

# Useful methods
c = Counter("abcabc")
c.most_common(2)      # [('a', 2), ('b', 2)]
c.total()             # 6
c.subtract("abc")     # Subtract counts
```

### 3. deque (Double-ended queue)
**Purpose**: Fast appends/pops from both ends
```python
from collections import deque

d = deque([1, 2, 3])
d.appendleft(0)       # deque([0, 1, 2, 3])
d.append(4)           # deque([0, 1, 2, 3, 4])
d.popleft()           # 0, deque([1, 2, 3, 4])
d.pop()               # 4, deque([1, 2, 3])

# Rotating
d.rotate(1)           # deque([3, 1, 2])
d.rotate(-1)          # deque([1, 2, 3])
```

### 4. namedtuple
**Purpose**: Tuple with named fields
```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)      # 1 2
print(p[0], p[1])    # 1 2 (still works like tuple)

# With defaults
Person = namedtuple('Person', ['name', 'age'], defaults=[0])
p1 = Person('Alice')  # Person(name='Alice', age=0)
```

### 5. OrderedDict
**Purpose**: Dictionary that remembers insertion order
```python
from collections import OrderedDict

# Note: Regular dicts maintain order in Python 3.7+
od = OrderedDict([('a', 1), ('b', 2)])
od.move_to_end('a')   # Move 'a' to end
od.popitem(last=False)  # Pop from beginning
```

### 6. ChainMap
**Purpose**: Group multiple dicts into single view
```python
from collections import ChainMap

dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
cm = ChainMap(dict1, dict2)

print(cm['a'])        # 1 (from dict1)
print(cm['b'])        # 2 (dict1 takes precedence)
print(cm['c'])        # 4 (from dict2)
```

## Built-in Data Structures

### 7. set
**Purpose**: Unordered collection of unique elements
```python
s = {1, 2, 3, 3}      # {1, 2, 3}
s.add(4)              # {1, 2, 3, 4}
s.remove(2)           # {1, 3, 4}

# Set operations
s1 = {1, 2, 3}
s2 = {3, 4, 5}
s1 & s2               # {3} intersection
s1 | s2               # {1, 2, 3, 4, 5} union
s1 - s2               # {1, 2} difference
```

### 8. frozenset
**Purpose**: Immutable set
```python
fs = frozenset([1, 2, 3])
# fs.add(4)           # Error! Immutable
# Can be used as dict keys or in other sets
```

### 9. bytearray
**Purpose**: Mutable sequence of bytes
```python
ba = bytearray(b"hello")
ba[0] = ord('H')      # bytearray(b'Hello')
ba.append(33)         # bytearray(b'Hello!')
```

### 10. memoryview
**Purpose**: Memory-efficient slicing without copying
```python
data = bytearray(b"hello world")
mv = memoryview(data)
slice_mv = mv[6:11]   # No copy made
print(slice_mv.tobytes())  # b'world'
```

## When to Use What

| Use Case | Best Choice |
|----------|-------------|
| Count occurrences | Counter |
| Default values for missing keys | defaultdict |
| Fast queue operations | deque |
| Structured data with named fields | namedtuple |
| Unique elements, set operations | set |
| Immutable unique collection | frozenset |
| Multiple dict lookup | ChainMap |
| Memory-efficient byte operations | memoryview |
| Mutable byte sequences | bytearray |

## Performance Notes

- **deque**: O(1) append/pop from both ends vs list's O(n) for left operations
- **Counter**: Optimized for counting, faster than manual dict counting
- **set**: O(1) average lookup vs list's O(n)
- **defaultdict**: Avoids key existence checks
- **memoryview**: No memory copying for slices