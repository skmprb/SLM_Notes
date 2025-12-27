# Regex Notes - Text Preprocessing

## Whitespace Normalization

```python
text = re.sub(r'\s+', ' ', text)
```

**Purpose:** Normalizes whitespace in text by replacing multiple consecutive whitespace characters with a single space.

**Components:**
- `re.sub()` - Python's regex substitution function
- `r'\s+'` - Pattern to match:
  - `\s` matches any whitespace (spaces, tabs, newlines, etc.)
  - `+` means "one or more" occurrences
- `' '` - Replacement: single space
- `text` - Input string

**Examples:**
- `"Hello    world"` → `"Hello world"`
- `"Text\n\nwith\tmultiple\r\nspaces"` → `"Text with multiple spaces"`
- `"  Extra   spaces  "` → `" Extra spaces "`

**Use Case:** Common NLP preprocessing step to standardize text format before tokenization or analysis.

## String Strip Method

```python
text = text.strip()
```

**Purpose:** Removes leading and trailing whitespace from a string.

**What it removes:**
- Spaces at the beginning and end
- Tabs, newlines, carriage returns
- Any whitespace characters

**Examples:**
- `"  Hello world  "` → `"Hello world"`
- `"\n\tText\r\n"` → `"Text"`
- `"   "` → `""` (empty string)

**Use Case:** Clean text boundaries before processing, often used with regex normalization.

## repr() Function

```python
print(repr(sample_text))
```

**Purpose:** Returns the "representation" of an object - shows the exact string with escape sequences visible.

**What it shows:**
- Escape sequences like `\n`, `\t`, `\r`
- Quotes around strings
- Hidden whitespace characters
- Exact string content for debugging

**Examples:**
- `repr("Hello\nWorld")` → `'Hello\nWorld'`
- `repr("Text\twith\ttabs")` → `'Text\twith\ttabs'`
- `repr("  spaces  ")` → `'  spaces  '`

**Use Case:** Debug text processing to see invisible characters before/after normalization.

## Punctuation Removal with translate()

```python
import string  # Required import
text = text.translate(str.maketrans('', '', string.punctuation))
```

**Library:** `string` (Python standard library)

**Purpose:** Removes all punctuation characters from text using character translation.

**Components:**
- `string.punctuation` - Contains all punctuation characters
- `str.maketrans('', '', string.punctuation)` - Creates translation table
  - First `''` - characters to replace (none)
  - Second `''` - replacement characters (none)
  - Third parameter - characters to delete
- `translate()` - Applies the translation table

**What gets removed:**
`!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`

**Examples:**
- `"Hello, world!"` → `"Hello world"`
- `"It's a test."` → `"Its a test"`
- `"Price: $19.99"` → `"Price 1999"`

**Use Case:** Clean text for NLP analysis by removing punctuation marks.

## string.punctuation Constant

```python
import string
print(string.punctuation)
# Output: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
```

**Content:** Contains all 32 ASCII punctuation characters

**Regex Equivalent:**
```python
# Remove punctuation with regex
import re
text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', '', text)
# Or using character class
text = re.sub(r'[^\w\s]', '', text)  # Keep only word chars and whitespace
```

**Alternative Methods:**
```python
# Method 1: List comprehension
text = ''.join(c for c in text if c not in string.punctuation)

# Method 2: Filter function
text = ''.join(filter(lambda x: x not in string.punctuation, text))

# Method 3: Replace each punctuation
for p in string.punctuation:
    text = text.replace(p, '')
```

**Performance:** `translate()` is fastest, regex is flexible, list comprehension is readable.

## String Module Constants

```python
import string

# All available string constants:
string.ascii_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
string.ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
string.ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
string.digits = '0123456789'
string.hexdigits = '0123456789abcdefABCDEF'
string.octdigits = '01234567'
string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
string.printable = 'digits + ascii_letters + punctuation + whitespace'
string.whitespace = ' \t\n\r\x0b\x0c'
```

**Common Usage:**
```python
# Remove digits
text = text.translate(str.maketrans('', '', string.digits))

# Keep only letters
text = ''.join(c for c in text if c in string.ascii_letters)

# Remove whitespace
text = text.translate(str.maketrans('', '', string.whitespace))
```

## Advanced Tokenization Pattern

```python
pattern = r"\b\w+(?:'\w+)?\b|[.,!?;]"
return re.findall(pattern, text)
```

**Purpose:** Tokenizes text into words (including contractions) and specific punctuation marks.

**Pattern Breakdown:**
- `\b` - Word boundary (start)
- `\w+` - One or more word characters (letters, digits, underscore)
- `(?:'\w+)?` - Non-capturing group (optional):
  - `'` - Literal apostrophe
  - `\w+` - One or more word characters after apostrophe
  - `?` - Makes the entire group optional
- `\b` - Word boundary (end)
- `|` - OR operator
- `[.,!?;]` - Character class matching specific punctuation

**Examples:**
- `"Hello, world!"` → `['Hello', ',', 'world', '!']`
- `"It's a test."` → `['It\'s', 'a', 'test', '.']`
- `"Don't worry; be happy!"` → `['Don\'t', 'worry', ';', 'be', 'happy', '!']`

**Use Case:** NLP tokenization that preserves contractions and important punctuation.