# Sindarin Strings & Arrays

## String Interpolation

```sindarin
var name: str = "world"
var msg: str = $"Hello {name}!"
var calc: str = $"Result: {a + b}"
```

## String Methods

```sindarin
s.trim()                    # strip whitespace
s.toLower()                 # lowercase
s.toUpper()                 # uppercase
s.contains("sub")           # substring check
s.startsWith("pre")         # prefix check
s.endsWith("suf")           # suffix check
s.indexOf("sub")            # find position (-1 if not found)
s.charAt(0)                 # character at index
s.substring(0, 5)           # extract substring
s.split(",")                # split into array
s.splitLines()              # split by newline
s.replace("old", "new")    # replace occurrences
s.isBlank()                 # whitespace-only check
s.length                    # character count (property)
```

## Array Creation

```sindarin
var nums: int[] = {1, 2, 3}
var empty: str[] = {}
```

## Array Methods

```sindarin
arr.push(item)              # append
arr.pop()                   # remove + return last
arr.length                  # element count (property)
arr.indexOf(item)           # find position
arr.join(", ")              # join into string
len(arr)                    # element count (function)
```

## Array Indexing

```sindarin
var first: int = arr[0]
arr[2] = 99
```
