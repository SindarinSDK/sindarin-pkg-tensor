# Sindarin Types & Variables

## Primitive Types

| Type | Size | Maps to C |
|------|------|-----------|
| `int` | 64-bit signed | `long long` |
| `int32` | 32-bit signed | `int` |
| `uint` | 64-bit unsigned | `unsigned long long` |
| `uint32` | 32-bit unsigned | `unsigned int` |
| `long` | 64-bit signed | `long long` |
| `double` | 64-bit float | `double` |
| `float` | 32-bit float | `float` |
| `str` | string | `char *` |
| `char` | single character | `char` |
| `bool` | true/false | `int` |
| `byte` | 8-bit unsigned | `unsigned char` |
| `void` | no value | `void` |
| `nil` | null value | `NULL` |

## Composite Types

```sindarin
var arr: int[] = {1, 2, 3}          # array
var names: str[] = {"a", "b"}       # string array
var matrix: int[][] = {{1,2},{3,4}} # nested array
var ptr: *byte = addressOf(b)       # pointer
```

## Variables

```sindarin
var x: int = 42
var name: str = "hello"
var pi: double = 3.14159
var flag: bool = true
var empty: int[] = {}
```

Type inference with `var`:

```sindarin
var x = 42              # inferred as int
var s = "hello"         # inferred as str
```

## Type Conversions

All type conversions are **explicit method calls** using dot notation. There is no `as` cast operator — `as` is only for memory qualifiers (`as ref` / `as val`).

### Conversion Methods

| From | Method | To | Example |
|------|--------|----|---------|
| `int` | `.toDouble()` | `double` | `42.toDouble()` → `42.0` |
| `int` | `.toLong()` | `long` | `x.toLong()` |
| `int` | `.toUint()` | `uint` | `x.toUint()` |
| `int` | `.toByte()` | `byte` | `x.toByte()` |
| `int` | `.toChar()` | `char` | `65.toChar()` → `'A'` |
| `long` | `.toInt()` | `int` | `ts.toInt()` |
| `long` | `.toDouble()` | `double` | `ms.toDouble()` |
| `double` | `.toInt()` | `int` | `3.14.toInt()` → `3` |
| `double` | `.toLong()` | `long` | `val.toLong()` |
| `uint` | `.toInt()` | `int` | `u.toInt()` |
| `uint` | `.toLong()` | `long` | `u.toLong()` |
| `uint` | `.toDouble()` | `double` | `u.toDouble()` |
| `byte` | `.toInt()` | `int` | `b.toInt()` |
| `byte` | `.toChar()` | `char` | `b.toChar()` |
| `bool` | `.toInt()` | `int` | `true.toInt()` → `1` |
| `char` | `.toInt()` | `int` | `'A'.toInt()` → `65` |
| `char` | `.toString()` | `str` | `c.toString()` |
| `str` | `.toInt()` | `int` | `"42".toInt()` |
| `str` | `.toLong()` | `long` | `"100".toLong()` |
| `str` | `.toDouble()` | `double` | `"3.14".toDouble()` |
| `str` | `.toBytes()` | `byte[]` | `"hello".toBytes()` |
| `byte[]` | `.toString()` | `str` | `bytes.toString()` |

### Usage Examples

```sindarin
# Numeric widening
var x: int = 42
var d: double = x.toDouble()
var l: long = x.toLong()

# Numeric narrowing
var ms: long = 5000
var sleepTime: int = ms.toInt()
Time.sleep(sleepTime)

# double to int (truncates)
var pi: double = 3.14159
var rounded: int = pi.toInt()           # 3

# Byte conversion for binary protocols
var value: int = 0xFF
var b: byte = value.toByte()
var back: int = b.toInt()

# String parsing
var input: str = "12345"
var num: int = input.toInt()
var precise: double = "3.14".toDouble()

# String/byte encoding
var text: str = "hello"
var encoded: byte[] = text.toBytes()
var decoded: str = encoded.toString()

# Chaining with expressions
var count: int = len(items)
var countAsDouble: double = count.toDouble()
```

### Common Patterns

```sindarin
# Pass int where double is needed (e.g., metrics)
metrics.setGauge("count", len(arr).toDouble(), labels)

# Pass long where int is needed (e.g., Time.sleep)
Time.sleep(intervalMs.toInt())

# Compute with mixed types
var rate: double = count.toDouble() / elapsed.toDouble()

# Convert pow() result (double) to long
var backoff: long = math.pow(2.0, attempt.toDouble()).toLong() * 1000
```
