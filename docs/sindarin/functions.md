# Sindarin Functions & Lambdas

## Functions

```sindarin
# Single expression
fn add(a: int, b: int): int => a + b

# Multi-line
fn greet(name: str): void =>
    println($"Hello {name}")

# No parameters
fn now(): int => 0

# Returning arrays
fn range(start: int, count: int): int[] =>
    var result: int[] = {}
    for i in 0..count =>
        result.push(start + i)
    return result
```

## Lambdas

```sindarin
# Typed lambda variable
var add: fn(int, int): int = fn(a: int, b: int): int => a + b

# Single param
var double_it: fn(int): int = fn(x: int): int => x * 2

# No params
var greet: fn(): str = fn(): str => "hello"

# Calling
var result: int = add(3, 4)
```

Lambdas capture variables from their enclosing scope.
