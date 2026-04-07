# Sindarin Control Flow

## if / else

```sindarin
if x > 0 =>
    println("positive")
else if x == 0 =>
    println("zero")
else =>
    println("negative")
```

## match

```sindarin
match value =>
    1 => println("one")
    2 => println("two")
    else => println("other")
```

## for loops

```sindarin
# C-style
for var i: int = 0; i < 10; i++ =>
    println($"{i}")

# Range (exclusive end)
for i in 0..10 =>
    println($"{i}")

# Iteration
for item in array =>
    println($"{item}")

# Iterating a collection (calls .iter() automatically)
for entry in myMap =>
    println($"{entry.key}: {entry.value}")
```

## while

```sindarin
while condition =>
    body
```

## break / continue

```sindarin
for i in 0..100 =>
    if i == 50 => break
    if i % 2 == 0 => continue
    println($"{i}")
```
