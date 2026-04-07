# Sindarin Threading

```sindarin
sync var counter: int = 0           # thread-safe variable

fn increment(): int =>
    for i in 1..101 =>
        lock (counter) =>           # mutex lock
            counter = counter + 1
    return 1

fn main(): void =>
    var t1: int = & increment()     # spawn thread (& operator)
    var t2: int = & increment()

    var r1 = t1 !                   # join thread (! operator)
    var r2 = t2 !

    println($"Counter: {counter}")  # 200
```

## Spawn, Sync, Detach

| Syntax | Behavior |
|--------|----------|
| `var r: T = &fn()` | Spawn thread, r is pending |
| `r!` | Block until synced, returns value |
| `[r1, r2, ...]!` | Wait for multiple threads |
| `var r: T = &fn()!` | Spawn and wait immediately |
| `x! + y!` | Sync in expressions |
| `r~` | Detach thread — runs independently, no auto-join |
| `&fn()` | Fire and forget (void functions only) |

## Multiple Thread Synchronization

Sync multiple threads at once with array syntax:

```sindarin
var r1: int = &add(1, 2)
var r2: int = &add(3, 4)
var r3: int = &multiply(5, 6)

// Wait for all to complete
[r1, r2, r3]!

// Now all are synchronized
print(r1 + r2 + r3)
```

Individual synchronization is also valid:

```sindarin
r1!
r2!
r3!
```

## Detach (`~`)

By default, spawned threads auto-join when the handle goes out of scope. This blocks the caller. Use `~` to detach:

```sindarin
fn handleClient(conn: Connection): int =>
    # long-running handler
    return 0

fn serve(listener: Listener): void =>
    while true =>
        var conn: Connection = listener.accept()
        var handle: int = &handleClient(conn)
        handle~    # detach: loop continues immediately
```

A detached variable cannot be synchronized:

```sindarin
var r: int = &compute()
r~
r!    # COMPILE ERROR: cannot sync detached thread
```

## Sync Variables and Locks

- `sync var` — declares a thread-safe variable (integer types only)
- `lock (syncVar) =>` — acquires the mutex for the block
- Single operations (`counter++`, `counter += 5`) are atomic on sync vars
- Multi-statement operations need `lock` blocks

```sindarin
sync var total: int = 0

fn addValues(data: int[]): void =>
    lock (total) =>
        for v in data =>
            total += v
```

## Common Patterns

### Parallel Computation

```sindarin
var r1: int = &compute_square(5)
var r2: int = &compute_square(10)
[r1, r2]!
print($"Squared values: {r1}, {r2}\n")
```

### Parallel File Reads

```sindarin
var f1: str = &TextFile.readAll("file1.txt")
var f2: str = &TextFile.readAll("file2.txt")
var f3: str = &TextFile.readAll("file3.txt")

[f1, f2, f3]!

print($"Total: {f1.length + f2.length + f3.length} bytes\n")
```

### Worker Pool Pattern

```sindarin
var r1: str = &process(items[0])
var r2: str = &process(items[1])
var r3: str = &process(items[2])

[r1, r2, r3]!

print($"{r1}\n{r2}\n{r3}\n")
```
