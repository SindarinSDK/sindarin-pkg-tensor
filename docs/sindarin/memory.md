# Sindarin Memory Semantics

## as ref / as val

```sindarin
var s: MyStruct as ref         # heap-allocated, pointer semantics
var s: MyStruct as val         # stack-allocated, value semantics (default)

struct Config as ref =>        # ALL instances are heap-allocated
    host: str
    port: int

struct Point as val =>         # ALL instances are stack-allocated
    x: double
    y: double

fn process(data: str as ref): void     # parameter by reference
fn getData(): Config as val            # return by value
```

**Rule**: `as ref` params require an lvalue at the call site (a variable or field, not a literal or expression).

## copyOf / addressOf / valueOf

```sindarin
var p: Point = Point { x: 1.0, y: 2.0 }
var q: Point = copyOf(p)               # deep copy

var x: int = 42
var ptr: *int = addressOf(x)           # get pointer
var val: int = valueOf(ptr)            # dereference pointer
```

## Scoped Cleanup (using/dispose)

```sindarin
struct Resource =>
    name: str

    fn dispose(): void =>
        # cleanup logic here

using r = Resource { name: "db" } =>
    r.doWork()
# r.dispose() called automatically when block exits
```

The struct must have a `fn dispose(): void` method.
