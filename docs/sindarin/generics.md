# Sindarin Generics, Interfaces, Iterators & Operators

## Generic Structs

```sindarin
struct List < T > =>
    items: T[]
    count: int

    static fn new(): List < T > =>
        return List < T > { items: {}, count: 0 }

    fn push(item: T): void =>
        self.items.push(item)
        self.count = self.count + 1

    fn get(index: int): T =>
        return self.items[index]
```

## Multiple Type Parameters

```sindarin
struct Entry < K, V > =>
    key: K
    value: V

struct HashMap < K: Hashable, V > =>
    keys: K[]
    values: V[]
    count: int
```

## Generic Functions

```sindarin
fn mergeSort < T: Comparable > (arr: T[]): T[] =>
    # T must satisfy the Comparable interface
    ...
```

## Instantiation

```sindarin
var nums: List<int> = List<int>.new()
nums.push(42)

var m: HashMap<IntKey, str> = HashMap<IntKey, str>.new()
m.set(IntKey { value: 1 }, "one")
```

## Interfaces

Structural typing — a struct satisfies an interface by implementing its methods.

```sindarin
interface Hashable =>
    fn hash(): int
    fn equals(other: Self): bool

interface Comparable =>
    fn compare(other: Self): int
```

`Self` refers to the implementing type. No explicit `implements` keyword — just define the methods:

```sindarin
struct IntKey =>
    value: int

    fn hash(): int =>
        return self.value * 2654435761

    fn equals(other: IntKey): bool =>
        return self.value == other.value
```

Now `IntKey` can be used wherever `Hashable` is required:

```sindarin
var set: HashSet<IntKey> = HashSet<IntKey>.new()
```

## Iterators

Collections are iterable via the `hasNext()` / `next()` pattern:

```sindarin
struct ListIter < T > =>
    items: T[]
    index: int

    fn hasNext(): bool =>
        return self.index < len(self.items)

    fn next(): T =>
        var v: T = self.items[self.index]
        self.index = self.index + 1
        return v
```

The parent collection exposes an `iter()` method:

```sindarin
struct List < T > =>
    items: T[]
    # ...

    fn iter(): ListIter < T > =>
        return ListIter < T > { items: self.items, index: 0 }
```

Then `for ... in` works automatically:

```sindarin
var nums: List<int> = List<int>.new()
nums.push(10)
nums.push(20)

for n in nums =>
    println($"{n}")    # calls iter(), hasNext(), next() under the hood
```

## Operator Overloading

```sindarin
struct Point =>
    x: int
    y: int

    operator == (other: Point): bool =>
        return self.x == other.x && self.y == other.y
```

Usage:

```sindarin
var p1: Point = Point { x: 1, y: 2 }
var p2: Point = Point { x: 1, y: 2 }
assert(p1 == p2, "should be equal")
```
