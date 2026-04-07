# Sindarin Structs

## Definition

```sindarin
struct Point =>
    x: double
    y: double

    # Instance method — accesses fields via self
    fn distance(other: Point): double =>
        var dx: double = self.x - other.x
        var dy: double = self.y - other.y
        return Math.sqrt(dx * dx + dy * dy)

    # Static method — called on the type, not an instance
    static fn origin(): Point =>
        return { x: 0.0, y: 0.0 }
```

## Instantiation

When the variable type is specified, the struct name can be omitted from the literal:

```sindarin
# Shorthand — type is inferred from the variable declaration
var p: Point = { x: 1.0, y: 2.0 }

# Explicit — equivalent to the above
var p: Point = Point { x: 1.0, y: 2.0 }

# Static factory methods
var o: Point = Point.origin()

# Method calls
var d: double = p.distance(o)
```

The shorthand `{ field: value }` works anywhere the target type is known from context:

```sindarin
# Variable declarations
var config: GnnConfig = { inputDim: 5, hiddenDim: 128, numActions: 3, numLayers: 2, arch: "gat", dropoutRate: 0.1 }

# Multi-line
var obs: Observation = {
  id: "obs-1",
  source: "csv-replay",
  timestamp: 1000,
  features: {NumericField { key: "price", value: 100.5 }},
  metadata: {}
}
```

**Note:** The explicit form `Type { ... }` is still required when the type cannot be inferred — for example, inside array literals or when passing to a function that accepts multiple types:

```sindarin
# Array literal elements need the explicit type
var edges: Edge[] = {
  Edge { source: 0, target: 1, weight: 0.9, kind: "temporal" },
  Edge { source: 1, target: 2, weight: 0.8, kind: "temporal" }
}

# Nested struct fields inside a shorthand literal
var pipeline: GuardrailPipeline = {
  confidence: ConfidenceGuardrail { threshold: 0.7 },
  rateLimiter: RateLimiterGuardrail { maxActions: 100, windowMs: 60000, actionTimestamps: {} }
}
```

## Utility Class Pattern (Static-Only)

```sindarin
struct Path =>
    _unused: int32

    static fn join(a: str, b: str): str =>
        return sn_path_join(a, b)

    static fn exists(path: str): bool =>
        return sn_path_exists(path) != 0
```
