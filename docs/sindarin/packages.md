# Sindarin Packages, Imports & Tooling

## sn.yaml

```yaml
name: my-project
dependencies:
- name: sindarin-pkg-sdk
  git: https://github.com/SindarinSDK/sindarin-pkg-sdk.git
  branch: main
- name: sindarin-pkg-postgres
  git: https://github.com/SindarinSDK/sindarin-pkg-postgres.git
  branch: main
```

## Imports

```sindarin
# Import a module (relative path)
import "sdk/core/uuid"
import "sdk/io/textfile"
import "../src/list"

# Import with alias
import "sdk/core/math" as math

# Usage
var id: UUID = UUID.create()
var angle: double = math.degToRad(45.0)
```

## CLI Commands

```bash
sn --init                  # create new project
sn --install               # install dependencies
sn source.sn               # compile + run
sn source.sn -o output     # compile to named binary
sn source.sn -g            # debug build (ASAN)
sn source.sn --emit-c      # output generated C
sn source.sn --emit-model  # output JSON model
sn --clear-cache           # clear package cache
sn --clean                 # remove build cache
```

## Project Layout

```
my-project/
├── src/          # .sn source files
├── native/       # .sn.c interop files
├── tests/        # test files
├── sn.yaml       # package manifest
└── Makefile      # build targets
```

Dependencies install to `.sn/<pkg-name>/` in the project root.

## Built-in Functions

| Function | Description |
|----------|-------------|
| `print(s)` | Print string (no newline) |
| `println(s)` | Print string with newline |
| `assert(cond, msg)` | Assert condition, abort with message |
| `len(arr)` | Array length |
| `typeOf(expr)` | Type name as string (parentheses required) |
| `sizeOf(type)` | Size of type in bytes |
| `copyOf(val)` | Deep copy a value |
| `addressOf(val)` | Get pointer to value |
| `valueOf(ptr)` | Dereference pointer |
