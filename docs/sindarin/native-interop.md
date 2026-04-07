# Sindarin Native C Interop

## Binding C Functions

```sindarin
@include <math.h>
@link m

@alias "sin"
native fn sin(x: double): double

@alias "cos"
native fn cos(x: double): double
```

## Binding C Structs

```sindarin
@source "uuid.sn.c"

@alias "RtUuid"
native struct UUID as ref =>
    @alias "high"
    _high: long
    @alias "low"
    _low: long

    # Static factory (calls native C function)
    static fn create(): UUID =>
        return sn_uuid_create()

    # Native instance method
    @alias "sn_uuid_get_version"
    native fn version(): int

    # Sindarin method that calls native
    fn toString(): str =>
        return sn_uuid_to_string(self)
```

## The .sn.c File

The corresponding C implementation:

```c
// uuid.sn.c
#include "runtime/array/runtime_array.h"
#include "runtime/string/runtime_string_h.h"

typedef __sn__UUID RtUuid;

RtUuid sn_uuid_create(void) {
    RtUuid u;
    // ... fill with random bytes
    return u;
}

int sn_uuid_get_version(RtUuid *self) {
    return (self->high >> 12) & 0xF;
}

char *sn_uuid_to_string(RtUuid *self) {
    char *buf = sn_alloc_str(37);
    // ... format UUID string
    return buf;
}
```

## Type Mapping (Native Boundary)

| Sindarin | C type |
|----------|--------|
| `str` | `char *` (auto-converted via `rt_managed_pin()`) |
| `int` | `long long` |
| `int32` | `int` |
| `double` | `double` |
| `float` | `float` |
| `bool` | `int` |
| `byte` | `unsigned char` |
| `Struct` (as val) | `__sn__StructName` |
| `Struct` (as ref) | `__sn__StructName *` |
| `T[]` | `SnArray *` |
| `*T` | `T *` |

## Annotations

| Annotation | Purpose |
|------------|---------|
| `@source "file.sn.c"` | Include a C implementation file |
| `@include <header.h>` | Add a C include directive |
| `@link libname` | Link against a library |
| `@alias "c_name"` | Map Sindarin name to C name |
| `native` | Mark function/struct as C interop |
| `@serializable` | Auto-generate encode/decode methods |

## Critical Include Paths

```c
#include "runtime/array/runtime_array.h"       // NOT runtime/runtime_array.h
#include "runtime/string/runtime_string_h.h"   // NOT runtime/runtime_string_h.h
#include "runtime/arena/managed_arena.h"
```
