# Package Development Workflows

This document describes workflows for developing the Skynet autonomous learning system, built in Sindarin.

## !!! ALWAYS DO THIS !!!

!!! ALWAYS ASK FOR PERMISSION BEFORE MAKING CODE CHANGES !!!
!!! CHANGING CODE WITHOUT MY APPROVAL IS A VIOLATION OF THIS INSTRUCTION !!!
!!! NEVER MENTION A FAILURE/ISSUE AS "PRE-EXISTING", INVESTIGATE THE ISSUE AND SUGGEST A FIX !!!

!!! WHEN I SAY "ALWAYS ASK FOR PERMISSION BEFORE MAKING CODE CHANGES" I MEAN:
- DO NOT write code, edit files, or run edit/write tools until the user explicitly says "yes" or "go ahead"
- DO NOT propose a plan and then execute it in the same message
- DO NOT revert, fix, or "clean up" code without explicit approval
- "Can I proceed?" is ASKING. Touching a file is NOT asking.
- INVESTIGATE and REPORT findings. Then WAIT for instruction.
- The ONLY exception is reading files for research purposes !!!

## !!! WORKFLOW ENFORCEMENT !!!

The correct workflow is ALWAYS:
1. RESEARCH the problem (read files, grep, explore)
2. REPORT your findings to the user
3. SUGGEST what you think should change (describe in words, do NOT write the code yet)
4. WAIT for the user to approve or redirect
5. ONLY THEN make the approved changes — nothing more, nothing less
6. If you discover additional issues during implementation, STOP and report them. DO NOT fix them without approval.

NEVER skip steps 3 and 4. NEVER combine steps 3-5 into one action.


## Sindarin Language Reference

This is a pure Sindarin project. All application code is in `.sn` files, with `.sn.c` for C library interop.

Full reference: `docs/sindarin-language.md`

**Before writing or reviewing Sindarin code, read the relevant topic(s) from `docs/sindarin/`:**

| Topic | File | When to read |
|-------|------|-------------|
| Types & variables | `docs/sindarin/types.md` | Declaring variables, choosing types, type inference, **type conversions** |
| Functions & lambdas | `docs/sindarin/functions.md` | Writing functions, closures, callbacks |
| Structs | `docs/sindarin/structs.md` | Defining structs, methods, static methods |
| Memory | `docs/sindarin/memory.md` | `as ref`/`as val`, `copyOf`, `using`/`dispose` |
| Control flow | `docs/sindarin/control-flow.md` | if/else, match, for, while, break/continue |
| Strings & arrays | `docs/sindarin/strings-and-arrays.md` | String methods, interpolation, array operations |
| Generics & interfaces | `docs/sindarin/generics.md` | Generic structs/functions, interfaces, iterators, operators |
| Serialization | `docs/sindarin/serialization.md` | `@serializable`, encode/decode |
| Threading | `docs/sindarin/threading.md` | `sync var`, `lock`, thread spawn (`&`) and join (`!`) |
| Native C interop | `docs/sindarin/native-interop.md` | `@source`, `@alias`, `.sn.c` files, type mapping |
| Packages & tooling | `docs/sindarin/packages.md` | `sn.yaml`, imports, CLI commands, built-in functions |

Do NOT load all topics at once. Read only what is relevant to the current task.

## Type Conversions (Critical)

Sindarin has **NO `as` cast operator**. The `as` keyword is only for memory qualifiers (`as ref` / `as val`).

All type conversions use **explicit dot-method calls**:

```sindarin
var x: int = 42
var d: double = x.toDouble()       # int → double
var l: long = x.toLong()           # int → long
var b: byte = x.toByte()           # int → byte

var ms: long = 5000
var i: int = ms.toInt()            # long → int

var pi: double = 3.14
var n: int = pi.toInt()            # double → int (truncates)
var m: long = pi.toLong()          # double → long

var b: byte = 0xFF
var bi: int = b.toInt()            # byte → int

var s: str = "42"
var parsed: int = s.toInt()        # str → int
var pd: double = "3.14".toDouble() # str → double
var bytes: byte[] = s.toBytes()    # str → byte[]
var text: str = bytes.toString()   # byte[] → str
```

**Never write `x as int` or `val as double` — this will not compile.**

Full reference: `docs/sindarin/types.md` (Type Conversions section).
