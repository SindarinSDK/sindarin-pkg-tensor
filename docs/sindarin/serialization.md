# Sindarin Serialization

The `@serializable` annotation auto-generates `encode()` / `decode()` methods:

```sindarin
@serializable
struct Address =>
    street: str
    city: str

@serializable
struct Person =>
    name: str
    age: int
    address: Address
    tags: str[]
```

## Generated Methods

```sindarin
# Encode to JSON (via Encoder vtable)
var enc: Encoder = getJsonEncoder()
person.encode(enc)
var json: str = enc.result()

# Decode from JSON (via Decoder vtable)
var dec: Decoder = getJsonDecoder(jsonString)
var person: Person = Person.decode(dec)

# Array encode/decode
Person.encodeArray(people, enc)
var people: Person[] = Person.decodeArray(dec)
```
