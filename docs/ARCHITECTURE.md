# Architecture

## Overview
## Design Patterns

### Strategy Pattern
Every engine implements BaseEngine and exposes one method: _perturb().
EchoContext holds any engine and calls it — swapping engines requires
one line of code.

### RAII / Resource Safety
SurrogateModel loads once and is reused across all engine calls.
PyTorch tensors are cloned before modification to prevent side effects.

### Value Object
PerturbationReport is a frozen dataclass — immutable result container
produced after every attack.

## Gradient Flow
