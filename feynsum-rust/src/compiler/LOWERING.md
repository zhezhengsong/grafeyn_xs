# Lowering GraFeyn kernels to the kernel IR

This note describes step 2 of the kernel compiler (the project plan's "intermediate representation"): how each
kernel produced by the gate scheduler is translated into a short list of primitive operations. Lowering
translates only; it does not optimize. Optimization passes are the next step.

Code: `src/compiler/ir.rs` (the IR and its reference semantics), `src/compiler/lower.rs` (the lowering),
`src/compiler/check.rs` (differential checking against the interpreter), `src/compiler/mod.rs` (`--dump-ir`).

## 1. Context

GraFeyn simulates a circuit by walking Feynman paths. A state is a set of nonzero amplitudes, each a basis
state `b` (one bit per qubit) with a complex weight `w`. The scheduler groups consecutive gates into kernels.
For every input amplitude, the simulator applies the kernel's gates one by one. A non-branching gate maps
`(b, w)` to one new pair; a branching gate (for example `h`) splits the path in two. At the end of the
kernel the resulting leaves are added into the hash table that holds the next state.

The interpreter (`state_expander::apply_gates1`) does this by dispatching on the gate kind for every gate of
every path, through a closure per gate. The scheduler (`gfq-fused`) decides kernels from the gates' qubits
and branching flags alone, never from the state, so the whole kernel sequence is known before simulation
starts. Each kernel can therefore be translated ahead of time.

## 2. The IR

A lowered kernel is the list of its source gate indices plus a list of operations on one path `(b, w)`:

| op | effect on one path `(b, w)` |
|---|---|
| `Flip { controls, targets }` | if every control bit is 1, flip every target bit |
| `Swap { controls, a, b }` | if every control bit is 1, exchange bits `a` and `b` |
| `Phase { controls, qubit, c0, c1 }` | if every control bit is 1, multiply `w` by `c1` when `b[qubit] = 1`, else by `c0` |
| `Branch { qubit, m }` | with `j = b[qubit]`, replace the path by `(b[qubit := 0], w * m[0][j])` and `(b[qubit := 1], w * m[1][j])`; `m` is the gate's 2x2 matrix, the input bit selects the column |
| `Prune` | drop paths whose weight is zero (both parts below 1e-8) |

One input path yields at most `2^(number of Branch ops)` leaves. The reference semantics is `ir::eval`, which
runs the ops over a list of paths and finally drops zero-weight leaves. Two kernels are equivalent when, for
every input basis state, the sums of their leaves per output basis state agree.

Design choices:

- Ops name qubits by index, so the IR does not depend on the width of the basis index (`BasisIdx64` or
  `BasisIdxUnlimited`). A backend turns index lists into machine-word masks.
- Constants are `f64`. Gate parameters arrive as `f32` (the simulator's precision). Folding phases in `f64` at
  compile time and rounding once at code generation loses less precision than the interpreter does.
- Whether a single-qubit unitary branches is decided exactly as in the interpreter (section 3.2), so a
  lowered kernel has the number of branches the scheduler assumed when it formed the kernel.

## 3. The algorithm

### 3.1 Per kernel

```
lower_kernel(circuit, gate_indices):
    ops = []
    for gi in gate_indices:                # kernel order
        gate_ops = lower_gate(circuit.gates[gi])
        if gate_ops is None:               # gate kind without a lowering
            return Unsupported(gi, name)   # the kernel stays on the interpreter
        ops.extend(gate_ops)
    return Kernel(gates = gate_indices, ops = ops)
```

Lowering is local: each gate is translated on its own, and the ops are concatenated in gate order. Every case
of `lower_gate` mirrors the gate's `push_apply` in `circuit/gate.rs`, so the lowered kernel computes the same
leaves as the interpreter, gate by gate.

### 3.2 Per gate

Notation: `phase q ? c1 : c0` multiplies by `c1` if bit `q` is 1 and by `c0` otherwise (this is also how
`--dump-ir` prints it); `cis(x) = e^(i x)`; `r = 1/sqrt(2)`.

| gate | ops |
|---|---|
| `x q` | `flip q` |
| `cx c, t` | `flip t if c` |
| `ccx c1, c2, t` | `flip t if c1 & c2` |
| `swap a, b` | `swap a, b` |
| `cswap c, a, b` | `swap a, b if c` |
| `y q` | `phase q ? -i : i`, then `flip q` (Y\|0> = i\|1>, Y\|1> = -i\|0>: the factor depends on the bit before the flip) |
| `z q` | `phase q ? -1 : 1` |
| `s q` / `sdg q` | `phase q ? i : 1` / `phase q ? -i : 1` |
| `t q` / `tdg q` | `phase q ? cis(pi/4) : 1` / `phase q ? cis(-pi/4) : 1` |
| `phase(l) q` (also `p`) | `phase q ? cis(l) : 1` |
| `rz(t) q` | `phase q ? cis(t/2) : cis(-t/2)` |
| `cz c, t` | `phase t ? -1 : 1 if c` |
| `cphase(l) c, t` (also `cp`) | `phase t ? cis(l) : 1 if c` |
| `h q` | `branch q [[r, r], [r, -r]]`, `prune` |
| `ry(t) q` | `branch q [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]`, `prune` |
| `sx q` / `sxdg q` | `branch q [[p, m], [m, p]]` / `branch q [[m, p], [p, m]]` with `p = (1+i)/2`, `m = (1-i)/2`, then `prune` |
| `rx(t) q` | single-qubit rule below, with `M = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]` |
| `u(theta, phi, lambda) q` (also `u3`, and `u1`, `u2`, which the parser turns into `u`) | single-qubit rule below, with `M = [[cos(theta/2), -cis(lambda) sin(theta/2)], [cis(phi) sin(theta/2), cis(phi+lambda) cos(theta/2)]]` |
| `fsim`, unknown gates | no lowering (`Unsupported`) |

**Single-qubit rule** for a general unitary `M = [[a, b], [c, d]]` (`rx`, `u`). It follows the interpreter's
`single_qubit_unitary_push`:

1. Read the interpreter's own `f32` matrix of the gate and test entries for zero with its threshold
   (`|x| < 1e-8`).
2. If `a` and `d` are zero (anti-diagonal): `|0> -> c|1>`, `|1> -> b|0>`, so emit `phase q ? b : c`, then
   `flip q`.
3. Otherwise, if `b` and `c` are zero (diagonal, for example `u1` or `rx(0)`): emit `phase q ? d : a`.
4. Otherwise: emit `branch q M`, `prune`.

The constants themselves are computed in `f64`; only the zero test uses the `f32` matrix. Recomputing the zero
pattern in `f64` could classify a nearly-zero entry differently from the interpreter. That would change the
number of branches in a kernel and break the scheduler's assumption (gfq-fused closes a kernel after two
branching gates).

**Prune after every branch.** The interpreter tests the weight for zero before applying each gate
(`apply_gates1` returns early on a zero weight). A branch creates a zero weight when its matrix has a zero entry,
for example `ry(0)`, whose matrix is the identity. The `Prune` op makes that test explicit at the same
points. Interference between different paths, as in `h` followed by `h`, does not produce zero weights on
individual paths; it cancels only when the leaves are summed into the next state.

**Branching gates always branch.** `h`, `ry`, `sx` and `sxdg` lower to a `Branch` even when the angle makes
the matrix diagonal, because their `push_apply` always branches. The zero leaf is removed by the following
`Prune`.

## 4. Correctness checking

`--check-ir` lowers every kernel of a circuit and compares it with the interpreter (`check::check_circuit`):

- Let `k` be the number of qubits the kernel touches. If `k <= 12`, all `2^k` assignments of those bits are
  tried. Otherwise 512 random assignments are used, or 64 for kernels longer than 10000 gates.
- Each assignment is embedded in two backgrounds for the untouched qubits: all zeros and a random pattern.
  This catches a lowering that touches the wrong qubit.
- The input weight is `0.6 + 0.8i`. The interpreter side (`check::interpret`) makes the same `push_apply` calls
  with the same zero pruning as `apply_gates1`, written as a loop instead of a recursion. It therefore also
  handles the 57237-gate kernel of multiplier_n400, on which the recursive interpreter overflows its stack.
- Leaves are summed per output basis state on both sides. Basis states are compared by their bits, because
  `BasisIdxUnlimited` equality also compares vector lengths. The check fails if any summed weight differs by
  more than `1e-5`, or if a kernel produces more leaves than `2^(number of Branch ops)`.

Results on the 10 circuits of experiment 1 (`--scheduler gfq-fused --check-ir`, 2026-11-10):

| circuit | kernels | lowered | exhaustive / sampled | inputs | max error | max leaves per input |
|---|---|---|---|---|---|---|
| adder_n28 | 1 | 1 | 0 / 1 | 1024 | 2.7e-8 | 1 |
| adder_n433 | 1 | 1 | 0 / 1 | 1024 | 2.7e-8 | 1 |
| multiplier_n45 | 1 | 1 | 0 / 1 | 1024 | 2.7e-8 | 1 |
| multiplier_n400 | 1 | 1 | 0 / 1 | 128 | 2.7e-8 | 1 |
| bv_n280 | 280 | 280 | 280 / 0 | 3444 | 2.7e-8 | 4 |
| qram_n20 | 1 | 1 | 0 / 1 | 1024 | 2.7e-8 | 1 |
| qft_n18 | 9 | 9 | 5 / 4 | 9552 | 4.9e-7 | 4 |
| ising_n26 | 40 | 40 | 40 / 0 | 700 | 2.3e-7 | 4 |
| knn_n25 | 13 | 13 | 13 / 0 | 376 | 9.1e-8 | 4 |
| dnn_n16 | 625 | 625 | 625 / 0 | 4088 | 2.0e-7 | 4 |

All 972 kernels lower, none is unsupported, and every check passes (tolerance 1e-5). The errors are at the
level of `f32` rounding in the interpreter.

Unit tests (`cargo test compiler`, 4 tests, all passing):

- random circuits over every gate kind that has a lowering, with angles drawn from {0, pi/2, pi, random}, so
  that diagonal, anti-diagonal and degenerate matrices (`u1`, `u3(pi, ...)`, `rx(0)`, `ry(0)`) all occur; run
  with both `BasisIdx64` and `BasisIdxUnlimited`;
- a deliberately wrong lowering is caught by the checker;
- `eval` on `h` followed by `cx`.

## 5. Example

Kernel 1 of ising_n26. Its source gates 27 to 31 are one ZZ-interaction block on qubits 0 and 1, and gates
176 to 178 are the transverse-field step on qubit 0, whose angle is 0 in this circuit. Output of `--dump-ir`,
with comments added:

```
#1 kernel gates=[27, 28, 29, 30, 31, 176, 177, 178] k=2 max_leaves=4
  phase q1 ? 0.6402+0.7682i : 0.6402-0.7682i       # rz(1.7521) q1
  phase q1 ? 0.6402+0.7682i : 0.6402-0.7682i       # rz(1.7521) q1
  flip q1 if q0                                    # cx q0, q1
  phase q1 ? 0.6402-0.7682i : 0.6402+0.7682i       # rz(-1.7521) q1
  flip q1 if q0                                    # cx q0, q1
  branch q0 [[0.7071, 0.7071], [0.7071, -0.7071]]  # h q0
  prune
  phase q0 ? 1.0000 : 1.0000                       # rz(0) q0
  branch q0 [[0.7071, 0.7071], [0.7071, -0.7071]]  # h q0
  prune
```

The example also shows what lowering leaves to the passes. The two phases on `q1` could be merged into one,
`phase ? 1 : 1` is the identity, and `h · rz(0) · h` is the identity as well. As lowered, though, it still
produces two branches and up to four leaves per input path.

## 6. Usage

```bash
cargo build --release
./target/release/feynsum-rust -i CIRCUIT.qasm --scheduler gfq-fused --dump-ir out.ir   # text IR, no simulation
./target/release/feynsum-rust -i CIRCUIT.qasm --scheduler gfq-fused --check-ir         # differential check
cargo test compiler                                                                    # unit tests
```

## 7. Not done yet (next step: passes)

- Merge consecutive flips into one mask. Commute phase multiplications past flips and merge them. Drop
  identity phases.
- Once a branch fixes a qubit's value on each leaf, fold later conditions on that qubit into constants.
  Merge leaves that end in the same flip pattern, so that for example `h · rz(0) · h` collapses to one leaf.
- Bring every kernel to the fixed leaf shape `b' = b XOR flip[...]`, `w' = w * phase[...]`. For kernels on a
  small number `k` of qubits this is a table of `2^k` entries. Large kernels (the arithmetic circuits, with `k`
  up to 433) need straight-line bit operations instead, because the table would be too big.
- `fsim` and unknown gates have no lowering; kernels that contain them stay on the interpreter.
