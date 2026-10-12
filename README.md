# COMPASS-PRO

**COMPASS with Progressive block ReOrdering** — a dynamic, energy-driven *Ansatz*
construction method for the Variational Quantum Eigensolver (VQE).


This repository contains a parallel implementation of COMPASS-PRO and its
workflow, for the algorithm introduced in:

> D. Mondal, D. Mukherjee and R. Maitra, *Operator commutativity screening and
> progressive operator block reordering toward many-body inspired quantum state
> preparation*, J. Chem. Phys. **164**, 074113 (2026).
> https://doi.org/10.1063/5.0307670

## Motivation

VQE needs an *Ansatz* that is expressive enough to capture electron correlation
but compact enough for noisy (NISQ) hardware, and whose optimization landscape
does not trap the optimizer in local minima. Fixed UCC *Ansätze* are deep;
gradient-based adaptive methods such as ADAPT-VQE can stall on "gradient
troughs" near degeneracies, where operator gradients decay exponentially.

## The idea

1. **Operator blocks.** Dominant two-body (T2) excitation operators are
   selected by a one-parameter energy optimisation and an energy threshold.
   Each T2 is placed in its own *operator block*.
2. **Scatterers give higher-rank effects cheaply.** Each block is expanded with
   *scattering* operators (single excitations with one quasi-destruction
   operator) that do not commute with the T2. Their commutators generate
   effective triple (T3) excitations, so higher-order correlation is captured
   using only low-rank operators and hence fewer quantum gates.
3. **Redundancy removal.** If several blocks generate the same effective T3,
   it is kept only in the block where it stabilises the energy the most.
4. **Progressive, energy-guided growth.** Starting from the Hartree-Fock state,
   every candidate block is tried in a local VQE micro-cycle (previous
   parameters frozen), the block giving the largest energy lowering is appended
   to the *Ansatz*, and a global VQE macro-cycle re-optimises all parameters.
   This repeats until the energy changes by less than 1e-7 Ha between cycles.

Selection by energy stabilization (rather than by gradient) steers the
optimization along the steepest "burrowing" direction, which lets the method
bypass local traps and reach the ground state even in strongly correlated
regions.
