"""
COMPASS-PRO (parallel implementation)
=====================================

Author: Dibyendu Mondal

COMPASS-PRO (Correlation-energy-based Operator-Pool Adaptive Screening and
Selection, with Products of Rank-Ordered Operators) is a chemistry-inspired,
energy-driven ansatz-construction algorithm for the Variational Quantum
Eigensolver (VQE).

Instead of a fixed, chemistry-agnostic ansatz (e.g. full UCCSD), the ansatz is
built operator-by-operator, guided at every step by the correlation energy
each candidate excitation operator actually recovers:

  1.  Build the molecular problem (PySCF + frozen core) and map the fermionic
      Hamiltonian to qubits (Jordan-Wigner).
  2.  Generate the pool of double (T2) excitations and screen each one alone
      against the Hartree-Fock reference, keeping those that lower the energy
      by more than a numerical-noise threshold.
  3.  Sort the surviving T2 operators by the correlation energy they recover.
  4.  For every T2 that touches an "active space" orbital, test which single
      (S) hole/particle excitations couple to it and lower the energy further.
      Each kept (T2 + S) pair is relabelled as an implicit triple (T3).
  5.  Resolve T3 labels shared by several batches (keep the copy with the
      largest energy contribution), then append the plain single excitations
      as their own one-operator batches.
  6.  Adaptively grow the final ansatz (ADAPT-VQE style) from those batches.


"""

import os
import warnings

import numpy as np
from joblib import Parallel, delayed

from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Number of worker processes (-1 = all cores, 1 = run serially in-process).
N_JOBS = int(os.environ.get("COMPASS_N_JOBS", -1))

# Molecule (example: BeH2) and the active-space orbitals that trigger the
# T2 -> coupled-S/T3 screening in step 8.
COORD = "Be 0.0 0.0 0.0; H 0.0 0.0 3.0; H 0.0 0.0 -3.0"
ACTIVE_SPACE = [1, 2, 7, 8]


# =============================================================================
# WORKER SIDE: one VQE = one parallel task
# =============================================================================
def _get_estimator():
    """Exact (shot-free) statevector Estimator (cheap to build, one per VQE job).

    ``max_parallel_threads=1`` stops each Aer instance from spawning its own
    OpenMP threads, which would oversubscribe the cores when many workers run
    at once. (At 12 qubits Aer is single-threaded anyway, so results are not
    affected.)
    """
    return AerEstimator(
        backend_options={"method": "statevector", "max_parallel_threads": 1},
        run_options={"seed": None, "shots": None},
        approximation=True,
    )


def _vqe_job(qubit_op, init_state, ops, x0, ref=None):
    """Run ONE VQE and return ``(electronic_energy, optimal_point)``.

    Parameters
    ----------
    ops : operator, or list of operators, forming the trial ansatz
          ``prod_k exp(theta_k * op_k)`` applied to the reference state.
    x0  : starting parameters for the optimiser.
    ref : ``None`` -> reference is the Hartree-Fock state.
          ``(pauli_list, params)`` -> reference is the already-grown ansatz
          with ``params`` frozen (used in the adaptive growth rounds).

    The returned energy excludes the nuclear-repulsion/frozen-core ``shift``;
    the caller adds it (exactly as the serial script does).
    """
    if ref is None:
        state = init_state
    else:
        state = EvolvedOperatorAnsatz(ref[0], initial_state=init_state).assign_parameters(ref[1])
    ansatz = EvolvedOperatorAnsatz(ops, initial_state=state)
    optimizer = L_BFGS_B(maxfun=200000, maxiter=10000)
    vqe = VQE(_get_estimator(), ansatz, optimizer=optimizer, initial_point=x0)
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    return np.real(result.eigenvalue), result.optimal_point


def run_vqes(qubit_op, init_state, jobs):
    """Run a list of independent VQE jobs in parallel; results keep job order.

    Each job is a tuple ``(ops, x0)`` or ``(ops, x0, ref)``.
    """
    return Parallel(n_jobs=N_JOBS)(
        delayed(_vqe_job)(qubit_op, init_state, *job) for job in jobs
    )


# =============================================================================
# HAND-PICKED SINGLE (S) EXCITATION POOLS TIED TO THE ACTIVE SPACE
# =============================================================================
# Sh_ex: occ -> unocc and occ -> occ(active);  Sp_ex: occ -> unocc and unocc(active) -> unocc
def Sh_ex(num_spatial_orbitals, num_particles):
    """Hole-type S excitations."""
    return [((6, 0), (8, 1)), ((6, 0), (9, 1)), ((6, 0), (10, 1)), ((6, 0), (11, 1)),
            ((7, 0), (8, 1)), ((7, 0), (9, 1)), ((7, 0), (10, 1)), ((7, 0), (11, 1)),
            ((0, 6), (2, 7)), ((0, 6), (3, 7)), ((0, 6), (4, 7)), ((0, 6), (5, 7)),
            ((1, 6), (2, 7)), ((1, 6), (3, 7)), ((1, 6), (4, 7)), ((1, 6), (5, 7))]


def Sp_ex(num_spatial_orbitals, num_particles):
    """Particle-type S excitations."""
    return ([((o, 2), (v, c)) for o in (6, 7) for v in (8, 9, 10, 11) for c in (3, 4, 5)]
            # ordering above: (6,2) x [8:(3,4,5), 9:(3,4,5), ...], then (7,2) x ...
            + [((o, 8), (v, c)) for o in (0, 1) for v in (2, 3, 4, 5) for c in (9, 10, 11)])


def build_excitations(num_spatial_orbital, num_particles, mapper, init_state, excitations):
    """Return (excitation index list, matching qubit-operator list) for a UCC pool."""
    ucc = UCC(num_spatial_orbital, num_particles, qubit_mapper=mapper,
              initial_state=init_state, excitations=excitations)
    index_list = ucc._get_excitation_list()
    pauli_list = [mapper.map(ex) for ex in ucc.excitation_ops()]
    return index_list, pauli_list


# =============================================================================
# MAIN WORKFLOW
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1-2. Molecular problem and qubit Hamiltonian (serial, one-off cost)
    # -------------------------------------------------------------------------
    driver = PySCFDriver(atom=COORD, charge=0, spin=0, basis="sto3g")
    molecule = driver.run()
    es_problem = FreezeCoreTransformer(freeze_core=True).transform(molecule)

    mapper = JordanWignerMapper()
    qubit_op = mapper.map(es_problem.second_q_ops()[0])
    num_particles = es_problem.num_particles
    num_spin_orbitals = es_problem.num_spin_orbitals
    num_spatial_orbital = es_problem.num_spatial_orbitals
    nuclear_repulsion_energy = es_problem.nuclear_repulsion_energy
    energy_shift = es_problem.hamiltonian.constants["FreezeCoreTransformer"]
    shift = nuclear_repulsion_energy + energy_shift

    print("Number Of Particles: ", num_particles)
    print("Number of Spin Orbitals: ", num_spin_orbitals)
    print("Nuclear Repulsion Energy: ", nuclear_repulsion_energy)
    print("Energy Shift: ", shift)

    # -------------------------------------------------------------------------
    # 3. Hartree-Fock reference state
    # -------------------------------------------------------------------------
    init_state = HartreeFock(num_spatial_orbital, num_particles, mapper)

    # -------------------------------------------------------------------------
    # 4. T2 (double excitation) pool and the Hartree-Fock energy
    # -------------------------------------------------------------------------
    excitation_list5, excitation_list_pauli5 = build_excitations(
        num_spatial_orbital, num_particles, mapper, init_state, "d")

    # <HF|H|HF> through the same ansatz machinery: exp(0 * T2)|HF> = |HF>.
    ansatz_hf = EvolvedOperatorAnsatz(excitation_list_pauli5[0],
                                      initial_state=init_state).assign_parameters([0.0])
    HF_energy = _get_estimator().run(ansatz_hf, qubit_op).result().values[0] + shift
    print("HF", HF_energy)

    # -------------------------------------------------------------------------
    # 5. T2 screening -- PARALLEL over all candidate T2 operators
    #    Keep a T2 only if it lowers the energy below HF by more than 1e-5 Ha.
    # -------------------------------------------------------------------------
    start = [0.0]
    results = run_vqes(qubit_op, init_state, [(op, start) for op in excitation_list_pauli5])

    energy_list = []            # standalone energies of surviving T2s
    optimal_list = []           # their optimised thetas
    excitation_list = []        # their index tuples
    excitation_list_pauli = []  # their qubit operators
    for i, (e_elec, op_pt) in enumerate(results):
        E1 = e_elec + shift
        if abs(HF_energy - E1) > 1e-5:
            energy_list.append(E1)
            optimal_list.append(op_pt[0])
            excitation_list.append(excitation_list5[i])
            excitation_list_pauli.append(excitation_list_pauli5[i])
        print("state and average value is", excitation_list5[i], E1)

    # -------------------------------------------------------------------------
    # 6. S-excitation pools (hole and particle type)
    # -------------------------------------------------------------------------
    excitation_list1, excitation_list_pauli1 = build_excitations(
        num_spatial_orbital, num_particles, mapper, init_state, Sh_ex)
    excitation_list2, excitation_list_pauli2 = build_excitations(
        num_spatial_orbital, num_particles, mapper, init_state, Sp_ex)

    # -------------------------------------------------------------------------
    # 7. Sort surviving T2s by standalone energy (lowest first).
    #    The serial script's "match and zero out" loop is exactly a STABLE sort
    #    (ties keep their original relative order), so we use one directly.
    # -------------------------------------------------------------------------
    order = sorted(range(len(energy_list)), key=lambda j: energy_list[j])
    energy_list1 = [energy_list[j] for j in order]
    order_excitation_list1 = [excitation_list[j] for j in order]
    excitation_list_pauli = [excitation_list_pauli[j] for j in order]
    t_op1 = [optimal_list[j] for j in order]

    # ((h0,h1),(p0,p1))  ->  flat [h0, p0, h1, p1] for index comparisons
    order_excitation_list = [[ex[0][0], ex[1][0], ex[0][1], ex[1][1]]
                             for ex in order_excitation_list1]
    print(order_excitation_list)
    print("no of Sorted T2 is", len(order_excitation_list))

    # -------------------------------------------------------------------------
    # 8. Coupled T2 + S screening -> implicit T3 construction
    # -------------------------------------------------------------------------
    # 8a. Enumerate every (T2, S) pair the serial script would evaluate and
    #     collect the unique ones. A pair is needed when the S operator's
    #     contraction index equals an active orbital present in the T2.
    pair_jobs = {}  # (i, 'h'|'p', j) -> (ops, x0); dict keeps first-seen order
    for i in range(len(excitation_list_pauli)):
        t_list = order_excitation_list[i]
        for k in range(len(t_list)):
            for l in range(4):
                if t_list[k] == ACTIVE_SPACE[l]:
                    act_orbital = ACTIVE_SPACE[l]
                    for j in range(len(excitation_list1)):
                        if excitation_list1[j][1][1] == act_orbital:
                            pair_jobs.setdefault(
                                (i, "h", j),
                                ([excitation_list_pauli[i], excitation_list_pauli1[j]],
                                 np.array([t_op1[i], 0.0], dtype=np.float64)))
                    for j in range(len(excitation_list2)):
                        if excitation_list2[j][0][1] == act_orbital:
                            pair_jobs.setdefault(
                                (i, "p", j),
                                ([excitation_list_pauli[i], excitation_list_pauli2[j]],
                                 np.array([t_op1[i], 0.0], dtype=np.float64)))

    # 8b. Run all unique pair VQEs in parallel.
    pair_results = dict(zip(pair_jobs, run_vqes(qubit_op, init_state, list(pair_jobs.values()))))

    # 8c. Serial assembly, in the original loop order, using the pre-computed
    #     energies. An S is kept if it improves on the T2-only energy by > 1e-6 Ha.
    batch_T2_operators = []
    batch_opt_pt = []
    batch_pauli_operators = []
    T3_list = []
    t3_energy = []
    t3_h = t3_p = None  # NOTE: like the original, these persist between iterations

    for i in range(len(excitation_list_pauli)):
        batch = [order_excitation_list1[i]]
        batch_pauli = [excitation_list_pauli[i]]
        batch_t2 = [t_op1[i]]
        batch_t3 = [order_excitation_list1[i]]
        batch_energy = [0.0]
        new_excitation_S_list = []
        t_list = order_excitation_list[i]
        E2 = energy_list1[i]           # T2-only energy
        T2 = order_excitation_list1[i]

        for k in range(len(t_list)):
            for l in range(4):
                if t_list[k] == ACTIVE_SPACE[l]:
                    act_orbital = ACTIVE_SPACE[l]

                    # ---- S hole operators
                    for j in range(len(excitation_list1)):
                        temp_sh = excitation_list1[j]
                        if temp_sh[1][1] == act_orbital:
                            e_elec, optimal_point = pair_results[(i, "h", j)]
                            E1 = e_elec + shift
                            if abs(E2 - E1) > 1e-6:
                                new_excitation_S_list.append(temp_sh)
                                batch.append(temp_sh)
                                batch_t2.append(optimal_point[1])
                                batch_pauli.append(excitation_list_pauli1[j])
                                if temp_sh[1][1] == T2[0][0]:
                                    t3_h = [temp_sh[0][0], temp_sh[0][1], T2[0][1]]
                                    t3_p = [temp_sh[1][0], T2[1][0], T2[1][1]]
                                if temp_sh[1][1] == T2[0][1]:
                                    t3_h = [temp_sh[0][0], temp_sh[0][1], T2[0][0]]
                                    t3_p = [temp_sh[1][0], T2[1][1], T2[1][0]]
                                t3_h.sort()
                                t3_p.sort()
                                batch_t3.append((tuple(t3_h), tuple(t3_p)))
                                batch_energy.append(abs(E2 - E1))

                    # ---- S particle operators
                    for j in range(len(excitation_list2)):
                        temp_sp = excitation_list2[j]
                        if temp_sp[0][1] == act_orbital:
                            e_elec, optimal_point = pair_results[(i, "p", j)]
                            E1 = e_elec + shift
                            if abs(E2 - E1) > 1e-6:
                                new_excitation_S_list.append(temp_sp)
                                batch.append(temp_sp)
                                batch_t2.append(optimal_point[1])
                                batch_pauli.append(excitation_list_pauli2[j])
                                if temp_sp[0][1] == T2[1][0]:
                                    t3_h = [temp_sp[0][0], T2[0][0], T2[0][1]]
                                    t3_p = [temp_sp[1][0], temp_sp[1][1], T2[1][1]]
                                if temp_sp[0][1] == T2[1][1]:
                                    t3_h = [temp_sp[0][0], T2[0][1], T2[0][0]]
                                    t3_p = [temp_sp[1][0], temp_sp[1][1], T2[1][0]]
                                t3_h.sort()
                                t3_p.sort()
                                batch_t3.append((tuple(t3_h), tuple(t3_p)))
                                batch_energy.append(abs(E2 - E1))

        print("T operator", order_excitation_list[i], "Corresponding S", new_excitation_S_list)
        batch_T2_operators.append(batch)
        batch_opt_pt.append(batch_t2)
        batch_pauli_operators.append(batch_pauli)
        T3_list.append(batch_t3)
        t3_energy.append(batch_energy)

    print("Batch", batch_T2_operators)
    print("Batch T2 Opt", batch_opt_pt)
    print("T3", T3_list)
    print("Energy", t3_energy)

    # -------------------------------------------------------------------------
    # 9. De-duplicate shared T3 labels across batches ("New Batch" step).
    #    Cheap bookkeeping (no VQEs) -> kept serial and identical.
    #    A T3 beyond the first entry of a batch stays only if its energy there
    #    is strictly larger than in every other batch that holds the same T3.
    # -------------------------------------------------------------------------
    new_batch_pauli = []
    new_batch_T2 = []
    for i in range(len(T3_list)):
        batch_no = T3_list[i]
        batch_pauli = [batch_pauli_operators[i][0]]
        T2_batch = [batch_T2_operators[i][0]]
        if len(batch_no) > 1:
            for j in range(1, len(batch_no)):
                T3 = batch_no[j]
                ee = t3_energy[i][j]
                print("E", ee)
                e_max = []
                for k in range(len(T3_list)):
                    for kk in range(len(T3_list[k])):
                        if k != i and T3 == T3_list[k][kk]:
                            print("yes")
                            print(k, kk)
                            e_max.append(t3_energy[k][kk])
                print(e_max)
                if len(e_max) == 0:
                    e_max.append(0.0)
                new_max = max(e_max)
                print("new_max", new_max)
                if ee > new_max:
                    batch_pauli.append(batch_pauli_operators[i][j])
                    T2_batch.append(batch_T2_operators[i][j])
        new_batch_T2.append(T2_batch)
        new_batch_pauli.append(batch_pauli)
    print("New Batch", new_batch_T2)
    print("--------------------------------------------------------")

    # -------------------------------------------------------------------------
    # 10. Plain single excitations become their own one-operator batches
    # -------------------------------------------------------------------------
    excitation_list20, excitation_list_pauli20 = build_excitations(
        num_spatial_orbital, num_particles, mapper, init_state, "s")

    batch_pauli_operators = new_batch_pauli.copy()
    for i in range(len(excitation_list_pauli20)):
        batch_pauli_operators.append([excitation_list_pauli20[i]])
        batch_T2_operators.append(excitation_list20[i])
        batch_opt_pt.append([0.0])

    f_e_list = []

    # -------------------------------------------------------------------------
    # 11. Adaptive ansatz growth
    # -------------------------------------------------------------------------
    # Round 0 -- PARALLEL: every batch on top of bare HF; lowest energy seeds
    # the ansatz. (min + list.index = "first batch with the lowest energy",
    # the same choice the serial script makes.)
    results = run_vqes(qubit_op, init_state,
                       [(list(b), [0.0] * len(b)) for b in batch_pauli_operators])
    energy_list = []
    point_list = []
    for i, (E, opt_pt) in enumerate(results):
        print(batch_T2_operators[i], E + shift)
        print(opt_pt)
        energy_list.append(E + shift)
        point_list.append(list(opt_pt))

    best = energy_list.index(min(energy_list))
    grow_pauli = list(batch_pauli_operators[best])
    grow_point = list(point_list[best])
    E_opt = min(energy_list)
    par_list = [len(grow_pauli)]
    f_e_list.append(E_opt)
    print("Energy", E_opt)
    print(grow_point)
    print("_____________________________________________________________________")

    # Later rounds: freeze the current ansatz as the reference, try each batch
    # on top of it (PARALLEL), append the winner, then re-optimise ALL
    # parameters of the grown ansatz (serial -- it needs the winner).
    # Stop when the energy changes by < 1e-7 Ha between rounds.
    e_old = 0.0
    while True:
        ref = (list(grow_pauli), list(grow_point))
        results = run_vqes(qubit_op, init_state,
                           [(list(b), [0.0] * len(b), ref) for b in batch_pauli_operators])
        energy_list = []
        point_list = []
        for E, opt_pt in results:
            print(E + shift)
            energy_list.append(E + shift)
            point_list.append(list(opt_pt))

        energy_sorted = sorted(energy_list)
        print("Energy list", energy_sorted)

        best = energy_list.index(energy_sorted[0])
        grow_point2 = list(grow_point) + point_list[best]
        grow_pauli.extend(batch_pauli_operators[best])

        e_elec, optimal_point = _vqe_job(qubit_op, init_state, grow_pauli, grow_point2)
        E_opt = e_elec + shift
        print("Energy", E_opt)
        if abs(e_old - E_opt) < 1e-7:
            break
        e_old = E_opt
        f_e_list.append(E_opt)
        par_list.append(len(grow_pauli))

        print("******************************************************************************")
        grow_point = optimal_point

    # -------------------------------------------------------------------------
    # 12. Final report
    # -------------------------------------------------------------------------
    # f_e_list: energy after each accepted growth round.
    # par_list: cumulative number of ansatz parameters after each round.
    print("Final Energy List")
    print(f_e_list)
    print("par_list", par_list)


if __name__ == "__main__":
    main()
