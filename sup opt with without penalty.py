### Superstructure Optimization
import numpy as np
import sys
import time
from itertools import combinations

try:
    from pyomo.environ import (
        ConcreteModel, Set, Param, Var, Objective, Constraint,
        Binary, NonNegativeReals, minimize, value, SolverFactory)
except ImportError:
    sys.exit(1)

# OCFE setup
# We discretize each membrane element using Gauss-Lobatto-Legendre (GLL) points or Hahn polynomial roots.
# These are optimal for polynomial collocation because they include both endpoints
# and cluster points near the boundaries where gradients are steepest.
# The 5-point GLL set on [-1,1] maps to [0,1] via s = (p+1)/2.
#
# The differentiation matrix D encodes the Lagrange polynomial derivatives:
#   D[j,k] = d(phi_k)/ds evaluated at s_j
# where phi_k is the k-th Lagrange basis polynomial.
# This turns the membrane ODE (dF/dA = -J) into an algebraic system the
# optimizer can handle directly — no time-stepping needed inside the solver.

def build_ocfe_matrices():
    p = np.array([-1, -np.sqrt(3/7), 0, np.sqrt(3/7), 1], dtype=float)
    s = (p + 1) / 2
    # Alternatively, we can also try Hahn polynomials. Both are fine.
    #p = np.array([0, 0.40585, 0.74153, 0.94911, 1])

    n = len(s)
    D = np.zeros((n, n))

    for j in range(n):
        for k in range(n):
            if j == k:
                D[j, k] = sum(1.0 / (s[k] - s[m]) for m in range(n) if m != k)
            else:
                num = 1.0
                den = 1.0
                for m in range(n):
                    if m not in (j, k):
                        num *= s[j] - s[m]
                    if m != k:
                        den *= s[k] - s[m]
                D[j, k] = num / den

    return s, D


S, DM = build_ocfe_matrices()

# Discretization parameters
NEL = 2    # finite elements per module (2 is usually enough, more = slower)
NCP = 4    # interior collocation points per element
NPT = 5    # total points per element including boundaries (= NCP + 1)
NMOD = 5   # max modules in superstructure (slots 1..5)

# Physical constants and problem data
# Permeability in consistent units (kmol / (m2 h bar))
# Starting from GPU (gas permeation unit): 1 GPU = 3.35e-10 mol/(m2 s Pa)
# Convert: x 1e-3 (mol->kmol) x 3600 (s->h) x 1e5 (Pa->bar)
GPU = 3.35e-10 * 1e-3 * 3600 * 1e5

# Component indices: 1=N2, 2=O2
# O2/N2 selectivity = 10 (typical for many polymeric membranes)
# So O2 permeates ~10x faster, N2 stays in retentate -> N2 product
PI = {1: 160 * GPU,   # N2 permeability
      2: 1600 * GPU}  # O2 permeability (10x higher)

PR = 10.0  # retentate (feed) side pressure, bar
PP = 1.0   # permeate side pressure, bar

# Feed specifications
FT = 2300.0               # total feed flow, kmol/h
XF = {1: 0.79, 2: 0.21}  # mole fractions (air: 79% N2, 21% O2)
FF = {i: FT * XF[i] for i in (1, 2)}  # component feed flows: N2=1817, O2=483 kmol/h

# Cost parameters (these drive the objective)
CA  = 20.0    # membrane area cost, $/m2/h
PHI = 1000  # fixed cost per active module, $/h, set zero for no penalty

# Bounds
AMAX = 5000.0  # max area module, m2 (physical constraint)
AMIN = 100.0   # min area if module is active (avoid degenerate solutions)
FMAX = FT * 5  # generous upper bound on any flow variable
EPS  = 1e-4    # small number to avoid division by zero

# Big-M value for flux/mass-balance deactivation.
# Must exceed the largest LHS magnitude that any active-module constraint
# can produce.  50 (max flux) * AMAX/NEL ≈ 137 500; round up generously.
BIGM_FLUX = 60.0           # per-component flux can't exceed ~50
BIGM_MB   = 200000.0       # |D*F| + |J*dA| upper bound

# Pyomo model

def build_model():
    m = ConcreteModel("AirSep")

    # Index sets
    m.M  = Set(initialize=range(1, NMOD + 1))
    m.I  = Set(initialize=[1, 2])
    m.L  = Set(initialize=range(1, NEL + 1))
    m.J  = Set(initialize=range(NPT))
    m.Ji = Set(initialize=range(1, NPT))
    m.NR = Set(initialize=[1])
    m.PR = Set(initialize=[1])
    m.PS = Set(initialize=[1])

    # Parameters
    m.Pi = Param(m.I, initialize=PI)
    m.Ff = Param(m.I, initialize=FF)
    m.sp = Param(m.J, initialize={j: float(S[j]) for j in range(NPT)})
    m.Dm = Param(m.J, m.J, initialize={(j, k): float(DM[j, k])
                                        for j in range(NPT)
                                        for k in range(NPT)})

    # Topology binaries
    m.y   = Var(m.M, within=Binary)
    m.zIR = Var(m.NR, m.M, within=Binary)
    m.zRR = Var(m.M, m.M, within=Binary)
    m.zSR = Var(m.M, m.M, within=Binary)
    m.zRP = Var(m.M, m.PR, within=Binary)
    m.zSP = Var(m.M, m.PS, within=Binary)

    # Membrane areas
    m.A   = Var(m.M, bounds=(0, AMAX))
    m.dAe = Var(m.M, bounds=(0, AMAX / NEL))

    # Network flows
    m.FIR = Var(m.I, m.NR, m.M, bounds=(0, FMAX))
    m.FRI = Var(m.I, m.M, bounds=(0, FMAX))
    m.FRO = Var(m.I, m.M, bounds=(0, FMAX))
    m.FSI = Var(m.I, m.M, bounds=(0, FMAX))
    m.FSO = Var(m.I, m.M, bounds=(0, FMAX))
    m.FRR = Var(m.I, m.M, m.M, bounds=(0, FMAX))
    m.FSR = Var(m.I, m.M, m.M, bounds=(0, FMAX))
    m.FRP = Var(m.I, m.M, m.PR, bounds=(0, FMAX))
    m.FSP = Var(m.I, m.M, m.PS, bounds=(0, FMAX))
    m.FPR = Var(m.I, m.PR, bounds=(0, FMAX))
    m.FPS = Var(m.I, m.PS, bounds=(0, FMAX))

    # OCFE profile variables
    m.FR  = Var(m.I, m.M, m.L, m.J, bounds=(0, FMAX))
    m.FS  = Var(m.I, m.M, m.L, m.J, bounds=(0, FMAX))
    m.FRt = Var(m.M, m.L, m.J, bounds=(EPS, FMAX))
    m.FSt = Var(m.M, m.L, m.J, bounds=(EPS, FMAX))
    m.xR  = Var(m.I, m.M, m.L, m.J, bounds=(0, 1))
    m.xS  = Var(m.I, m.M, m.L, m.J, bounds=(0, 1))
    m.Jf  = Var(m.I, m.M, m.L, m.J, bounds=(0, 50))

    # Objective
    m.obj = Objective(
        expr=sum(CA * m.A[md] for md in m.M) + PHI * sum(m.y[md] for md in m.M),
        sense=minimize)

    # Superstructure constraints
    @m.Constraint(m.M)
    def c_dA(m, md):
        return m.dAe[md] == m.A[md] / NEL

    @m.Constraint(m.M)
    def c_Au(m, md):
        return m.A[md] <= m.y[md] * AMAX

    @m.Constraint(m.M)
    def c_Al(m, md):
        return m.A[md] >= m.y[md] * AMIN

    @m.Constraint(m.I, m.NR)
    def c_fs(m, i, nr):
        return m.Ff[i] == sum(m.FIR[i, nr, md] for md in m.M)

    @m.Constraint(m.I, m.M)
    def c_rm(m, i, md):
        return m.FRI[i, md] == (
            sum(m.FIR[i, nr, md] for nr in m.NR)
            + sum(m.FRR[i, mp, md] for mp in m.M if mp != md)
            + sum(m.FSR[i, mp, md] for mp in m.M if mp != md))

    @m.Constraint(m.I, m.M)
    def c_si(m, i, md):
        return m.FSI[i, md] == 0

    @m.Constraint(m.I, m.M)
    def c_rs(m, i, md):
        return m.FRO[i, md] == (
            sum(m.FRR[i, md, mp] for mp in m.M if mp != md)
            + sum(m.FRP[i, md, pr] for pr in m.PR))

    @m.Constraint(m.I, m.M)
    def c_ps(m, i, md):
        return m.FSO[i, md] == (
            sum(m.FSP[i, md, ps] for ps in m.PS)
            + sum(m.FSR[i, md, mp] for mp in m.M if mp != md))

    @m.Constraint(m.I, m.PR)
    def c_rp(m, i, pr):
        return m.FPR[i, pr] == sum(m.FRP[i, md, pr] for md in m.M)

    @m.Constraint(m.I, m.PS)
    def c_pp(m, i, ps):
        return m.FPS[i, ps] == sum(m.FSP[i, md, ps] for md in m.M)

    # Big-M on network flows
    @m.Constraint(m.I, m.NR, m.M)
    def c_bI(m, i, nr, md):
        return m.FIR[i, nr, md] <= m.zIR[nr, md] * FMAX

    @m.Constraint(m.I, m.M, m.M)
    def c_bR(m, i, md, mp):
        if md == mp:
            return Constraint.Skip
        return m.FRR[i, md, mp] <= m.zRR[md, mp] * FMAX

    @m.Constraint(m.I, m.M, m.M)
    def c_bSR(m, i, md, mp):
        if md == mp:
            return Constraint.Skip
        return m.FSR[i, md, mp] <= m.zSR[md, mp] * FMAX

    @m.Constraint(m.I, m.M, m.PR)
    def c_bP(m, i, md, pr):
        return m.FRP[i, md, pr] <= m.zRP[md, pr] * FMAX

    @m.Constraint(m.I, m.M, m.PS)
    def c_bS(m, i, md, ps):
        return m.FSP[i, md, ps] <= m.zSP[md, ps] * FMAX

    # Logic constraints
    @m.Constraint(m.NR, m.M)
    def c_l1(m, nr, md):
        return m.zIR[nr, md] <= m.y[md]

    @m.Constraint(m.M, m.M)
    def c_l2(m, md, mp):
        if md == mp:
            return Constraint.Skip
        return m.zRR[md, mp] <= m.y[md]

    @m.Constraint(m.M, m.M)
    def c_l3(m, md, mp):
        if md == mp:
            return Constraint.Skip
        return m.zRR[md, mp] <= m.y[mp]

    @m.Constraint(m.M, m.M)
    def c_l4a(m, md, mp):
        if md == mp:
            return Constraint.Skip
        return m.zSR[md, mp] <= m.y[md]

    @m.Constraint(m.M, m.M)
    def c_l4b(m, md, mp):
        if md == mp:
            return Constraint.Skip
        return m.zSR[md, mp] <= m.y[mp]

    @m.Constraint(m.M, m.PR)
    def c_l5(m, md, pr):
        return m.zRP[md, pr] <= m.y[md]

    @m.Constraint(m.M, m.PS)
    def c_l6(m, md, ps):
        return m.zSP[md, ps] <= m.y[md]

    # No self-loops
    @m.Constraint(m.M)
    def c_nsR(m, md):
        return m.zRR[md, md] == 0

    @m.Constraint(m.M)
    def c_nsSR(m, md):
        return m.zSR[md, md] == 0

    @m.Constraint(m.I, m.M)
    def c_nfR(m, i, md):
        return m.FRR[i, md, md] == 0

    @m.Constraint(m.I, m.M)
    def c_nfSR(m, i, md):
        return m.FSR[i, md, md] == 0

    # Flows in inactive modules must be zero
    @m.Constraint(m.I, m.M)
    def c_bRI(m, i, md):
        return m.FRI[i, md] <= m.y[md] * FMAX

    @m.Constraint(m.I, m.M)
    def c_bRO(m, i, md):
        return m.FRO[i, md] <= m.y[md] * FMAX

    @m.Constraint(m.I, m.M)
    def c_bSO(m, i, md):
        return m.FSO[i, md] <= m.y[md] * FMAX

    m.c_min = Constraint(expr=sum(m.y[md] for md in m.M) >= 1)

    @m.Constraint(m.M)
    def c_ro(m, md):
        return (sum(m.zRP[md, pr] for pr in m.PR)
                + sum(m.zRR[md, mp] for mp in m.M if mp != md) >= m.y[md])

    @m.Constraint(m.M)
    def c_po(m, md):
        return (sum(m.zSP[md, ps] for ps in m.PS)
                + sum(m.zSR[md, mp] for mp in m.M if mp != md) >= m.y[md])

    # OCFE constraints
    # Zero out OCFE variables in inactive modules
    @m.Constraint(m.I, m.M, m.L, m.J)
    def c_bFR(m, i, md, l, j):
        return m.FR[i, md, l, j] <= m.y[md] * FMAX

    @m.Constraint(m.I, m.M, m.L, m.J)
    def c_bFS(m, i, md, l, j):
        return m.FS[i, md, l, j] <= m.y[md] * FMAX

    @m.Constraint(m.M, m.L, m.J)
    def c_bTR(m, md, l, j):
        return m.FRt[md, l, j] <= m.y[md] * FMAX + EPS

    @m.Constraint(m.M, m.L, m.J)
    def c_bTS(m, md, l, j):
        return m.FSt[md, l, j] <= m.y[md] * FMAX + EPS

    @m.Constraint(m.I, m.M, m.L, m.J)
    def c_bJf(m, i, md, l, j):
        return m.Jf[i, md, l, j] <= m.y[md] * 50

    # Boundary conditions
    @m.Constraint(m.I, m.M)
    def c_rI(m, i, md):
        return m.FR[i, md, 1, 0] == m.FRI[i, md]

    @m.Constraint(m.I, m.M)
    def c_sI(m, i, md):
        return m.FS[i, md, 1, 0] == m.FSI[i, md]

    @m.Constraint(m.I, m.M)
    def c_rO(m, i, md):
        return m.FRO[i, md] == m.FR[i, md, NEL, NCP]

    @m.Constraint(m.I, m.M)
    def c_sO(m, i, md):
        return m.FSO[i, md] == m.FS[i, md, NEL, NCP]

    # Element continuity
    @m.Constraint(m.I, m.M, m.L)
    def c_ecR(m, i, md, l):
        if l == 1:
            return Constraint.Skip
        return m.FR[i, md, l, 0] == m.FR[i, md, l - 1, NCP]

    @m.Constraint(m.I, m.M, m.L)
    def c_ecS(m, i, md, l):
        if l == 1:
            return Constraint.Skip
        return m.FS[i, md, l, 0] == m.FS[i, md, l - 1, NCP]

    # Total flows
    @m.Constraint(m.M, m.L, m.J)
    def c_tR(m, md, l, j):
        return m.FRt[md, l, j] == sum(m.FR[i, md, l, j] for i in m.I)

    @m.Constraint(m.M, m.L, m.J)
    def c_tS(m, md, l, j):
        return m.FSt[md, l, j] == sum(m.FS[i, md, l, j] for i in m.I)

    # Mole fractions (bilinear: xR * FRt = FR)
    @m.Constraint(m.I, m.M, m.L, m.J)
    def c_xR(m, i, md, l, j):
        return m.xR[i, md, l, j] * m.FRt[md, l, j] == m.FR[i, md, l, j]

    @m.Constraint(m.I, m.M, m.L, m.J)
    def c_xS(m, i, md, l, j):
        return m.xS[i, md, l, j] * m.FSt[md, l, j] == m.FS[i, md, l, j]

    # Flux with big-M deactivation
    # When y[md]=1 (active): reduces to  Jf == Pi*(xR*PR - xS*PP)
    # When y[md]=0 (inactive): becomes   -BIGM <= Jf - Pi*(...) <= BIGM
    # which is trivially satisfied for any values within bounds.
    @m.Constraint(m.I, m.M, m.L, m.J)
    def c_flux_lo(m, i, md, l, j):
        lhs = m.Jf[i, md, l, j] - m.Pi[i] * (m.xR[i, md, l, j] * PR
                                                - m.xS[i, md, l, j] * PP)
        return lhs >= -BIGM_FLUX * (1 - m.y[md])

    @m.Constraint(m.I, m.M, m.L, m.J)
    def c_flux_hi(m, i, md, l, j):
        lhs = m.Jf[i, md, l, j] - m.Pi[i] * (m.xR[i, md, l, j] * PR
                                                - m.xS[i, md, l, j] * PP)
        return lhs <= BIGM_FLUX * (1 - m.y[md])

    # OCFE mass balances with big-M deactivation
    # Active:   sum_k D[j,k]*FR[i,md,l,k] + Jf*dAe == 0
    # Inactive: |sum_k D[j,k]*FR[...] + Jf*dAe| <= BIGM_MB
    @m.Constraint(m.I, m.M, m.L, m.Ji)
    def c_mbR_lo(m, i, md, l, j):
        expr = (sum(m.Dm[j, k] * m.FR[i, md, l, k] for k in m.J)
                + m.Jf[i, md, l, j] * m.dAe[md])
        return expr >= -BIGM_MB * (1 - m.y[md])

    @m.Constraint(m.I, m.M, m.L, m.Ji)
    def c_mbR_hi(m, i, md, l, j):
        expr = (sum(m.Dm[j, k] * m.FR[i, md, l, k] for k in m.J)
                + m.Jf[i, md, l, j] * m.dAe[md])
        return expr <= BIGM_MB * (1 - m.y[md])

    @m.Constraint(m.I, m.M, m.L, m.Ji)
    def c_mbS_lo(m, i, md, l, j):
        expr = (sum(m.Dm[j, k] * m.FS[i, md, l, k] for k in m.J)
                - m.Jf[i, md, l, j] * m.dAe[md])
        return expr >= -BIGM_MB * (1 - m.y[md])

    @m.Constraint(m.I, m.M, m.L, m.Ji)
    def c_mbS_hi(m, i, md, l, j):
        expr = (sum(m.Dm[j, k] * m.FS[i, md, l, k] for k in m.J)
                - m.Jf[i, md, l, j] * m.dAe[md])
        return expr <= BIGM_MB * (1 - m.y[md])

    # Product specifications
    m.c_pur = Constraint(expr=m.FPR[1, 1] == 0.95 * (m.FPR[1, 1] + m.FPR[2, 1]))
    m.c_rec = Constraint(expr=m.FPR[1, 1] == 0.90 * FF[1])

    return m

# State vector y = [rN, rO, pN, pO].
def ren_perm_out_flows(fN, fO, A, ns=60):
    """
    Returns (rN, rO, pN, pO): retentate and permeate outlet flows [kmol/h]
    """
    if A < 1:
        return float(fN), float(fO), 0.0, 0.0
    # State: [retentate_N2, retentate_O2, permeate_N2, permeate_O2]
    y = np.array([float(fN), float(fO), 0.0, 0.0])
    dA = A/ns
    def rhs(state):
        """Right-hand side dy/dA for the membrane ODE."""
        rN, rO, pN, pO = state
        Ft_r = rN + rO
        Ft_p = pN + pO
        if Ft_r < 1e-12:
            return np.zeros(4)
        # Mole fractions
        xNr = rN / Ft_r
        xOr = rO / Ft_r
        if Ft_p > 1e-12:
            xNp = pN / Ft_p
            xOp = pO / Ft_p
        else:
            # Initial guess when permeate is empty: O2-rich since it
            # permeates ~10x faster
            xOp = 0.8
            xNp = 0.2
        # Transmembrane fluxes
        JN = PI[1] * max(xNr * PR - xNp * PP, 0.0)
        JO = PI[2] * max(xOr * PR - xOp * PP, 0.0)
        # dy/dA: retentate loses flux, permeate gains it
        return np.array([-JN, -JO, JN, JO])
    # Classical RK4 integration
    for _ in range(ns):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dA * k1)
        k3 = rhs(y + 0.5 * dA * k2)
        k4 = rhs(y + dA * k3)
        y = y + (dA / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Floor at zero — can't have negative flows
        y = np.maximum(y, 0.0)
    return float(y[0]), float(y[1]), float(y[2]), float(y[3])
    
# Network simulation
def simulate_network(active, feed_to, rr, sr, rp, sp, areas):
    rec = {}
    for iteration in range(200):
        flows = {}
        for md in active:
            inN = FF[1] / len(feed_to) if md in feed_to else 0.0
            inO = FF[2] / len(feed_to) if md in feed_to else 0.0
            for src in active:
                if src == md:
                    continue
                if (src, md) in rr:
                    inN += rec.get((1, src, md), 0.0)
                    inO += rec.get((2, src, md), 0.0)
                if (src, md) in sr:
                    inN += rec.get((1, src, md), 0.0)
                    inO += rec.get((2, src, md), 0.0)
            rN, rO, pN, pO = ren_perm_out_flows(max(inN, EPS), max(inO, EPS),
                                    areas.get(md, 2000))
            flows[md] = {'inN': inN, 'inO': inO,
                         'rN': rN,  'rO': rO,
                         'pN': pN,  'pO': pO}
        new_rec = {}
        max_diff = 0.0
        for src in active:
            f = flows[src]
            for dst in active:
                if dst == src:
                    continue
                if (src, dst) in rr:
                    n_out = (sum(1 for d in active if d != src and (src, d) in rr)
                             + (1 if (src, 1) in rp else 0))
                    frac = 1.0 / max(n_out, 1)
                    new_rec[(1, src, dst)] = f['rN'] * frac
                    new_rec[(2, src, dst)] = f['rO'] * frac
                if (src, dst) in sr:
                    n_out = (sum(1 for d in active if d != src and (src, d) in sr)
                             + (1 if (src, 1) in sp else 0))
                    frac = 1.0 / max(n_out, 1)
                    new_rec[(1, src, dst)] = f['pN'] * frac
                    new_rec[(2, src, dst)] = f['pO'] * frac
        for key in new_rec:
            max_diff = max(max_diff, abs(new_rec[key] - rec.get(key, 0.0)))
            rec[key] = rec.get(key, 0.0) * 0.5 + new_rec[key] * 0.5
        if max_diff < 0.1 and iteration > 5:
            break
    prodN = prodO = 0.0
    for md in active:
        if (md, 1) in rp:
            f = flows[md]
            n_out = (sum(1 for d in active if d != md and (md, d) in rr)
                     + (1 if (md, 1) in rp else 0))
            frac = 1.0 / max(n_out, 1)
            prodN += f['rN'] * frac
            prodO += f['rO'] * frac
    pt = prodN + prodO
    if pt < 1.0:
        return None
    return {
        'flows':    flows,
        'pur':      prodN / pt,
        'rec':      prodN / FF[1],
        'recycles': rec
    }

# Initialization
def init_from_topo(m, active, feed_to, rr, sr, rp, sp, areas):
    sim_result = simulate_network(active, feed_to, rr, sr, rp, sp, areas)
    flows = sim_result['flows'] if sim_result else {}
    for md in m.M:
        is_active = md in active
        m.y[md].value = 1 if is_active else 0
        m.y[md].fix(m.y[md].value)
        m.zIR[1, md].value = 1 if md in feed_to else 0
        m.zIR[1, md].fix(m.zIR[1, md].value)
        m.zRP[md, 1].value = 1 if (md, 1) in rp else 0
        m.zRP[md, 1].fix(m.zRP[md, 1].value)
        m.zSP[md, 1].value = 1 if (md, 1) in sp else 0
        m.zSP[md, 1].fix(m.zSP[md, 1].value)
        for mp in m.M:
            m.zRR[md, mp].value = 1 if (md, mp) in rr else 0
            m.zRR[md, mp].fix(m.zRR[md, mp].value)
            m.zSR[md, mp].value = 1 if (md, mp) in sr else 0
            m.zSR[md, mp].fix(m.zSR[md, mp].value)
    for md in m.M:
        if md not in active:
            m.A[md].value = 0.0
            m.dAe[md].value = 0.0
            for i in m.I:
                for v in [m.FIR[i, 1, md], m.FRI[i, md], m.FRO[i, md],
                          m.FSI[i, md], m.FSO[i, md],
                          m.FRP[i, md, 1], m.FSP[i, md, 1]]:
                    v.value = 0.0
                for mp in m.M:
                    m.FRR[i, md, mp].value = 0.0
                    m.FSR[i, md, mp].value = 0.0
            for l in m.L:
                for j in m.J:
                    for i in m.I:
                        m.FR[i, md, l, j].value = 0.0
                        m.FS[i, md, l, j].value = 0.0
                        m.Jf[i, md, l, j].value = 0.0
                        m.xR[i, md, l, j].value = 0.5
                        m.xS[i, md, l, j].value = 0.5
                    m.FRt[md, l, j].value = EPS
                    m.FSt[md, l, j].value = EPS
            continue
        f = flows.get(md, {
            'inN': FF[1], 'inO': FF[2],
            'rN':  FF[1] * 0.7, 'rO': FF[2] * 0.3,
            'pN':  FF[1] * 0.3, 'pO': FF[2] * 0.7})
        A_md = areas.get(md, 2000)
        m.A[md].value = A_md
        m.dAe[md].value = A_md / NEL
        for i in m.I:
            m.FIR[i, 1, md].value = FF[i] / len(feed_to) if md in feed_to else 0.0
        m.FRI[1, md].value = max(f['inN'], EPS)
        m.FRI[2, md].value = max(f['inO'], EPS)
        m.FRO[1, md].value = max(f['rN'], EPS)
        m.FRO[2, md].value = max(f['rO'], EPS)
        m.FSO[1, md].value = max(f['pN'], EPS)
        m.FSO[2, md].value = max(f['pO'], EPS)
        m.FSI[1, md].value = 0.0
        m.FSI[2, md].value = 0.0
        n_rr  = sum(1 for d in active if d != md and (md, d) in rr)
        has_rp = (md, 1) in rp
        fr = 1.0 / max(n_rr + (1 if has_rp else 0), 1)
        n_sr  = sum(1 for d in active if d != md and (md, d) in sr)
        has_sp = (md, 1) in sp
        fs = 1.0 / max(n_sr + (1 if has_sp else 0), 1)
        for i in m.I:
            ro = m.FRO[i, md].value
            so = m.FSO[i, md].value
            m.FRP[i, md, 1].value = ro * fr if has_rp else 0.0
            m.FSP[i, md, 1].value = so * fs if has_sp else 0.0
            for mp in m.M:
                m.FRR[i, md, mp].value = ro * fr if (md, mp) in rr else 0.0
                m.FSR[i, md, mp].value = so * fs if (md, mp) in sr else 0.0
        for l in m.L:
            for j in m.J:
                p = ((l - 1) + S[j]) / NEL
                for i in m.I:
                    fi = m.FRI[i, md].value
                    fo = m.FRO[i, md].value
                    po = m.FSO[i, md].value
                    m.FR[i, md, l, j].value = max(fi * (1 - p) + fo * p, EPS)
                    m.FS[i, md, l, j].value = max(po * p, EPS)
                frt = sum(m.FR[i, md, l, j].value for i in m.I)
                fst = sum(m.FS[i, md, l, j].value for i in m.I)
                m.FRt[md, l, j].value = max(frt, EPS)
                m.FSt[md, l, j].value = max(fst, EPS)
                for i in m.I:
                    m.xR[i, md, l, j].value = m.FR[i, md, l, j].value / m.FRt[md, l, j].value
                    m.xS[i, md, l, j].value = m.FS[i, md, l, j].value / m.FSt[md, l, j].value
                    m.Jf[i, md, l, j].value = max(
                        PI[i] * (m.xR[i, md, l, j].value * PR
                                 - m.xS[i, md, l, j].value * PP), 0.0)
    for i in m.I:
        m.FPR[i, 1].value = sum(m.FRP[i, md, 1].value or 0.0 for md in m.M)
        m.FPS[i, 1].value = sum(m.FSP[i, md, 1].value or 0.0 for md in m.M)

def unfix_binaries(m):
    for md in m.M:
        m.y[md].unfix()
        m.zIR[1, md].unfix()
        m.zRP[md, 1].unfix()
        m.zSP[md, 1].unfix()
        for mp in m.M:
            m.zRR[md, mp].unfix()
            m.zSR[md, mp].unfix()

def report_config(max_mod=5):
    configs = []
    pool = list(range(1, NMOD + 1))
    for n in range(1, max_mod + 1):
        for active in combinations(pool, n):
            active = list(active)
            pairs = [(a, b) for a in active for b in active if a != b]
            feed_opts = [list(fo) for r in range(1, n + 1)
                         for fo in combinations(active, r)]
            if n == 1:
                conn_list = [{'rr': set(), 'sr': set()}]
            else:
                rr_sets = [set()]
                for p in pairs:
                    rr_sets = [c | {p} for c in rr_sets] + rr_sets
                sr_sets = [set()]
                for p in pairs:
                    sr_sets = [c | {p} for c in sr_sets] + sr_sets
                conn_list = [{'rr': r, 'sr': s} for r in rr_sets for s in sr_sets]
            for feed_to in feed_opts:
                for conn in conn_list:
                    rr = conn['rr']
                    sr = conn['sr']
                    for rp_bits in range(1, 2**n):
                        rp = set()
                        for idx, md in enumerate(active):
                            if rp_bits & (1 << idx):
                                rp.add((md, 1))
                        sp = set()
                        for md in active:
                            sp.add((md, 1))
                        ok = True
                        for md in active:
                            has_ret_out = (any((md, d) in rr for d in active if d != md)
                                          or (md, 1) in rp)
                            has_perm_out = (any((md, d) in sr for d in active if d != md)
                                           or (md, 1) in sp)
                            if not has_ret_out or not has_perm_out:
                                ok = False
                                break
                        if not ok:
                            continue
                        areas = {
                            md: (4000 if any((s, md) in rr or (s, md) in sr
                                             for s in active if s != md)
                                 else 2000)
                            for md in active
                        }
                        configs.append({
                            'active':  active,
                            'feed_to': feed_to,
                            'rr':      rr,
                            'sr':      sr,
                            'rp':      rp,
                            'sp':      sp,
                            'areas':   areas
                        })
    return configs
    
# Physics verification
def verify_physics(m, active):
    for md in active:
        fri_N = value(m.FRI[1, md])
        fri_O = value(m.FRI[2, md])
        fro_N = value(m.FRO[1, md])
        fro_O = value(m.FRO[2, md])
        fri_t = fri_N + fri_O
        fro_t = fro_N + fro_O
        if fri_t < 1.0 or fro_t < 1.0:
            continue
        in_pur  = fri_N / fri_t
        out_pur = fro_N / fro_t
        if out_pur < in_pur - 0.01:
            return False
        if out_pur > 0.999 and in_pur < 0.95:
            return False
        prev_xN = 0.0
        for l in m.L:
            for j in m.J:
                frt = value(m.FRt[md, l, j])
                if frt > EPS:
                    xN = value(m.FR[1, md, l, j]) / frt
                    if xN < prev_xN - 0.02:
                        return False
                    prev_xN = xN
    return True

SOLVER_OPTIONS = {
    'max_iter':                   5000,
    'tol':                        1e-6,
    'mu_strategy':               'adaptive',
    'bound_relax_factor':         1e-8,
    'acceptable_tol':             1e-4,
    'acceptable_dual_inf_tol':    1e4,
    'acceptable_iter':            3,
    'acceptable_constr_viol_tol': 1e-4,
    'nlp_scaling_method':        'gradient-based',
    'obj_scaling_factor':         1e-3,
}

def solve(m):
    sol = SolverFactory('mindtpy')
    configs = report_config(max_mod=5)
    promising = []
    for cfg in configs:
        result = simulate_network(
            cfg['active'], cfg['feed_to'],
            cfg['rr'], cfg['sr'],
            cfg['rp'], cfg['sp'],
            cfg['areas'])
        if result and result['pur'] > 0.85 and result['rec'] > 0.80:
            promising.append((cfg, result))
    best_obj = float('inf')
    best_cfg = None
    n_feasible = 0
    t_start = time.time()
    import logging
    logging.getLogger('pyomo.core').setLevel(logging.ERROR)
    for idx, (cfg, sim_result) in enumerate(promising):
        if time.time() - t_start > 300:
            print(" Time limit reached.")
            break
        init_from_topo(m, cfg['active'], cfg['feed_to'],
                       cfg['rr'], cfg['sr'],
                       cfg['rp'], cfg['sp'],
                       cfg['areas'])
        try:
            res = sol.solve(m, tee=False, options=SOLVER_OPTIONS,
                              load_solutions=False)
            tc = res.solver.termination_condition
            solver_ok = tc in (TerminationCondition.optimal,
                               TerminationCondition.locallyOptimal,
                               TerminationCondition.feasible)
            if not solver_ok:
                unfix_binaries(m)
                continue
            m.solutions.load_from(res)
            FN = value(m.FPR[1, 1])
            FO = value(m.FPR[2, 1])
            Fp = FN + FO
            if Fp < 1.0:
                unfix_binaries(m)
                continue
            pur = FN / Fp
            rec = FN / FF[1]
            obj = value(m.obj)
            phys_ok = verify_physics(m, cfg['active'])
            feasible = pur >= 0.9495 and rec >= 0.8995 and phys_ok
            if feasible:
                n_feasible += 1
                if obj < best_obj:
                    best_obj = obj
                    best_cfg = cfg
        except Exception:
            pass
        unfix_binaries(m)
    logging.getLogger('pyomo.core').setLevel(logging.WARNING)
    if best_cfg is None:
        return False
    print(f"  Solving (obj = ${best_obj:,.0f})")
    init_from_topo(m, best_cfg['active'], best_cfg['feed_to'],
                   best_cfg['rr'], best_cfg['sr'],
                   best_cfg['rp'], best_cfg['sp'],
                   best_cfg['areas'])
    sol.solve(m, tee=False, options=SOLVER_OPTIONS)
    unfix_binaries(m)
    return True

# Results reporting
def report(m, solver_ok):
    print("\n" + "=" * 92)
    print("  Air Separation Superstructure — Results")
    print("=" * 92)
    print(f"  Feed: {FT:.0f} kmol/h  ({XF[1]*100:.0f}% N2, {XF[2]*100:.0f}% O2)"
          f"  |  P_ret = {PR:.0f} bar, P_perm = {PP:.0f} bar  |  alpha(O2/N2) = 10")

    active_modules = [md for md in sorted(m.M) if (value(m.y[md]) or 0) > 0.5]

    FN = value(m.FPR[1, 1])
    FO = value(m.FPR[2, 1])
    Fp = FN + FO
    pur = FN / Fp if Fp > 0 else 0.0
    rec = FN / FF[1]
    total_area = sum(value(m.A[md]) for md in active_modules)
    n_modules  = len(active_modules)
    print(f"\n  Product:  purity = {pur:.4f} (== 0.95)   recovery = {rec:.4f} (== 0.90)")
    print(f"            N2 = {FN:.2f}  O2 = {FO:.2f}  total = {Fp:.2f} kmol/h")
    print(f"  Cost:     {total_area:.0f} m2 x ${CA}/m2 = ${total_area*CA:,.0f}"
          f"  +  {n_modules} modules x ${PHI:.0f} = ${n_modules*PHI:,.0f}"
          f"  =>  total ${value(m.obj):,.0f}/h")
    print(f"\n  Topology:")
    for md in active_modules:
        if (value(m.zIR[1, md]) or 0) > 0.5:
            ft = sum(value(m.FIR[i, 1, md]) for i in m.I)
            if ft > 0.01:
                print(f"    Feed -> M{md}:  {ft:.2f} kmol/h")
        for mp in m.M:
            if mp != md:
                if (value(m.zRR[md, mp]) or 0) > 0.5:
                    ft = sum(value(m.FRR[i, md, mp]) for i in m.I)
                    if ft > 0.01:
                        print(f"    M{md} retentate -> M{mp} inlet:  "
                              f"{ft:.2f} kmol/h")
                if (value(m.zSR[md, mp]) or 0) > 0.5:
                    ft = sum(value(m.FSR[i, md, mp]) for i in m.I)
                    if ft > 0.01:
                        print(f"    M{md} permeate -> M{mp} inlet:  "
                              f"{ft:.2f} kmol/h")
        if (value(m.zRP[md, 1]) or 0) > 0.5:
            ft = sum(value(m.FRP[i, md, 1]) for i in m.I)
            if ft > 0.01:
                print(f"    M{md} retentate -> product:  {ft:.2f} kmol/h")
        if (value(m.zSP[md, 1]) or 0) > 0.5:
            ft = sum(value(m.FSP[i, md, 1]) for i in m.I)
            if ft > 0.01:
                print(f"    M{md} permeate -> waste:  {ft:.2f} kmol/h")
    for md in active_modules:
        vals = []
        for l in m.L:
            for j in m.J:
                frt = value(m.FRt[md, l, j])
                xN2 = value(m.FR[1, md, l, j]) / frt if frt > EPS else 0.0
                vals.append(f"{xN2:.4f}")
        print(f"    M{md} (A = {value(m.A[md]):.0f} m2):  [{', '.join(vals)}]")
    pN = value(m.FPR[1, 1])
    pO = value(m.FPR[2, 1])
    sN = value(m.FPS[1, 1])
    sO = value(m.FPS[2, 1])
    mass_bal_err = abs(pN + sN - FF[1]) + abs(pO + sO - FF[2])
    print(f"\n  Mass balance:  feed = {FT:.0f}  product = {pN+pO:.2f}"
          f"  waste = {sN+sO:.2f}  error = {mass_bal_err:.4f} kmol/h")
    status = ("Solver confirmed optimality" if solver_ok
              else "Check solution")
    print(f"  Status: {status}")
    print("=" * 92)

# Entry point
def main():
    print("=" * 72)
    print("Air Separation Superstructure MINLP")
    print(f"  {NMOD} module slots  |  OCFE: {NEL} elements x {NCP} points")
    print(f"  Feed: {FT:.0f} kmol/h  |  Target: == 95% N2 purity, == 90% recovery")
    print(f"  Objective: minimize CA*A + PHI*y  (CA=${CA}/m2, PHI=${PHI}/module)")
    print("=" * 72)
    m = build_model()
    n_vars = sum(1 for _ in m.component_data_objects(Var, active=True))
    n_cons = sum(1 for _ in m.component_data_objects(Constraint, active=True))
    n_bin  = sum(1 for v in m.component_data_objects(Var, active=True)
                 if v.is_binary())
    print(f"\n  Model size: {n_vars} variables, "
          f"{n_cons} constraints")
    solver_ok = solve(m)
    report(m, solver_ok)
    return m

if __name__ == "__main__":
    m = main()
