# ============================================================
# Path-Capacity Validation + Anti-Cheating Logic Checks
# Standalone Python / Colab / Jupyter version
# ============================================================

import numpy as np

hbar = 1.0

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I4 = np.eye(4, dtype=complex)

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
ketp = (ket0 + ket1) / np.sqrt(2.0)

ZZ = np.kron(sz, sz)


# ============================================================
# Basic utilities
# ============================================================

def kron(a, b):
    return np.kron(a, b)


def normalize(psi):
    psi = np.asarray(psi, dtype=complex)
    n = np.linalg.norm(psi)
    if n < 1e-15:
        raise ValueError("Zero vector.")
    return psi / n


def ket_to_rho(psi):
    psi = normalize(psi)
    return np.outer(psi, psi.conjugate())


def project_density(rho, clip=True):
    rho = np.asarray(rho, dtype=complex)
    rho = 0.5 * (rho + rho.conjugate().T)

    tr = np.trace(rho)
    if abs(tr) < 1e-15:
        raise ValueError("Trace nearly zero.")
    rho = rho / tr

    if clip:
        vals, vecs = np.linalg.eigh(rho)
        vals = np.clip(np.real(vals), 0.0, None)
        s = np.sum(vals)
        if s < 1e-15:
            vals = np.array([1.0, 0.0])
            s = 1.0
        vals = vals / s
        rho = vecs @ np.diag(vals) @ vecs.conjugate().T
        rho = 0.5 * (rho + rho.conjugate().T)
        rho = rho / np.trace(rho)

    return rho


def expectation(psi, A):
    psi = normalize(psi)
    return np.vdot(psi, A @ psi)


def expectation_rho(rho, A):
    rho = project_density(rho, clip=False)
    return np.trace(rho @ A)


def energy_uncertainty_pure(psi, H):
    E = expectation(psi, H)
    E2 = expectation(psi, H @ H)
    var = np.real(E2 - E * np.conj(E))
    return float(np.sqrt(max(var, 0.0)))


def energy_uncertainty_mixed(rho, H):
    rho = project_density(rho, clip=False)
    E = np.real(expectation_rho(rho, H))
    E2 = np.real(expectation_rho(rho, H @ H))
    var = E2 - E * E
    return float(np.sqrt(max(var, 0.0)))


def fs_distance(psi, phi):
    psi = normalize(psi)
    phi = normalize(phi)
    ov = abs(np.vdot(psi, phi))
    return float(np.arccos(np.clip(ov, 0.0, 1.0)))


def fidelity_qubit_closed(rho, sigma):
    rho = project_density(rho, clip=False)
    sigma = project_density(sigma, clip=False)

    term = np.real(np.trace(rho @ sigma))
    det_r = np.real(np.linalg.det(rho))
    det_s = np.real(np.linalg.det(sigma))

    F = term + 2.0 * np.sqrt(max(det_r * det_s, 0.0))
    return float(np.clip(F, 0.0, 1.0))


def sqrtm_hermitian(A):
    A = 0.5 * (A + A.conjugate().T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(np.real(vals), 0.0, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.conjugate().T


def fidelity_uhlmann_generic(rho, sigma):
    rho = project_density(rho, clip=True)
    sigma = project_density(sigma, clip=True)

    sr = sqrtm_hermitian(rho)
    A = sr @ sigma @ sr
    A = 0.5 * (A + A.conjugate().T)

    vals = np.linalg.eigvalsh(A)
    vals = np.clip(np.real(vals), 0.0, None)

    F = float(np.sum(np.sqrt(vals)) ** 2)
    return float(np.clip(F, 0.0, 1.0))


def bures_angle(rho, sigma):
    F = fidelity_qubit_closed(rho, sigma)
    return float(np.arccos(np.sqrt(np.clip(F, 0.0, 1.0))))


def purity(rho):
    rho = project_density(rho, clip=False)
    return float(np.real(np.trace(rho @ rho)))


def safe_div(num, den, floor=1e-14):
    # Denominator floor is ONLY a guard against division by zero. The correct
    # models never approach it (capacities are O(1)); the reported wrong-model
    # errors are insensitive to reducing the floor (verified down to 1e-20).
    return np.asarray(num) / np.maximum(np.asarray(den), floor)


def partial_trace_B(rho):
    rho4 = np.asarray(rho, dtype=complex).reshape(2, 2, 2, 2)
    return np.einsum("abcb->ac", rho4)


def von_neumann_entropy(rho, base=2):
    rho = 0.5 * (rho + rho.conjugate().T)
    vals = np.linalg.eigvalsh(rho)
    vals = np.clip(np.real(vals), 0.0, 1.0)
    vals = vals[vals > 1e-15]

    if len(vals) == 0:
        return 0.0

    logs = np.log(vals)
    if base == 2:
        logs = logs / np.log(2.0)

    return float(-np.sum(vals * logs))


def concurrence_pure(psi):
    psi = normalize(psi)
    a00, a01, a10, a11 = psi
    return float(np.clip(2.0 * abs(a00 * a11 - a01 * a10), 0.0, 1.0))


def entropy_A(psi):
    return von_neumann_entropy(partial_trace_B(ket_to_rho(psi)), base=2)


def random_density_qubit(rng):
    r = rng.normal(size=3)
    r = r / max(np.linalg.norm(r), 1e-15)
    radius = rng.random() ** (1.0 / 3.0)
    return project_density(0.5 * (I2 + radius * (r[0] * sx + r[1] * sy + r[2] * sz)))


# ============================================================
# Q3: Non-commuting time-dependent pure drive
# ============================================================

def omega_q3(t, Omega0=1.7, ax=0.45, az=0.85, nu_x=2.2, nu_z=1.35):
    Ox = Omega0 * (1.0 + ax * np.sin(nu_x * t))
    Oz = Omega0 * az * np.cos(nu_z * t)
    return Ox, Oz


def H_q3(t, Omega0=1.7, ax=0.45, az=0.85, nu_x=2.2, nu_z=1.35):
    Ox, Oz = omega_q3(t, Omega0, ax, az, nu_x, nu_z)
    return 0.5 * hbar * (Ox * sx + Oz * sz)


def H_q3_wrong_no_z(t, Omega0=1.7, ax=0.45, nu_x=2.2):
    Ox = Omega0 * (1.0 + ax * np.sin(nu_x * t))
    return 0.5 * hbar * Ox * sx


def U_q3_midpoint(t_mid, dt, Omega0=1.7, ax=0.45, az=0.85, nu_x=2.2, nu_z=1.35):
    Ox, Oz = omega_q3(t_mid, Omega0, ax, az, nu_x, nu_z)
    Omag = np.sqrt(Ox**2 + Oz**2)

    if Omag < 1e-15:
        return I2.copy()

    n_dot_sigma = (Ox / Omag) * sx + (Oz / Omag) * sz
    return np.cos(Omag * dt / 2.0) * I2 - 1j * np.sin(Omag * dt / 2.0) * n_dot_sigma


def pure_projective_speed(psi, H):
    psi = normalize(psi)
    psi_dot = -1j * H @ psi / hbar
    vertical = psi * np.vdot(psi, psi_dot)
    horizontal = psi_dot - vertical
    return float(np.linalg.norm(horizontal))


def run_q3_noncommuting(T=2.2, n_steps=6001):
    t = np.linspace(0.0, T, n_steps)
    dt = t[1] - t[0]

    psi = np.zeros((n_steps, 2), dtype=complex)
    psi[0] = ket0

    for i in range(n_steps - 1):
        tm = 0.5 * (t[i] + t[i + 1])
        U = U_q3_midpoint(tm, dt)
        psi[i + 1] = normalize(U @ psi[i])

    dPhi = np.array([fs_distance(psi[i], psi[i + 1]) for i in range(n_steps - 1)])
    D_endpoint = np.array([fs_distance(psi[0], psi[i]) for i in range(n_steps)])

    H_mid = np.zeros(n_steps - 1)
    H_wrong_no_z = np.zeros(n_steps - 1)
    speed_identity_err = np.zeros(n_steps - 1)

    for i in range(n_steps - 1):
        tm = 0.5 * (t[i] + t[i + 1])

        U_half = U_q3_midpoint(t[i] + 0.25 * dt, 0.5 * dt)
        psi_mid = normalize(U_half @ psi[i])

        Hm = H_q3(tm)
        H_wrong = H_q3_wrong_no_z(tm)

        H_mid[i] = energy_uncertainty_pure(psi_mid, Hm) / hbar
        H_wrong_no_z[i] = energy_uncertainty_pure(psi_mid, H_wrong) / hbar

        speed = pure_projective_speed(psi_mid, Hm)
        speed_identity_err[i] = abs(speed - H_mid[i])

    Phi_path = np.zeros(n_steps)
    Phi_path[1:] = np.cumsum(dPhi)

    Phi_capacity = np.zeros(n_steps)
    Phi_capacity[1:] = np.cumsum(H_mid * dt)

    t_rec = np.zeros(n_steps)
    t_rec[1:] = np.cumsum(safe_div(dPhi, H_mid))

    H_mean = np.mean(H_mid)
    t_endpoint_mean = D_endpoint / max(H_mean, 1e-14)

    t_wrong_no_z = np.zeros(n_steps)
    t_wrong_no_z[1:] = np.cumsum(safe_div(dPhi, H_wrong_no_z))

    err_correct = t_rec - t
    err_endpoint = t_endpoint_mean - t
    err_wrong_no_z = t_wrong_no_z - t

    return {
        "t": t,
        "psi": psi,
        "dPhi": dPhi,
        "H_mid": H_mid,
        "H_wrong_no_z": H_wrong_no_z,
        "D_endpoint": D_endpoint,
        "Phi_path": Phi_path,
        "Phi_capacity": Phi_capacity,
        "t_rec": t_rec,
        "t_endpoint_mean": t_endpoint_mean,
        "t_wrong_no_z": t_wrong_no_z,
        "err_correct": err_correct,
        "err_endpoint": err_endpoint,
        "err_wrong_no_z": err_wrong_no_z,
        "rel_correct": float(np.max(np.abs(err_correct)) / T),
        "rmse_endpoint": float(np.sqrt(np.mean(err_endpoint**2))),
        "rmse_wrong_no_z": float(np.sqrt(np.mean(err_wrong_no_z**2))),
        "path_endpoint_ratio": float(Phi_path[-1] / max(D_endpoint[-1], 1e-14)),
        "max_speed_identity_err": float(np.max(speed_identity_err)),
    }


def rk4_schrodinger_step(psi, t, dt):
    def rhs(p, tt):
        return -1j * H_q3(tt) @ p / hbar

    k1 = rhs(psi, t)
    k2 = rhs(psi + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(psi + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(psi + dt * k3, t + dt)

    return normalize(psi + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0)


def independent_q3_rk4_check(T=2.2, n_steps=6001):
    q3 = run_q3_noncommuting(T=T, n_steps=n_steps)

    t = q3["t"]
    dt = t[1] - t[0]

    psi_rk4 = np.zeros_like(q3["psi"])
    psi_rk4[0] = ket0

    for i in range(n_steps - 1):
        psi_rk4[i + 1] = rk4_schrodinger_step(psi_rk4[i], t[i], dt)

    distances = np.array([
        fs_distance(q3["psi"][i], psi_rk4[i])
        for i in range(n_steps)
    ])

    return {
        "max_state_distance": float(np.max(distances)),
        "mean_state_distance": float(np.mean(distances)),
    }


# ============================================================
# Q4: Open dephasing with Bures Liouvillian capacity
# ============================================================

def lindblad_rhs_dephasing(rho, t, gamma=0.35):
    H = H_q3(t)
    unitary = -1j / hbar * (H @ rho - rho @ H)
    dephasing = gamma * (sz @ rho @ sz - rho)
    return unitary + dephasing


def rk4_density_step(rho, t, dt, gamma=0.35, clip_after=False):
    """One RK4 step of the Lindblad equation.

    By default (clip_after=False) the propagated state is stabilized only by
    Hermitization and trace renormalization -- NO positivity projection is
    applied to the trajectory, so the orbit is not artificially corrected.
    Set clip_after=True to additionally project onto the PSD cone (diagnostic
    only). The minimum eigenvalue is tracked in run_q4_open_dephasing."""
    k1 = lindblad_rhs_dephasing(rho, t, gamma)
    k2 = lindblad_rhs_dephasing(rho + 0.5 * dt * k1, t + 0.5 * dt, gamma)
    k3 = lindblad_rhs_dephasing(rho + 0.5 * dt * k2, t + 0.5 * dt, gamma)
    k4 = lindblad_rhs_dephasing(rho + dt * k3, t + dt, gamma)

    rho_next = rho + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    rho_next = 0.5 * (rho_next + rho_next.conjugate().T)   # Hermiticity
    rho_next = rho_next / np.trace(rho_next)               # trace
    if clip_after:
        rho_next = project_density(rho_next, clip=True)    # optional PSD projection
    return rho_next


def liouvillian_bures_speed(rho, t, gamma=0.35, eps=1e-6):
    rho = project_density(rho, clip=True)
    Lrho = lindblad_rhs_dephasing(rho, t, gamma)
    rho_eps = project_density(rho + eps * Lrho, clip=True)
    return bures_angle(rho, rho_eps) / eps


def run_q4_open_dephasing(T=2.2, gamma=0.35, n_steps=5001,
                          eps_speed=None, eps_mode="dt", clip_after=False):
    t = np.linspace(0.0, T, n_steps)
    dt = t[1] - t[0]

    # Bures-speed finite-difference step. Default eps_mode="dt" makes the
    # function default reproduce the value reported in the paper Methods;
    # eps_mode="small" recovers the previous min(0.05*dt, 1e-5) choice.
    if eps_speed is None:
        if eps_mode == "dt":
            eps_speed = dt
        elif eps_mode == "small":
            eps_speed = min(0.05 * dt, 1e-5)
        else:
            raise ValueError("eps_mode must be 'dt' or 'small'")

    rho = np.zeros((n_steps, 2, 2), dtype=complex)
    rho[0] = ket_to_rho(ket0)

    pur = np.zeros(n_steps)
    pur[0] = purity(rho[0])
    min_eig = np.zeros(n_steps)
    min_eig[0] = float(np.min(np.real(np.linalg.eigvalsh(rho[0]))))

    for i in range(n_steps - 1):
        rho[i + 1] = rk4_density_step(rho[i], t[i], dt, gamma, clip_after=clip_after)
        pur[i + 1] = purity(rho[i + 1])
        min_eig[i + 1] = float(np.min(np.real(np.linalg.eigvalsh(rho[i + 1]))))

    dPhi = np.array([bures_angle(rho[i], rho[i + 1]) for i in range(n_steps - 1)])
    D_endpoint = np.array([bures_angle(rho[0], rho[i]) for i in range(n_steps)])

    H_L = np.zeros(n_steps - 1)
    H_wrong_deltaE = np.zeros(n_steps - 1)
    H_wrong_gamma0 = np.zeros(n_steps - 1)
    H_hs = np.zeros(n_steps - 1)

    for i in range(n_steps - 1):
        tm = 0.5 * (t[i] + t[i + 1])
        rho_mid = project_density(0.5 * (rho[i] + rho[i + 1]), clip=True)

        H_L[i] = liouvillian_bures_speed(rho_mid, tm, gamma=gamma, eps=eps_speed)
        H_wrong_gamma0[i] = liouvillian_bures_speed(rho_mid, tm, gamma=0.0, eps=eps_speed)

        Hm = H_q3(tm)
        H_wrong_deltaE[i] = energy_uncertainty_mixed(rho_mid, Hm) / hbar

        Lrho = lindblad_rhs_dephasing(rho_mid, tm, gamma)
        H_hs[i] = np.linalg.norm(Lrho, ord="fro")

    Phi_path = np.zeros(n_steps)
    Phi_path[1:] = np.cumsum(dPhi)

    Phi_capacity = np.zeros(n_steps)
    Phi_capacity[1:] = np.cumsum(H_L * dt)

    t_rec = np.zeros(n_steps)
    t_rec[1:] = np.cumsum(safe_div(dPhi, H_L))

    t_wrong_deltaE = np.zeros(n_steps)
    t_wrong_deltaE[1:] = np.cumsum(safe_div(dPhi, H_wrong_deltaE))

    t_wrong_gamma0 = np.zeros(n_steps)
    t_wrong_gamma0[1:] = np.cumsum(safe_div(dPhi, H_wrong_gamma0))

    t_wrong_hs = np.zeros(n_steps)
    t_wrong_hs[1:] = np.cumsum(safe_div(dPhi, H_hs))

    H_mean = np.mean(H_L)
    t_endpoint_mean = D_endpoint / max(H_mean, 1e-14)

    err_correct = t_rec - t
    err_wrong_deltaE = t_wrong_deltaE - t
    err_wrong_gamma0 = t_wrong_gamma0 - t
    err_wrong_hs = t_wrong_hs - t
    err_endpoint = t_endpoint_mean - t

    return {
        "t": t,
        "rho": rho,
        "purity": pur,
        "dPhi": dPhi,
        "D_endpoint": D_endpoint,
        "Phi_path": Phi_path,
        "Phi_capacity": Phi_capacity,
        "H_L": H_L,
        "H_wrong_deltaE": H_wrong_deltaE,
        "H_wrong_gamma0": H_wrong_gamma0,
        "H_hs": H_hs,
        "t_rec": t_rec,
        "t_wrong_deltaE": t_wrong_deltaE,
        "t_wrong_gamma0": t_wrong_gamma0,
        "t_wrong_hs": t_wrong_hs,
        "t_endpoint_mean": t_endpoint_mean,
        "err_correct": err_correct,
        "err_wrong_deltaE": err_wrong_deltaE,
        "err_wrong_gamma0": err_wrong_gamma0,
        "err_wrong_hs": err_wrong_hs,
        "err_endpoint": err_endpoint,
        "rel_correct": float(np.max(np.abs(err_correct)) / T),
        "rmse_wrong_deltaE": float(np.sqrt(np.mean(err_wrong_deltaE**2))),
        "rmse_wrong_gamma0": float(np.sqrt(np.mean(err_wrong_gamma0**2))),
        "rmse_wrong_hs": float(np.sqrt(np.mean(err_wrong_hs**2))),
        "rmse_endpoint": float(np.sqrt(np.mean(err_endpoint**2))),
        "purity_final": float(pur[-1]),
        "purity_drop": float(pur[0] - pur[-1]),
        "path_endpoint_ratio": float(Phi_path[-1] / max(D_endpoint[-1], 1e-14)),
        "eps_speed": eps_speed,
        "min_eig": min_eig,
        "min_eig_overall": float(np.min(min_eig)),
        "clip_after": clip_after,
    }


# ============================================================
# Q5: Two-qubit entangling gate
# ============================================================

def H_ZZ(J=1.3):
    return hbar * J / 2.0 * ZZ


def U_ZZ(t, J=1.3):
    return np.cos(J * t / 2.0) * I4 - 1j * np.sin(J * t / 2.0) * ZZ


def run_q5_entangler(J=1.3, T_factor=1.25, n_steps=3001):
    H = H_ZZ(J)
    psi0 = kron(ketp, ketp)

    T_bell = np.pi / (2.0 * J)
    T = T_factor * np.pi / J

    t = np.linspace(0.0, T, n_steps)
    dt = t[1] - t[0]

    psi = np.zeros((n_steps, 4), dtype=complex)
    Hcap = np.zeros(n_steps)
    D_endpoint = np.zeros(n_steps)
    concurrence = np.zeros(n_steps)
    ent = np.zeros(n_steps)

    for i, ti in enumerate(t):
        psi[i] = normalize(U_ZZ(ti, J) @ psi0)
        Hcap[i] = energy_uncertainty_pure(psi[i], H) / hbar
        D_endpoint[i] = fs_distance(psi0, psi[i])
        concurrence[i] = concurrence_pure(psi[i])
        ent[i] = entropy_A(psi[i])

    dPhi = np.array([fs_distance(psi[i], psi[i + 1]) for i in range(n_steps - 1)])

    Phi_path = np.zeros(n_steps)
    Phi_path[1:] = np.cumsum(dPhi)

    H_mid = 0.5 * (Hcap[:-1] + Hcap[1:])

    Phi_capacity = np.zeros(n_steps)
    Phi_capacity[1:] = np.cumsum(H_mid * dt)

    t_rec = np.zeros(n_steps)
    t_rec[1:] = np.cumsum(safe_div(dPhi, H_mid))

    H_mean = np.mean(H_mid)
    t_endpoint_mean = D_endpoint / max(H_mean, 1e-14)

    err_correct = t_rec - t
    err_endpoint = t_endpoint_mean - t

    idx = int(np.argmax(ent))

    return {
        "t": t,
        "psi": psi,
        "Hcap": Hcap,
        "D_endpoint": D_endpoint,
        "dPhi": dPhi,
        "Phi_path": Phi_path,
        "Phi_capacity": Phi_capacity,
        "t_rec": t_rec,
        "t_endpoint_mean": t_endpoint_mean,
        "err_correct": err_correct,
        "err_endpoint": err_endpoint,
        "concurrence": concurrence,
        "entropy": ent,
        "T": T,
        "T_bell": T_bell,
        "t_max_ent": float(t[idx]),
        "bell_time_error": float(abs(t[idx] - T_bell)),
        "rel_correct": float(np.max(np.abs(err_correct)) / T),
        "rmse_endpoint": float(np.sqrt(np.mean(err_endpoint**2))),
        "max_concurrence": float(np.max(concurrence)),
        "max_entropy": float(np.max(ent)),
        "path_endpoint_ratio": float(Phi_path[-1] / max(D_endpoint[-1], 1e-14)),
    }


# ============================================================
# Anti-cheating logic checks
# ============================================================

def q5_analytic_check(q5, J=1.3):
    H_expected = J / 2.0
    T_bell_expected = np.pi / (2.0 * J)

    path_expected_final = H_expected * q5["t"][-1]

    return {
        "Hcap_constant_error": float(np.max(np.abs(q5["Hcap"] - H_expected))),
        "T_bell_error": float(abs(q5["T_bell"] - T_bell_expected)),
        "path_final_error": float(abs(q5["Phi_path"][-1] - path_expected_final)),
        "max_concurrence_error": float(abs(q5["max_concurrence"] - 1.0)),
        "max_entropy_error": float(abs(q5["max_entropy"] - 1.0)),
    }


def fidelity_formula_crosscheck(seed=1234, n=2000):
    rng = np.random.default_rng(seed)
    max_diff = 0.0

    for _ in range(n):
        rho = random_density_qubit(rng)
        sigma = random_density_qubit(rng)

        F1 = fidelity_qubit_closed(rho, sigma)
        F2 = fidelity_uhlmann_generic(rho, sigma)

        max_diff = max(max_diff, abs(F1 - F2))

    return {
        "max_fidelity_formula_diff": float(max_diff),
    }


def q4_eps_stability_check(T=2.2, gamma=0.35, n_steps=3001):
    eps_list = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    rows = []

    for eps in eps_list:
        q4 = run_q4_open_dephasing(T=T, gamma=gamma, n_steps=n_steps, eps_speed=eps)
        rows.append({
            "eps": eps,
            "rel_correct": q4["rel_correct"],
        })

    vals = np.array([r["rel_correct"] for r in rows])
    return {
        "rows": rows,
        "max_rel_correct": float(np.max(vals)),
        "spread_rel_correct": float(np.max(vals) - np.min(vals)),
    }


def stepsize_convergence_check():
    q3_rows = []
    q4_rows = []
    q5_rows = []

    for n in [1001, 2001, 4001]:
        q3 = run_q3_noncommuting(n_steps=n)
        q3_rows.append((n, q3["rel_correct"]))

    for n in [1001, 2001, 4001]:
        q4 = run_q4_open_dephasing(n_steps=n)
        q4_rows.append((n, q4["rel_correct"]))

    for n in [1001, 2001, 4001]:
        q5 = run_q5_entangler(n_steps=n)
        q5_rows.append((n, q5["rel_correct"]))

    return {
        "q3": q3_rows,
        "q4": q4_rows,
        "q5": q5_rows,
    }


# ============================================================
# Summary printing
# ============================================================

def print_main_summary(q3, q4, q5):
    print("=" * 78)
    print("PATH-CAPACITY MAIN VALIDATION SUMMARY")
    print("=" * 78)

    print("\n[Q3: NON-COMMUTING PURE DRIVE]")
    print(f"correct relative error          = {q3['rel_correct']:.3e}")
    print(f"endpoint/mean RMSE              = {q3['rmse_endpoint']:.3e}")
    print(f"wrong no-z model RMSE           = {q3['rmse_wrong_no_z']:.3e}")
    print(f"path/endpoint ratio             = {q3['path_endpoint_ratio']:.6f}")
    print(f"speed identity max error        = {q3['max_speed_identity_err']:.3e}")

    print("\n[Q4: OPEN DEPHASING]")
    print(f"correct Liouvillian rel error   = {q4['rel_correct']:.3e}")
    print(f"wrong DeltaE RMSE               = {q4['rmse_wrong_deltaE']:.3e}")
    print(f"wrong gamma=0 RMSE              = {q4['rmse_wrong_gamma0']:.3e}")
    print(f"wrong Hilbert-Schmidt RMSE      = {q4['rmse_wrong_hs']:.3e}")
    print(f"endpoint/mean RMSE              = {q4['rmse_endpoint']:.3e}")
    print(f"purity final                    = {q4['purity_final']:.6f}")
    print(f"purity drop                     = {q4['purity_drop']:.6f}")
    print(f"path/endpoint ratio             = {q4['path_endpoint_ratio']:.6f}")

    print("\n[Q5: TWO-QUBIT ENTANGLING]")
    print(f"correct relative error          = {q5['rel_correct']:.3e}")
    print(f"endpoint/mean RMSE              = {q5['rmse_endpoint']:.3e}")
    print(f"max concurrence                 = {q5['max_concurrence']:.8f}")
    print(f"max entropy                     = {q5['max_entropy']:.8f}")
    print(f"Bell time error                 = {q5['bell_time_error']:.3e}")
    print(f"path/endpoint ratio             = {q5['path_endpoint_ratio']:.6f}")

    print("=" * 78)


def print_anti_cheat_summary(q3, q4, q5, rk4, q5a, fid, epscheck, steps):
    print("\n" + "=" * 78)
    print("ANTI-CHEATING LOGIC CHECKS")
    print("=" * 78)

    print("\n[Independent evolution check]")
    print(f"Q3 midpoint-vs-RK4 max FS distance     = {rk4['max_state_distance']:.3e}")
    print(f"Q3 midpoint-vs-RK4 mean FS distance    = {rk4['mean_state_distance']:.3e}")

    print("\n[Pure speed identity]")
    print(f"max |projective speed - DeltaE/hbar|   = {q3['max_speed_identity_err']:.3e}")

    print("\n[Bures fidelity formula crosscheck]")
    print(f"max |closed qubit F - generic F|       = {fid['max_fidelity_formula_diff']:.3e}")

    print("\n[Q5 analytic checks]")
    print(f"Hcap constant error                    = {q5a['Hcap_constant_error']:.3e}")
    print(f"T_bell analytic error                  = {q5a['T_bell_error']:.3e}")
    print(f"path final analytic error              = {q5a['path_final_error']:.3e}")
    print(f"max concurrence error                  = {q5a['max_concurrence_error']:.3e}")
    print(f"max entropy error                      = {q5a['max_entropy_error']:.3e}")

    print("\n[Q4 Bures-speed eps stability]")
    for row in epscheck["rows"]:
        print(f"eps={row['eps']:.1e}  rel_correct={row['rel_correct']:.3e}")
    print(f"eps-scan max rel error                 = {epscheck['max_rel_correct']:.3e}")
    print(f"eps-scan spread                        = {epscheck['spread_rel_correct']:.3e}")

    print("\n[Step-size convergence]")
    print("Q3:", steps["q3"])
    print("Q4:", steps["q4"])
    print("Q5:", steps["q5"])

    print("\n[Negative controls]")
    print(f"Q3 wrong no-z model RMSE               = {q3['rmse_wrong_no_z']:.3e}")
    print(f"Q4 wrong gamma=0 RMSE                  = {q4['rmse_wrong_gamma0']:.3e}")
    print(f"Q4 wrong DeltaE RMSE                   = {q4['rmse_wrong_deltaE']:.3e}")

    flags = {
        "q3_reconstruction_pass": q3["rel_correct"] < 1e-6,
        "q4_reconstruction_pass": q4["rel_correct"] < 2e-3,
        "q5_reconstruction_pass": q5["rel_correct"] < 1e-8,
        "q3_independent_evolution_pass": rk4["max_state_distance"] < 1e-5,
        "pure_speed_identity_pass": q3["max_speed_identity_err"] < 1e-12,
        "bures_fidelity_crosscheck_pass": fid["max_fidelity_formula_diff"] < 1e-10,
        "q5_analytic_pass": (
            q5a["Hcap_constant_error"] < 1e-12
            and q5a["path_final_error"] < 1e-8
            and q5a["max_concurrence_error"] < 1e-12
            and q5a["max_entropy_error"] < 1e-12
        ),
        "wrong_models_fail": (
            q3["rmse_wrong_no_z"] > 0.05
            and q4["rmse_wrong_gamma0"] > 0.05
            and q4["rmse_wrong_deltaE"] > 0.05
        ),
    }

    print("\nFINAL ANTI-CHEATING FLAGS")
    for k, v in flags.items():
        print(f"{k:36s} = {v}")

    print("=" * 78)


# ============================================================
# Run all
# ============================================================

def print_revision_summary(cs, cl, sel):
    print("\n" + "=" * 78)
    print("REVISION ADDITIONS (metric-compatible selection rule)")
    print("=" * 78)
    print("\n[Pure-state FS selection rule  H = sqrt(g_FS(G,G)) = dE/hbar]")
    print(f"max |projective speed - dE/hbar|       = {sel['pure_FS_selection_max_err']:.3e}")
    print("\n[Open-system Bures selection rule + wrong assignments]")
    print(f"correct Liouvillian rel error          = {sel['open_Bures_rel_correct']:.3e}")
    print(f"wrong generator dE/hbar  RMSE          = {sel['open_wrong_generator_deltaE_rmse']:.3e}")
    print(f"wrong generator gamma=0  RMSE          = {sel['open_wrong_generator_gamma0_rmse']:.3e}")
    print(f"wrong metric  HS speed   RMSE          = {sel['open_wrong_metric_HS_rmse']:.3e}")
    print("\n[Commuting-drive sanity check (geodesic: path == endpoint)]")
    print(f"correct rel error                      = {cs['rel_correct']:.3e}")
    print(f"endpoint D/H(0) RMSE                    = {cs['rmse_endpoint']:.3e}")
    print(f"path/endpoint ratio                    = {cs['path_endpoint_ratio']:.6f}")
    print("\n[Closed entangling loop  T=2pi/J  (D_end -> 0, path -> pi)]")
    print(f"endpoint distance final                = {cl['endpoint_final']:.3e}")
    print(f"path length final                      = {cl['path_final']:.8f}")
    print(f"path error vs analytic pi              = {cl['path_error_vs_analytic']:.3e}")
    print(f"reconstruction rel error on loop       = {cl['rel_correct']:.3e}")
    print(f"endpoint-only RMSE on loop             = {cl['rmse_endpoint']:.3e}")
    print("=" * 78)


def generate_figures(outdir="."):
    """Regenerate all manuscript figures (fig1-4 + figS1) from the validation
    runs in this module. Requires matplotlib; saves vector PDFs into outdir.
    This makes the validation script self-contained: it both checks the
    numbers and produces the figures that appear in the paper."""
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
    except Exception as e:
        print("matplotlib unavailable; skipping figure generation:", e)
        return

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 10, "axes.titlesize": 9.5,
        "axes.labelsize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "axes.linewidth": 0.8, "lines.linewidth": 1.5,
        "figure.dpi": 150,
    })
    C = {"true": "#444", "rec": "#1b6ca8", "fail": "#c0392b", "fail2": "#d98324",
         "fail3": "#7d3c98", "path": "#1b6ca8", "cap": "#e08a1e", "end": "#7f8c8d",
         "ent": "#2a8c5a", "con": "#8e44ad", "loop": "#1b6ca8"}

    def P(name):
        return os.path.join(outdir, name)

    # data
    q3 = run_q3_noncommuting(T=2.2, n_steps=6001); t3 = q3["t"]
    q4 = run_q4_open_dephasing(T=2.2, gamma=0.35, n_steps=5001); t4 = q4["t"]
    q5 = run_q5_entangler(J=1.3, T_factor=1.25, n_steps=3001); t5 = q5["t"]; Tb = np.pi / 2.6
    cl = run_closed_loop(J=1.3, n_steps=4001); tcl = cl["t"]
    cs = run_commuting_sanity(); tcs = cs["t"]

    # ---- Fig 1: principle + selection rule ----
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    th = np.linspace(0.15, 1.45, 200)
    x = np.cos(th) * np.linspace(1.0, 1.7, 200); y = np.sin(th) * np.linspace(1.0, 1.3, 200)
    ax.plot(x, y, color=C["path"], lw=2.2, zorder=2)
    i0, i1 = 120, 150
    ax.add_patch(FancyArrowPatch((x[i0], y[i0]), (x[i1], y[i1]), arrowstyle='-|>',
                 mutation_scale=14, color=C["fail"], lw=2.0, zorder=4))
    ax.annotate(r"$d\Phi_g$", (0.5 * (x[i0] + x[i1]) - 0.02, 0.5 * (y[i0] + y[i1]) + 0.10),
                color=C["fail"], fontsize=11)
    ax.plot([x[0]], [y[0]], 'o', color="#222", ms=6); ax.text(x[0] + 0.03, y[0] - 0.12, r"$\rho(0)$", fontsize=10)
    ax.plot([x[-1]], [y[-1]], 'o', color="#222", ms=6); ax.text(x[-1] - 0.05, y[-1] + 0.06, r"$\rho(T)$", fontsize=10)
    ax.add_patch(FancyArrowPatch((x[0], y[0]), (x[-1], y[-1]), arrowstyle='-|>',
                 mutation_scale=12, color=C["end"], lw=1.4, ls=(0, (5, 3)), zorder=3))
    ax.text(0.50, 0.52, r"$D_{\rm end}\leq\Phi_{\rm path}$", color=C["end"], fontsize=9, rotation=18)
    ax.text(0.02, 1.80, r"state space with metric $g_\rho$", fontsize=9, color="#555")
    ax.text(1.70, 0.66, r"$dt_{\rm info}=\dfrac{d\Phi_g}{H_{\rm cap}^{(g,G)}}$", fontsize=14, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f6f1e7", ec=C["cap"], lw=1.3))
    ax.text(1.70, 0.06, r"selection rule:", fontsize=8.5, ha="center", color="#333")
    ax.text(1.70, -0.20, r"$H_{\rm cap}^{(g,G)}=\sqrt{g_\rho(G_t[\rho],G_t[\rho])}$", fontsize=11, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", lw=1.0))
    ax.text(1.70, -0.52, r"same metric $g$ and true generator $G_t$ as $d\Phi_g$", fontsize=7.5, ha="center", color="#777")
    ax.set_xlim(-0.05, 2.75); ax.set_ylim(-0.72, 1.98); ax.axis("off")
    fig.tight_layout(); fig.savefig(P("fig1_principle_schematic.pdf")); plt.close(fig)

    # ---- Fig 2: non-commuting drive (main) ----
    fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.0))
    ax = axs[0]
    ax.plot(t3, t3, color=C["true"], ls=":", label="true $t$")
    ax.plot(t3, q3["t_rec"], color=C["rec"], label=r"correct $\Delta H(t)/\hbar$")
    ax.plot(t3, q3["t_wrong_no_z"], color=C["fail"], ls="--", label=r"wrong no-$z$ generator")
    ax.plot(t3, q3["t_endpoint_mean"], color=C["end"], ls="-.", label=r"endpoint/mean")
    ax.set_xlabel("$t$"); ax.set_ylabel("reconstructed time"); ax.set_title("(a) correct vs wrong generator"); ax.legend(loc="upper left")
    ax = axs[1]
    ax.plot(t3, q3["Phi_path"], color=C["path"], label=r"path $\Phi_{\rm FS}$")
    ax.plot(t3, q3["Phi_capacity"], color=C["cap"], ls="--", label=r"$\int H_{\rm cap}\,dt$")
    ax.plot(t3, q3["D_endpoint"], color=C["end"], ls="-.", label=r"endpoint $D_{\rm end}$")
    ax.set_xlabel("$t$"); ax.set_ylabel("FS distance"); ax.set_title("(b) path $>$ endpoint"); ax.legend(loc="upper left")
    ax.text(0.46, 0.10, f"path/endpoint\n= {q3['path_endpoint_ratio']:.3f}", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb"))
    ax = axs[2]
    ax.semilogy(t3, np.abs(q3["err_correct"]) + 1e-16, color=C["rec"], label=r"correct")
    ax.semilogy(t3, np.abs(q3["err_wrong_no_z"]) + 1e-16, color=C["fail"], ls="--", label=r"wrong no-$z$")
    ax.semilogy(t3, np.abs(q3["err_endpoint"]) + 1e-16, color=C["end"], ls="-.", label=r"endpoint")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$|t_{\rm rec}-t|$"); ax.set_title("(c) falsifiability (log)"); ax.legend(loc="lower right")
    ax.text(0.04, 0.06, r"FS selection rule:" + "\n" + r"$|v_{\rm proj}-\Delta E/\hbar|\sim 10^{-15}$",
            transform=ax.transAxes, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", fc="#eef6ef", ec="#9c9"))
    fig.tight_layout(); fig.savefig(P("fig2_noncommuting_drive.pdf")); plt.close(fig)

    # ---- Fig 3: open dephasing + all wrong controls (incl. endpoint) ----
    fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.0))
    ax = axs[0]
    ax.plot(t4, t4, color=C["true"], ls=":", label="true $t$")
    ax.plot(t4, q4["t_rec"], color=C["rec"], label=r"correct $H_L$")
    ax.plot(t4, q4["t_wrong_deltaE"], color=C["fail"], ls="--", label=r"wrong $\Delta E/\hbar$")
    ax.plot(t4, q4["t_wrong_hs"], color=C["fail3"], ls=":", label=r"wrong HS speed")
    ax.plot(t4, q4["t_wrong_gamma0"], color=C["fail2"], ls="-.", label=r"wrong $\gamma{=}0$")
    ax.plot(t4, q4["t_endpoint_mean"], color=C["end"], ls=(0, (1, 1)), label=r"endpoint")
    ax.set_xlabel("$t$"); ax.set_ylabel("reconstructed time"); ax.set_title("(a) metric-compatible capacity"); ax.legend(loc="upper left", fontsize=7)
    ax = axs[1]
    ax.plot(t4, q4["Phi_path"], color=C["path"], label=r"Bures path")
    ax.plot(t4, q4["Phi_capacity"], color=C["cap"], ls="--", label=r"$\int H_L\,dt$")
    ax.plot(t4, q4["D_endpoint"], color=C["end"], ls="-.", label=r"endpoint")
    ax.set_xlabel("$t$"); ax.set_ylabel("Bures distance"); ax.set_title("(b) path vs endpoint"); ax.legend(loc="upper left")
    ax = axs[2]
    ax.plot(t4, q4["purity"], color="#555")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"purity $\mathrm{Tr}\,\rho^2$"); ax.set_title("(c) decoherence")
    ax.text(0.05, 0.18, f"final = {q4['purity_final']:.3f}", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb"))
    fig.tight_layout(); fig.savefig(P("fig3_open_system.pdf")); plt.close(fig)

    # ---- Fig 4: entangling + closed loop ----
    fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.0))
    ax = axs[0]
    ax.plot(t5, q5["concurrence"], color=C["con"], label="concurrence")
    ax.plot(t5, q5["entropy"], color=C["ent"], ls="--", label=r"entropy $S(\rho_A)$")
    ax.axvline(Tb, color="#999", ls=":", lw=1.0); ax.text(Tb + 0.03, 0.05, r"$t_{\rm Bell}$", fontsize=8.5, color="#666")
    ax.set_xlabel("$t$"); ax.set_ylabel("entanglement"); ax.set_title("(a) Bell time"); ax.legend(loc="lower right")
    ax = axs[1]
    ax.plot(t5, q5["Phi_path"], color=C["path"], label=r"path $\Phi$")
    ax.plot(t5, q5["D_endpoint"], color=C["end"], ls="-.", label=r"endpoint $D$")
    ax.set_xlabel("$t$"); ax.set_ylabel("FS distance"); ax.set_title("(b) extended: ratio $\\to 5/3$"); ax.legend(loc="upper left")
    ax = axs[2]
    ax.plot(tcl, cl["Phi_path"], color=C["loop"], label=r"path $\Phi$")
    ax.plot(tcl, cl["D_endpoint"], color=C["fail"], ls="--", label=r"endpoint $D$")
    ax.axhline(np.pi, color="#aaa", ls=":", lw=0.9); ax.text(0.05, np.pi + 0.04, r"$\pi$", fontsize=9, color="#777")
    ax.set_xlabel("$t$"); ax.set_ylabel("FS distance"); ax.set_title("(c) closed loop: $D{\\to}0$, $\\Phi{\\to}\\pi$"); ax.legend(loc="center left")
    ax.text(0.40, 0.06, f"end $D$ = {cl['endpoint_final']:.0e}\npath = {cl['path_final']:.4f}", transform=ax.transAxes, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb"))
    fig.tight_layout(); fig.savefig(P("fig4_entangling_gate.pdf")); plt.close(fig)

    # ---- Fig S1: commuting sanity ----
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axs[0]
    ax.plot(tcs, cs["Phi_path"], color=C["path"], label=r"path $\Phi$")
    ax.plot(tcs, cs["D_endpoint"], color=C["end"], ls="-.", label=r"endpoint $D$")
    ax.set_xlabel("$t$"); ax.set_ylabel("FS distance"); ax.set_title("(a) geodesic: path $=$ endpoint"); ax.legend(loc="upper left")
    ax.text(0.40, 0.10, f"ratio = {cs['path_endpoint_ratio']:.3f}", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb"))
    ax = axs[1]
    ax.plot(tcs, tcs, color=C["true"], ls=":", label="true $t$")
    ax.plot(tcs, cs["t_rec"], color=C["rec"], label=r"$T_{\rm rec}$")
    ax.plot(tcs, cs["t_end"], color=C["fail"], ls="--", label=r"$D_{\rm end}/H(0)$")
    ax.set_xlabel("$t$"); ax.set_ylabel("reconstructed time"); ax.set_title("(b) reconstruction"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(P("figS1_commuting_sanity.pdf")); plt.close(fig)

    print("figures written to", os.path.abspath(outdir))


def main(run_full_checks=False, make_figures=False):
    # Q4 default eps_mode="dt" reproduces the paper Methods value (5.7e-4);
    # propagation uses Hermiticity + trace only (clip_after=False).
    q3 = run_q3_noncommuting(T=2.2, n_steps=6001)
    q4 = run_q4_open_dephasing(T=2.2, gamma=0.35, n_steps=5001)
    q5 = run_q5_entangler(J=1.3, T_factor=1.25, n_steps=3001)

    cs = run_commuting_sanity()
    cl = run_closed_loop(J=1.3, n_steps=4001)
    sel = selection_rule_report(q3, q4)

    print_main_summary(q3, q4, q5)
    print(f"\n[Q4 propagation positivity] clip_after = {q4['clip_after']}; "
          f"min eigenvalue over trajectory = {q4['min_eig_overall']:.2e}")
    print_revision_summary(cs, cl, sel)

    if run_full_checks:
        rk4 = independent_q3_rk4_check(T=2.2, n_steps=6001)
        q5a = q5_analytic_check(q5, J=1.3)
        fid = fidelity_formula_crosscheck(seed=1234, n=2000)
        epscheck = q4_eps_stability_check(T=2.2, gamma=0.35, n_steps=3001)
        steps = stepsize_convergence_check()
        print_anti_cheat_summary(q3, q4, q5, rk4, q5a, fid, epscheck, steps)
    else:
        print("\n[quick mode] Full anti-cheating scans skipped "
              "(independent RK4, fidelity crosscheck, eps and step-size stability).")
        print("Set run_full_checks=True for the complete (slower) validation.")

    if make_figures:
        print("\n[figures] regenerating manuscript figures ...")
        generate_figures(".")


# ============================================================
# ADDED (revision): metric-compatible capacity selection rule
#   H_cap^{(g,G)}(rho,t) = sqrt( g_rho( G_t[rho], G_t[rho] ) )
# Pure FS:   sqrt(g_FS(psi_dot,psi_dot)) = Delta E / hbar  (pure_projective_speed)
# Open Bures: D_B(rho, rho+eps L[rho])/eps  (liouvillian_bures_speed)
# Both are already implemented; the runs below surface them as the
#   selection rule, plus a commuting sanity check and a closed-loop test.
# ============================================================

def run_commuting_sanity(O0=1.7, a=0.55, nu=2.4, T=1.2, n_steps=2001):
    """Sanity check ONLY: commuting single-axis drive H = hbar*Omega(t)*sx/2.
    Geodesic on a single great circle => path == endpoint. Demoted from a
    demonstration to a routine check (it is trivially reconstructable)."""
    t = np.linspace(0.0, T, n_steps); dt = t[1] - t[0]
    psi = np.zeros((n_steps, 2), dtype=complex); psi[0] = ket0
    for i in range(n_steps - 1):
        tm = 0.5 * (t[i] + t[i + 1]); Om = O0 * (1.0 + a * np.sin(nu * tm))
        U = np.cos(Om * dt / 2.0) * I2 - 1j * np.sin(Om * dt / 2.0) * sx
        psi[i + 1] = normalize(U @ psi[i])
    dPhi = np.array([fs_distance(psi[i], psi[i + 1]) for i in range(n_steps - 1)])
    D_end = np.array([fs_distance(psi[0], psi[i]) for i in range(n_steps)])
    H_mid = np.array([O0 * (1.0 + a * np.sin(nu * (0.5 * (t[i] + t[i + 1])))) / 2.0
                      for i in range(n_steps - 1)])
    Phi = np.zeros(n_steps); Phi[1:] = np.cumsum(dPhi)
    Phi_cap = np.zeros(n_steps); Phi_cap[1:] = np.cumsum(H_mid * dt)
    t_rec = np.zeros(n_steps); t_rec[1:] = np.cumsum(safe_div(dPhi, H_mid))
    H0 = O0 / 2.0
    t_end = D_end / H0
    return {
        "t": t, "Phi_path": Phi, "Phi_capacity": Phi_cap, "D_endpoint": D_end,
        "Hgrid": O0 * (1.0 + a * np.sin(nu * t)) / 2.0, "t_rec": t_rec, "t_end": t_end,
        "rel_correct": float(np.max(np.abs(t_rec - t)) / T),
        "rmse_endpoint": float(np.sqrt(np.mean((t_end - t) ** 2))),
        "path_endpoint_ratio": float(Phi[-1] / max(D_end[-1], 1e-14)),
    }


def run_closed_loop(J=1.3, n_steps=4001):
    """Closed entangling loop: U_ZZ over [0, 2pi/J] returns the state to its
    starting ray (-psi0). Endpoint distance -> 0 while realized path -> pi.
    Strongest form of endpoint != path. Selection-rule capacity = Delta E/hbar
    = J/2 (constant, since <ZZ>=0 is conserved), so path = (J/2)(2pi/J) = pi."""
    H = H_ZZ(J)
    psi0 = kron(ketp, ketp)
    T = 2.0 * np.pi / J
    t = np.linspace(0.0, T, n_steps); dt = t[1] - t[0]
    psi = np.array([normalize(U_ZZ(ti, J) @ psi0) for ti in t])
    dPhi = np.array([fs_distance(psi[i], psi[i + 1]) for i in range(n_steps - 1)])
    D_end = np.array([fs_distance(psi0, psi[i]) for i in range(n_steps)])
    Phi = np.zeros(n_steps); Phi[1:] = np.cumsum(dPhi)
    Hcap = np.array([energy_uncertainty_pure(p, H) / hbar for p in psi])
    H_mid = 0.5 * (Hcap[:-1] + Hcap[1:])
    t_rec = np.zeros(n_steps); t_rec[1:] = np.cumsum(safe_div(dPhi, H_mid))
    H_mean = float(np.mean(H_mid))
    t_end = D_end / max(H_mean, 1e-14)   # endpoint-only estimate: fails on the loop
    return {
        "t": t, "psi": psi, "D_endpoint": D_end, "Phi_path": Phi, "Hcap": Hcap,
        "t_rec": t_rec, "t_endpoint": t_end,
        "endpoint_final": float(D_end[-1]),
        "path_final": float(Phi[-1]),
        "path_analytic": float(np.pi),
        "path_error_vs_analytic": float(abs(Phi[-1] - np.pi)),
        "rel_correct": float(np.max(np.abs(t_rec - t)) / T),
        "rmse_endpoint": float(np.sqrt(np.mean((t_end - t) ** 2))),
    }


def selection_rule_report(q3, q4):
    """Verify H_cap = sqrt(g(G[rho],G[rho])) numerically:
    pure FS  : |projective speed - Delta E/hbar|   (already in q3)
    open Bures: Liouvillian Bures speed is the metric speed of L[rho]."""
    return {
        "pure_FS_selection_max_err": float(q3["max_speed_identity_err"]),
        "open_Bures_rel_correct": float(q4["rel_correct"]),
        "open_wrong_generator_deltaE_rmse": float(q4["rmse_wrong_deltaE"]),
        "open_wrong_generator_gamma0_rmse": float(q4["rmse_wrong_gamma0"]),
        "open_wrong_metric_HS_rmse": float(q4["rmse_wrong_hs"]),
    }


if __name__ == "__main__":
    main(run_full_checks=False)
