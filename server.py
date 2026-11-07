"""server.py -- the federated training loop for every algorithm in the study."""
import math
import time

import numpy as np

from client import local_train
from compress import Compressor
from metrics import evaluate, classification_metrics
from model import fgsm, predict_proba
from privacy import calibrate_sigma, epsilon, ldp_perturb
from robust import robust_aggregate, apply_attack, poison_labels
from secure import SecureAggregator, PaillierAggregator

ADAPTIVE = ("fedadam", "fedyogi", "fedadagrad")


def _steps(n, cfg):
    B = min(cfg.batch_size, n)
    return cfg.rounds * cfg.local_epochs * math.ceil(n / B), B / n


def resolve_sigma(cfg, clients):
    """Noise multiplier: given directly or calibrated to cfg.target_eps."""
    if cfg.dp is None:
        return 0.0
    if cfg.noise_multiplier is not None:
        return cfg.noise_multiplier
    if cfg.target_eps is None:
        raise ValueError("set noise_multiplier or target_eps when dp is enabled")
    if cfg.dp == "ldp":
        return calibrate_sigma(cfg.target_eps, cfg.delta, 1.0, cfg.rounds)
    sig = []
    for c in clients:
        steps, q = _steps(len(c["y"]), cfg)
        sig.append(calibrate_sigma(cfg.target_eps, cfg.delta, q, steps))
    return max(sig)                                  # worst-case hospital


def epsilon_spent(cfg, clients, sigma):
    if cfg.dp is None:
        return float("nan")                          # no formal guarantee
    if cfg.dp == "ldp":
        return epsilon(1.0, sigma, cfg.rounds, cfg.delta)
    eps = []
    for c in clients:
        steps, q = _steps(len(c["y"]), cfg)
        eps.append(epsilon(q, sigma, steps, cfg.delta))
    return max(eps)


def _aggregate(vectors, weights, part, rnd, cfg, secagg, he):
    """Combine client vectors. Returns (aggregate, uplink_bytes_per_client)."""
    d = vectors.shape[1]
    if cfg.aggregator != "mean":
        return robust_aggregate(cfg.aggregator, vectors, cfg.n_byzantine,
                                cfg.trim_beta), 4 * d
    if secagg is not None:
        masked = [secagg.mask(int(k), weights[i] * vectors[i], part, rnd)
                  for i, k in enumerate(part)]
        return secagg.aggregate(masked), 4 * d + 32 * (len(part) - 1)
    if he is not None:
        enc = [he.encrypt(weights[i] * vectors[i]) for i in range(len(part))]
        return he.decrypt(he.add(enc)), he.ct_bytes * d
    return weights @ vectors, 4 * d


def run_federated(fd, cfg):
    if cfg.aggregator != "mean" and (cfg.secagg or cfg.he):
        raise ValueError("robust aggregation needs individual updates, which "
                         "secure aggregation / HE deliberately hide")
    rng = np.random.default_rng(cfg.seed)
    clients = list(fd.clients)
    if cfg.attack == "label_flip":
        clients = [poison_labels(c) if i in cfg.malicious_ids else c
                   for i, c in enumerate(clients)]
    K, d = len(clients), fd.dim
    sizes = np.array([len(c["y"]) for c in clients], float)
    sigma = resolve_sigma(cfg, clients)

    w = np.zeros(d)
    c_glob, c_loc = np.zeros(d), [np.zeros(d) for _ in range(K)]
    m, v = np.zeros(d), np.full(d, cfg.tau ** 2)
    comps = [Compressor(cfg.compression, cfg.quant_bits, cfg.topk_frac, d,
                        cfg.seed + k) for k in range(K)]
    secagg = SecureAggregator(d, seed=cfg.seed) if cfg.secagg else None
    he = PaillierAggregator(cfg.he_key_bits) if cfg.he else None

    history, comm, t0 = [], 0, time.perf_counter()
    for r in range(1, cfg.rounds + 1):
        part = (np.arange(K) if cfg.participation >= 1 else
                np.sort(rng.choice(K, max(1, round(cfg.participation * K)),
                                   replace=False)))
        p = sizes[part] / sizes[part].sum()
        deltas, taus, dcs, up = [], [], [], 4 * d
        for k in part:
            ep = (cfg.local_epochs if cfg.het_epochs is None else
                  int(rng.integers(cfg.het_epochs[0], cfg.het_epochs[1] + 1)))
            wk, tau = local_train(w, clients[k], cfg, rng, ep, sigma, c_glob, c_loc[k])
            delta = wk - w
            if cfg.algo == "scaffold":                    # control-variate option II
                c_new = c_loc[k] - c_glob + (w - wk) / (tau * cfg.lr)
                dcs.append(c_new - c_loc[k])
                c_loc[k] = c_new
            if k in cfg.malicious_ids and cfg.attack in ("sign_flip", "gaussian"):
                delta = apply_attack(cfg.attack, delta, rng, cfg.attack_scale)
            if cfg.dp == "ldp":
                delta = ldp_perturb(delta, cfg.clip, sigma, rng)
            delta, up = comps[k](delta)                   # compression (optional)
            deltas.append(delta)
            taus.append(tau)
        U, taus = np.stack(deltas), np.array(taus, float)
        if cfg.algo == "fednova":                         # normalise by local steps
            U = U / taus[:, None]
        agg, up_agg = _aggregate(U, p, part, r, cfg, secagg, he)
        if cfg.compression is None:                       # else: compressed size
            up = up_agg

        if cfg.algo == "fednova":
            w = w + cfg.server_lr * float(p @ taus) * agg
        elif cfg.algo in ADAPTIVE:
            m = cfg.beta1 * m + (1 - cfg.beta1) * agg
            g2 = agg ** 2
            if cfg.algo == "fedadagrad":
                v = v + g2
            elif cfg.algo == "fedyogi":
                v = v - (1 - cfg.beta2) * g2 * np.sign(v - g2)
            else:
                v = cfg.beta2 * v + (1 - cfg.beta2) * g2
            w = w + cfg.server_lr * m / (np.sqrt(v) + cfg.tau)
        else:                                  # fedavg / fedprox / fedpaq / scaffold
            w = w + cfg.server_lr * agg
        if cfg.algo == "scaffold":
            uni = np.full(len(part), 1.0 / len(part))
            agg_c, _ = _aggregate(np.stack(dcs), uni, part, r, cfg, secagg, he)
            c_glob = c_glob + (len(part) / K) * agg_c

        factor = 2 if cfg.algo == "scaffold" else 1       # SCAFFOLD ships c as well
        comm += len(part) * (factor * up + factor * 4 * d)
        h = evaluate(w, fd.X_test, fd.y_test, cfg.l2)
        h.update(round=r, comm_bytes=comm, time=time.perf_counter() - t0)
        history.append(h)

    final = dict(history[-1])
    adv = fgsm(w, fd.X_test, fd.y_test, cfg.eval_adv_eps)
    final["adv_accuracy"] = classification_metrics(
        fd.y_test, predict_proba(w, adv))["accuracy"]
    final.update(epsilon=epsilon_spent(cfg, clients, sigma), sigma=sigma,
                 comm_kb=comm / 1024.0, time_s=time.perf_counter() - t0)
    return {"history": history, "final": final, "w": w}
