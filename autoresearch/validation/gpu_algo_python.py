"""
Python reference implementation of the GPU kernel's simulation algorithm.
Used for cross-validation chain:
  Python (this) <-> Mojo CPU (Python RNG) <-> Mojo CPU (LCG) <-> GPU

Uses per-agent random.Random instances, matching the GPU kernel's per-agent
RNG structure. All arithmetic in numpy.float32 to match Mojo/GPU Float32.

Grid: 33x33, 9-way movement, highest-index arrest selection.
"""

import math
import random
import numpy as np

f32 = np.float32

SUPPORT = 0
ACTIVE = 1
OPPOSE = 2
JAILED = 3
SECURITY_COND = 4

GRID_W = 33
GRID_H = 33
GRID_SIZE = GRID_W * GRID_H


def sigmoid(x):
    x = f32(x)
    neg_x = f32(-x)
    # Clamp to avoid overflow in exp
    if neg_x > f32(80):
        return f32(0)
    return f32(f32(1) / (f32(1) + f32(np.exp(neg_x))))


def run_sim(seed, citizen_density=0.7, sec_density=0.0, pp_mean=0.0,
            model_eps=0.5, threshold=2.94444, max_jail=100, n_steps=50,
            vision=7):
    """Run GPU algorithm on CPU with per-agent random.Random.

    Returns list of tuples: (active, support, oppose, jail, revolution)
    one per step.
    """
    n_citizens = round(GRID_SIZE * citizen_density)
    n_security = round(GRID_SIZE * sec_density)
    n_agents = n_citizens + n_security

    threshold_f = f32(threshold)
    threshold_sig = sigmoid(threshold_f)
    pp_mean_f = f32(pp_mean)
    model_eps_f = f32(model_eps)

    # Per-agent state arrays
    cond = [np.int32(SUPPORT)] * n_agents
    next_cond = [np.int32(SUPPORT)] * n_agents
    pos_x = [0] * n_agents
    pos_y = [0] * n_agents
    private_pref = [f32(0)] * n_agents
    eps_arr = [f32(0)] * n_agents
    eps_prob_arr = [f32(0)] * n_agents
    oppose_th = [f32(0)] * n_agents
    active_th = [f32(0)] * n_agents
    jail_sent = [0] * n_agents
    activation_val = [f32(0)] * n_agents

    # Per-agent RNG: agent i seeded with seed * 1000003 + i
    agent_rngs = [random.Random(seed * 1000003 + i) for i in range(n_agents)]

    # Initialize citizens
    for i in range(n_citizens):
        rng = agent_rngs[i]
        pos_x[i] = rng.randrange(GRID_W)
        pos_y[i] = rng.randrange(GRID_H)

        private_pref[i] = f32(rng.gauss(float(pp_mean_f), 1.0))
        e = f32(rng.gauss(0.0, float(model_eps_f)))
        eps_arr[i] = e
        eps_prob_arr[i] = sigmoid(e)
        t1 = f32(rng.gauss(float(threshold_f), float(e)))
        t2 = f32(rng.gauss(float(threshold_f), float(e)))
        if t1 < t2:
            oppose_th[i] = t1
            active_th[i] = t2
        else:
            oppose_th[i] = t2
            active_th[i] = t1

    # Initialize security
    for i in range(n_citizens, n_agents):
        rng = agent_rngs[i]
        pos_x[i] = rng.randrange(GRID_W)
        pos_y[i] = rng.randrange(GRID_H)
        cond[i] = np.int32(SECURITY_COND)
        next_cond[i] = np.int32(SECURITY_COND)

    revolution = False
    step_data = []

    for step in range(n_steps):
        if revolution:
            step_data.append(step_data[-1])
            continue

        # Phase 1: Citizens scan neighbors and determine condition
        for i in range(n_citizens):
            if jail_sent[i] > 0 or cond[i] == JAILED:
                continue

            ax, ay = pos_x[i], pos_y[i]
            actives = 1
            opposed = 0
            support_cnt = 1
            security = 0

            for j in range(n_agents):
                if j == i:
                    continue
                dx = abs(pos_x[j] - ax)
                dy = abs(pos_y[j] - ay)
                if dx > GRID_W // 2:
                    dx = GRID_W - dx
                if dy > GRID_H // 2:
                    dy = GRID_H - dy
                if dx <= vision and dy <= vision:
                    if j < n_citizens:
                        c = cond[j]
                        if c == ACTIVE:
                            actives += 1
                        elif c == OPPOSE:
                            opposed += 1
                        elif c == SUPPORT:
                            support_cnt += 1
                    else:
                        security += 1

            ep = eps_arr[i]
            ep_prob = eps_prob_arr[i]
            active_ratio = f32(f32(actives + opposed) / f32(support_cnt))
            base = f32(f32(actives) + f32(opposed) * ep_prob)
            exponent = f32(f32(1) / f32(ep * ep + f32(1)))
            perception = f32(base ** exponent)
            arrest_prob = f32(f32(1) - f32(np.exp(
                f32(f32(-2.3) * f32(security) / f32(actives) * f32(2) * ep_prob)
            )))
            opinion = f32(-private_pref[i] + perception * active_ratio)

            rand_act = f32(agent_rngs[i].random())

            activation_val[i] = sigmoid(opinion)
            active_level = f32(sigmoid(f32(opinion - active_th[i])) - arrest_prob)
            oppose_level = f32(sigmoid(f32(opinion - oppose_th[i])) - arrest_prob)

            if active_level > rand_act:
                next_cond[i] = np.int32(ACTIVE)
            elif oppose_level > rand_act:
                next_cond[i] = np.int32(OPPOSE)
            else:
                next_cond[i] = np.int32(SUPPORT)

        # Phase 2: Advance citizens
        for i in range(n_citizens):
            if jail_sent[i] > 0:
                jail_sent[i] -= 1
                continue
            elif cond[i] == JAILED:
                cond[i] = np.int32(SUPPORT)
                pos_x[i] = agent_rngs[i].randrange(GRID_W)
                pos_y[i] = agent_rngs[i].randrange(GRID_H)

            cond[i] = next_cond[i]
            choice = agent_rngs[i].randrange(9)
            dx = choice // 3 - 1
            dy = choice % 3 - 1
            pos_x[i] = (pos_x[i] + dx + GRID_W) % GRID_W
            pos_y[i] = (pos_y[i] + dy + GRID_H) % GRID_H

        # Phase 3: Security arrest and move
        for s in range(n_citizens, n_agents):
            sx, sy = pos_x[s], pos_y[s]
            best_active = -1
            best_oppose = -1

            for j in range(n_citizens):
                dx = abs(pos_x[j] - sx)
                dy = abs(pos_y[j] - sy)
                if dx > GRID_W // 2:
                    dx = GRID_W - dx
                if dy > GRID_H // 2:
                    dy = GRID_H - dy
                if dx <= 1 and dy <= 1:
                    if cond[j] == ACTIVE:
                        if j > best_active:
                            best_active = j
                    elif (cond[j] == OPPOSE
                          and float(activation_val[j]) > float(threshold_sig)):
                        if j > best_oppose:
                            best_oppose = j

            arrestee = best_active if best_active >= 0 else best_oppose
            if arrestee >= 0:
                jail_sent[arrestee] = agent_rngs[s].randrange(max_jail)
                cond[arrestee] = np.int32(JAILED)

            choice = agent_rngs[s].randrange(9)
            dx = choice // 3 - 1
            dy = choice % 3 - 1
            pos_x[s] = (pos_x[s] + dx + GRID_W) % GRID_W
            pos_y[s] = (pos_y[s] + dy + GRID_H) % GRID_H

        # Count metrics
        active = sum(1 for i in range(n_citizens) if cond[i] == ACTIVE)
        support = sum(1 for i in range(n_citizens) if cond[i] == SUPPORT)
        oppose = sum(1 for i in range(n_citizens) if cond[i] == OPPOSE)
        jail = sum(1 for i in range(n_citizens) if cond[i] == JAILED)

        if (active + jail) / n_citizens >= 0.95:
            revolution = True

        step_data.append((active, support, oppose, jail, int(revolution)))

    return step_data


if __name__ == "__main__":
    # Quick smoke test
    result = run_sim(42, sec_density=0.0, model_eps=0.2)
    print(f"Sim: seed=42 sd=0.0 eps=0.2 | {len(result)} steps")
    for i, (a, s, o, j, r) in enumerate(result[:5]):
        print(f"  step {i}: active={a} support={s} oppose={o} jail={j} rev={r}")
    final = result[-1]
    print(f"  final: active={final[0]} support={final[1]} oppose={final[2]} "
          f"jail={final[3]} rev={final[4]}")
