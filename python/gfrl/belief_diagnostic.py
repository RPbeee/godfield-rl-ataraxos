"""Offline-only unused-hand labels and sample metrics; never policy inputs."""
import numpy as np

METRICS = ('size_accuracy', 'size_mae', 'presence_brier', 'presence_log_loss',
           'active_brier', 'active_log_loss', 'copies_mse')


def public_used(env, row):
    p = env.obs_schema()['positions']
    codes = env.obs_cards[row, p['opponent_used_start']:p['history_start'], 0]
    return np.bincount(codes[codes > 0], minlength=297).astype(np.int32)


def true_unused(env, row):
    result = env.label_opp_hand[row].astype(np.int32) - public_used(env, row)
    assert (result >= 0).all(), 'label does not contain public used cards'
    return result


def sample_metrics(samples, truth, cards=None):
    """Scalar scores plus additive ten-bin ECE statistics. Padding is excluded.

    Brier/ECE use raw sample frequencies. Only log loss uses Jeffreys
    add-0.5 smoothing: (presence_count + .5)/(accepted_samples + 1).
    Active cards are selected BEFORE smoothing. Size mode ties favor smaller n.
    """
    samples, truth = np.asarray(samples), np.asarray(truth)
    if samples.ndim != 2 or samples.shape[1:] != truth.shape or not len(samples):
        raise ValueError('require nonempty [samples,cards] and matching truth')
    sizes = samples.sum(1).astype(int)
    size = truth.sum()
    result = [float(np.bincount(sizes).argmax() == size), float(np.abs(sizes-size).mean())]
    cards = np.arange(1, len(truth)) if cards is None else np.asarray(cards)
    x, y = samples[:, cards], (truth[cards] > 0).astype(float)
    p = (x > 0).mean(0)
    smooth = ((x > 0).sum(0)+.5)/(len(samples)+1)
    brier = (p-y)**2
    loss = -(y*np.log(smooth)+(1-y)*np.log1p(-smooth))
    active = (y > 0) | (p > 0)
    result += [brier.sum(), loss.sum(), brier[active].sum(), loss[active].sum(),
               ((x.mean(0)-truth[cards])**2).sum()]
    bins = np.minimum((p*10).astype(int), 9)
    for weights in (None, p, y):
        result.extend(np.bincount(bins, weights=weights, minlength=10))
    return np.asarray(result, dtype=float)


def summarize(values, games, bootstrap=1000, seed=0):
    """Position-weighted means; percentile CIs resample entire games."""
    values = np.asarray(values, dtype=float).reshape(-1, len(METRICS)+30)
    games = np.asarray(games)
    if not len(values):
        return dict(positions=0, games=0, metrics=None)
    _, inv = np.unique(games, return_inverse=True)
    n = int(inv.max()+1)
    sums = np.zeros((n, values.shape[1]))
    np.add.at(sums, inv, values)
    counts = np.bincount(inv)

    def score(total, count):
        ece = np.abs(total[17:27]-total[27:37]).sum()/total[7:17].sum()
        return np.r_[total[:7]/count, ece]

    mean = score(sums.sum(0), counts.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty((bootstrap, len(mean)))
    for i in range(bootstrap):
        selected = rng.integers(n, size=n)
        draws[i] = score(sums[selected].sum(0), counts[selected].sum())
    intervals = np.quantile(draws, [.025, .975], axis=0)
    return dict(positions=len(values), games=n,
                metrics={key: dict(mean=float(mean[i]), ci95=intervals[:, i].tolist())
                         for i, key in enumerate((*METRICS, 'presence_ece'))})
