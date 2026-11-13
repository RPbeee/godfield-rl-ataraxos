"""The FP32 lambda-return / GAE recurrences, one GPU lane per own trajectory."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _targets(Order, Ends, Prob, Outcome, Target, Advantage, G,
             LV:tl.constexpr, LA:tl.constexpr, GAMMA:tl.constexpr, B:tl.constexpr=128):
    group = tl.program_id(0)*B+tl.arange(0, B)
    end = tl.load(Ends+group, group<G, 0)
    start = tl.load(Ends+group-1, (group<G)&(group>0), 0)
    row = tl.load(Order+end-1, group<G, 0)
    outcome = tl.load(Outcome+row, group<G, 0).to(tl.int32)
    p0 = (outcome == 1).to(tl.float32)
    p1 = (outcome == -1).to(tl.float32)
    p2 = (outcome == 0).to(tl.float32)
    t0, t1, t2 = p0, p1, p2
    nv = outcome.to(tl.float32)
    na = tl.full((B,), 0., tl.float32)
    pos = end-1
    while tl.sum(((pos >= start)&(group<G)).to(tl.int32), 0) > 0:
        active = (pos >= start)&(group<G)
        row = tl.load(Order+pos, active, 0)
        t0 = (1-LV)*p0+LV*t0
        t1 = (1-LV)*p1+LV*t1
        t2 = (1-LV)*p2+LV*t2
        if GAMMA != 1:
            t0 = t0*GAMMA
            t1 = t1*GAMMA
            t2 = t2*GAMMA+(1-GAMMA)
        p0 = tl.load(Prob+row*3, active, 0)
        p1 = tl.load(Prob+row*3+1, active, 0)
        p2 = tl.load(Prob+row*3+2, active, 0)
        value = p0-p1
        if GAMMA == 1:
            na = nv-value+LA*na
        else:
            na = GAMMA*nv-value+(GAMMA*LA)*na
        nv = value
        tl.store(Target+row*3, t0, active)
        tl.store(Target+row*3+1, t1, active)
        tl.store(Target+row*3+2, t2, active)
        tl.store(Advantage+row, na, active)
        pos -= 1


def compute_targets(batch, lambda_v=.8, lambda_a=.5, gamma=1.):
    if not 0 < gamma <= 1:
        raise ValueError('gamma must be in (0,1]')
    if not 0 <= lambda_v <= 1 or not 0 <= lambda_a <= 1:
        raise ValueError('lambda must be in [0,1]')
    c = batch.columns
    n = len(batch)
    if not n:
        return (torch.empty((0, 3), device=c['env'].device, dtype=torch.float32),
                torch.empty(0, device=c['env'].device, dtype=torch.float32))
    order = torch.arange(n, device=c['env'].device)
    # Three stable passes implement np.lexsort((arrival,seat,game,env))
    # without packing arbitrary uint64 game ids into a smaller integer key.
    for key in ('seat', 'game', 'env'):
        order = order[c[key][order].argsort(stable=True)]
    env, game, seat = (c[k][order] for k in ('env', 'game', 'seat'))
    boundary = (env[1:] != env[:-1]) | (game[1:] != game[:-1]) | (seat[1:] != seat[:-1])
    ends = torch.cat((torch.where(boundary)[0]+1, order.new_full((1,), n)))
    target = torch.empty((n, 3), device=order.device, dtype=torch.float32)
    advantage = torch.empty(n, device=order.device, dtype=torch.float32)
    _targets[(tr.cdiv(len(ends), 128),)](order, ends, c['old_value'].float(), c['outcome'],
        target, advantage, len(ends), lambda_v, lambda_a, gamma, enable_fp_fusion=False)
    return target, advantage
