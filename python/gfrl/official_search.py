"""Shared offline/live search options and public-only diagnostics."""
import numpy as np

from .actor import Actor, PREFIX_FIELDS
from .model import OBS_FIELDS
from .search_belief import add_belief_arguments, variant_options, validate_variants
from .search import SearchConfig, SearchAgent, load_search_belief


def add_search_arguments(parser, *, advisor=False):
    add_belief_arguments(parser)
    defaults = SearchConfig()
    parser.add_argument('--belief', default='uniform', help='uniform / b1 / belief checkpoint (oracle forbidden)')
    parser.add_argument('--fit')
    for name in ('worlds', 'depth', 'candidates', 'candidate_samples'):
        default = (dict(depth=3, candidates=8).get(name, getattr(defaults, name))
                   if advisor else getattr(defaults, name))
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=default)
    for name in ('alpha', 'beta'):
        parser.add_argument('--'+name, type=float, default=getattr(defaults, name))
    parser.add_argument('--leaf-view', choices=('actor', 'searcher'), default='actor')
    parser.add_argument('--search-response', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)


def search_config(args):
    assert not getattr(args, 'diagnostic_true_hand_nll', False), 'true-hand diagnostics are forbidden in official bot/advisor'
    validate_variants(**variant_options(args))
    if args.belief.lower() == 'oracle':
        raise ValueError('oracle is forbidden for official advice')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    return SearchConfig(**{k: getattr(args, k) for k in (
        'worlds', 'depth', 'candidates', 'candidate_samples', 'alpha', 'beta', 'leaf_view', 'search_response')},
        select='argmax' if args.greedy else 'sample')


def search_agent(model, args, *, seed=0):
    config = search_config(args)  # Reject privileged options before loading anything.
    agent = SearchAgent(model, load_search_belief(args.belief, args.fit, args.device, **variant_options(args)), args.device,
                        config=config, seed=seed, num_threads=args.threads)
    agent.actor.sample = not args.greedy
    return agent


def root_matches(root, env):
    return all(np.array_equal(getattr(root, k), getattr(env, k))
               for k in (*OBS_FIELDS, *PREFIX_FIELDS, 'actor_seat', 'substep'))


def empty_diagnostics():
    return dict(search_worlds=0, search_candidates=0, search_seconds=0.,
                pi_search_top=[], pi_theta_top=[])


def public_report(agent, env):
    """Candidate paths refer only to own public cursor choices; no world data."""
    reports = getattr(agent, 'last_search', [])
    if not reports:
        failure = getattr(agent, 'last_candidate_diagnostics', {}).get(0, {}).get('failure')
        return dict(search_worlds=0, search_candidates=0, pi_search_top=[], pi_theta_top=[],
                    **({'search_failure_reason': failure} if failure else {}))
    report = reports[0]; candidates = report['candidates']
    lp = np.array([c['log_pi'] for c in candidates]); pi = np.exp(lp-lp.max()); pi /= pi.sum()
    def top(probs):
        from .official import OfficialDecision, KINDS
        output = []
        for i in np.argsort(-np.asarray(probs), kind='stable')[:3]:
            path = candidates[i]['path']
            cursor = OfficialDecision(env.public, env.events, env.history_len,
                excluded_ids=[h['id'] for i,h in enumerate(env.public['hand'])
                              if i in env.restrictions['excluded_slots']],
                excluded_kinds=[KINDS[k]
                                for k in env.restrictions['excluded_kinds']],
                exclude_enemy_default_self=env.restrictions['exclude_enemy_default_self'])
            for choice in path:
                cursor.step(np.array([choice], np.int32))
            if not cursor.complete:
                raise RuntimeError('incomplete_public_candidate')
            semantic = cursor.semantic_action()
            output.append(dict(path=path, probability=float(probs[i]),
                action={k: semantic[k] for k in ('kind', 'cards', 'target', 'allocation')}))
        return output
    return dict(search_worlds=max(c['valid_worlds'] for c in candidates),
                search_candidates=len(candidates), pi_search_top=top(report['probabilities']),
                pi_theta_top=top(pi))


def policy_at_prefix(actor, env, rows):
    """Re-encode the public observation when abandoning a root mid-selection."""
    if isinstance(actor, Actor) and actor.env is not env and np.any(env.substep[rows] != 0):
        actor._bind(env)
        buffers = {k: getattr(env, k) for k in (*OBS_FIELDS, *PREFIX_FIELDS)}
        # substep is an Actor cache marker, not a model input. Refresh the cache
        # without changing the real cursor, its prefix, or the policy RNG.
        buffers['substep'] = np.zeros_like(env.substep)
        return actor.act(rows, buffers=buffers).choices
    return actor(env, rows)
