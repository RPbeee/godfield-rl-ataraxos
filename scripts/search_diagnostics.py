"""Privileged diagnostics, deliberately kept outside the live gfrl package."""
import json
import numpy as np
import torch
from gfrl.transfer import tensor_batch
from gfrl.belief import Belief, IndependentBelief
from gfrl.belief_metrics import fit_baselines, likelihood_metrics
from gfrl.model import OBS_FIELDS
from gfrl.search import SearchAgent, load_search_belief, search_support
from gfrl.search_belief import wrap_belief


class DiagnosticSearchAgent(SearchAgent):
    def enable_diagnostic(self, args):
        if not args.fit:
            raise ValueError('true-hand diagnostic requires --fit and an AR --belief checkpoint')
        ar = load_search_belief(getattr(args, 'diagnostic_belief', args.belief), device=args.device)
        if not isinstance(ar, Belief) or isinstance(ar, IndependentBelief):
            raise ValueError('true-hand diagnostic requires an AR checkpoint')
        self.diagnostic_models = dict(ar_T1=ar,
            ar_tempered=wrap_belief(ar, getattr(args, 'belief_temperature', 1.)),
            b1=fit_baselines(torch.load(args.fit, map_location='cpu', weights_only=False))['B1'])
        self.diagnostic_models['hybrid'] = wrap_belief(ar, hybrid=True,
            count_temperature=getattr(args, 'belief_count_temperature', 1.),
            baseline=self.diagnostic_models['b1'])
        self.diagnostic_values = {k: [[], []] for k in self.diagnostic_models}
        self.diagnostic_cap = args.max_inference_batch or 128

    @torch.inference_mode()
    def search(self, src, roots, **kwargs):
        roots = np.asarray(roots, dtype=np.int32)
        roots = roots[search_support(src, roots, self.config.search_response) == 0]
        # Labels are full retained hands. Convert to the TRUE unused multiset,
        # just as world loading does; never feed privileged labels to inference.
        for lo in range(0, len(roots), self.diagnostic_cap):
            rr = roots[lo:lo+self.diagnostic_cap]
            labels = torch.as_tensor(np.array([src.unused_assignment(int(r),
                src.label_opp_hand[r].astype(np.int32)) for r in rr]), device=self.device)
            obs = tensor_batch({k: getattr(src, k)[rr] for k in OBS_FIELDS}, self.device)
            counts = labels.sum(1)
            for key, model in self.diagnostic_models.items():
                if key == 'b1':
                    lp = model.log_prob(labels.cpu())
                    correct = counts.cpu() == model.count_probs.argmax()
                else:
                    lp = model.log_prob(obs, labels)[0]
                    correct = counts == model.count_probs(obs).argmax(1)
                self.diagnostic_values[key][0].extend(lp.cpu().tolist())
                self.diagnostic_values[key][1].extend(correct.cpu().tolist())
        return super().search(src, roots, **kwargs)

    def diagnostic_report(self):
        def estimate(values):
            x = np.asarray(values, dtype=np.float64)
            if not len(x):
                return dict(mean=None, ci95=[None, None])
            mean = float(x.mean())
            if not np.isfinite(x).all():
                return dict(mean=mean, ci95=[mean, mean])
            delta = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.
            return dict(mean=mean, ci95=[float(mean-delta), float(mean+delta)])
        result = dict(privileged=True, target='true opponent unused multiset',
            interpretation='Direct model probability of unused cards; models were trained on retained hands including public used cards. This is not conditional/rejection-normalized likelihood.',
            ci_method='95% normal intervals over searched decisions (descriptive; within-game correlation is not adjusted)',
            positions=len(self.diagnostic_values['ar_T1'][0]), models={})
        for key, (lp, correct) in self.diagnostic_values.items():
            result['models'][key] = dict(nll=estimate(-np.asarray(lp)),
                log_likelihood=estimate(lp), hand_size_accuracy=estimate(correct))
            if lp:
                # Reuse the model-independent NLL implementation used by belief eval.
                result['models'][key]['multiset_nll'] = likelihood_metrics(lp, np.arange(len(lp)))['multiset_nll']
        return result


def diagnostic_markdown(report):
    if 'diagnostic_true_hand_nll' not in report:
        return ''
    return ('\n### diagnostic_true_hand_nll (privileged, offline only)\n\n```json\n'
            + json.dumps(report['diagnostic_true_hand_nll'], indent=2) + '\n```\n')
