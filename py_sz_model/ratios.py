"""Elemental ratios under the two protocols specified in the manuscript."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .randomness import MC_RANDOM_SEED, RandomState, generator

RATIOS = ['aFe/aSi', 'aFe/aAl', 'aSi/aAl', 'aFe/fFe', 'aSi/fFe', 'aAl/fFe']


def calculate_ratios(data: pd.DataFrame, method: str, n_mc_samples: int = 1000,
                     random_state: RandomState = MC_RANDOM_SEED) -> pd.DataFrame:
    """Use MC for Quaternary inputs and first-order propagation for Red Clay."""
    if method not in ('monte-carlo', 'first-order'):
        raise ValueError('Choose monte-carlo or first-order.')
    if n_mc_samples <= 0:
        raise ValueError('n_mc_samples must be positive.')
    result = data.copy()
    rng = generator(random_state)
    for ratio in RATIOS:
        a, b = ratio.split('/')
        am, bm, astd, bstd = [data[col].to_numpy(float) for col in (a,b,a+'_std',b+'_std')]
        if method == 'first-order':
            result[ratio] = am/bm
            result[ratio+'_std'] = np.hypot(astd/bm, am*bstd/bm**2)
        else:
            sampled_a = rng.normal(am[:,None], astd[:,None], (len(am),n_mc_samples))
            sampled_b = rng.normal(bm[:,None], bstd[:,None], (len(bm),n_mc_samples))
            sampled_ratio = sampled_a/sampled_b
            result[ratio] = sampled_ratio.mean(axis=1)
            result[ratio+'_std'] = sampled_ratio.std(axis=1, ddof=0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['monte-carlo','first-order'], required=True)
    parser.add_argument('--mc-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=MC_RANDOM_SEED)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Use a separate output file to preserve the recorded inputs.')
    result = calculate_ratios(pd.read_csv(args.input), args.method, args.mc_samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
