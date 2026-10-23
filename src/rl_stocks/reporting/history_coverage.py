"""Outcome-free eligibility and expected year mass for full-window sampling."""
from collections import Counter
from datetime import date


def window_coverage(sessions, steps=512):
    """Uniform starts; assign each transition to its execution (next-bar) date.

    This is the expectation for a complete sampled window, not measured training
    visitation. The final incomplete rollout window need not have this mass.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError('positive integer steps required')
    sessions=list(sessions)
    if len(sessions)<=steps:
        raise ValueError('complete transition window required')
    for value in sessions:
        if not isinstance(value,str) or date.fromisoformat(value).isoformat()!=value:
            raise ValueError('ISO session dates required')
    if any(a>=b for a,b in zip(sessions,sessions[1:])):
        raise ValueError('strictly ordered unique sessions required')
    if sessions[-1]>'2020-12-31':
        raise ValueError('training-only sessions required')
    starts=len(sessions)-steps
    bar_counts=Counter(s[:4] for s in sessions)
    start_counts=Counter(s[:4] for s in sessions[:starts])
    mass=Counter()
    # Transition t executes on bar t+1, for starts in [0, starts-1].
    for t in range(len(sessions)-1):
        multiplicity=max(0,min(t,starts-1)-max(0,t-steps+1)+1)
        mass[sessions[t+1][:4]]+=multiplicity
    assert sum(mass.values())==starts*steps
    return dict(first_session=sessions[0],last_session=sessions[-1],bars=len(sessions),
        window_transitions=steps,eligible_starts=starts,bars_by_year=dict(sorted(bar_counts.items())),
        starts_by_year=dict(sorted(start_counts.items())),
        expected_execution_year_fraction={y:n/(starts*steps) for y,n in sorted(mass.items())},
        actual_year_visitation_measured=False)
