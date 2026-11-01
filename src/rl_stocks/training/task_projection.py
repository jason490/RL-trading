"""Conservative fixed100k duration projections from research-geometry measurements."""
import math

from rl_stocks.training.task_conditioning import protocol
from rl_stocks.training.task_profile_worker import PROFILE_UPDATES


MARGIN = 1.5
SETUP_SECONDS = 900
RETRIEVAL_CLEANUP_SECONDS = 600


def project_screen(records, *, remaining_study_units, aggregate_rate):
    spec = protocol()
    if (len(records) != 3 or {r['task_role'] for r in records} != set(spec['roles'])
            or type(aggregate_rate) not in (int, float) or not math.isfinite(aggregate_rate)
            or aggregate_rate <= 0 or type(remaining_study_units) not in (int, float)
            or not math.isfinite(remaining_study_units) or remaining_study_units <= 0):
        raise ValueError('complete measured roles, positive rate and remaining allowance required')
    output = {}
    for record in records:
        learning = record['learning_seconds']
        evaluation = record['evaluation_seconds']
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in (learning, evaluation)):
            raise ValueError('finite positive measured component timings required')
        # Scaling the whole profile learning interval by update count also charges
        # its1000 warmup/environment steps again, conservatively.
        computation = MARGIN * (learning * 99000 / PROFILE_UPDATES + 10 * evaluation)
        total = math.ceil(computation + SETUP_SECONDS + RETRIEVAL_CLEANUP_SECONDS)
        output[record['task_role']] = dict(projected_allocation_seconds=total,
            within_two_hour_limit=total <= spec['execution']['allocation_deadline_seconds'])
    seconds = sum(r['projected_allocation_seconds'] for r in output.values())
    # Sequential CPU allocations; a supplied aggregate rate must bound the
    # observed profile rate. Add the ledger's0.01-unit/hour estimation margin.
    units = seconds * (aggregate_rate + .01) / 3600
    return dict(roles=output, projected_units=units,
        within_allowance=units < remaining_study_units,
        execution_admitted=all(r['within_two_hour_limit'] for r in output.values())
                           and units < remaining_study_units,
        margin=MARGIN, setup_seconds=SETUP_SECONDS,
        retrieval_cleanup_seconds=RETRIEVAL_CLEANUP_SECONDS,
        performance_promotion=False)
