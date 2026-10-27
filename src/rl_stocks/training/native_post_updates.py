"""Post-optimizer checkpoints for native SB3 controls, without serializing hooks."""
from contextlib import contextmanager

from rl_stocks.training.recurrent_replay import _integer


@contextmanager
def native_snapshot_view(model):
    """Temporarily remove our instance train wrapper from SB3's save data."""
    wrapper=model.__dict__.pop('train',None)
    try:
        yield
    finally:
        if wrapper is not None:
            model.train=wrapper


@contextmanager
def post_update_checkpoints(model, *, every, checkpoint):
    """Call checkpoint only after native train returns successfully at a due step.

    Does not itself call train/learn/backward/optimizer. Actual learning remains
    the Colab worker's responsibility. Callback runs without the closure attached,
    so a native model.save cannot pickle a live callback/model cycle.
    """
    _integer('checkpoint cadence',every,1)
    if 'train' in model.__dict__ or not callable(checkpoint):
        raise ValueError('unmodified native train method and checkpoint callback required')
    original=model.train
    last=None
    def wrapped(*args,**kwargs):
        nonlocal last
        result=original(*args,**kwargs)
        step=model.num_timesteps
        _integer('actual transitions',step)
        if step>0 and step%every==0 and step!=last:
            with native_snapshot_view(model):
                checkpoint(model,step)
            last=step
        return result
    model.train=wrapped
    try:
        yield
    finally:
        model.__dict__.pop('train',None)
