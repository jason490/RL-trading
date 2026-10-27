"""Episode-safe recurrent replay indexing, independent of tensors and learners.

Insertion IDs refer to immutable transition payloads in an owning replay store.
Call ``resolve`` immediately before gathering payloads: overwritten IDs fail closed.
No padding, cached collection hidden state, optimizer or global RNG is used here.
"""
from dataclasses import dataclass

import numpy as np


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


@dataclass(frozen=True)
class TransitionRef:
    insertion_id: int
    environment: int
    episode: int
    step: int
    symbol: str
    terminated: bool
    truncated: bool

    @property
    def ended(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class SequenceRef:
    transitions: tuple[TransitionRef, ...]
    burn_in: int

    @property
    def loss_mask(self) -> tuple[bool, ...]:
        return tuple(i >= self.burn_in for i in range(len(self.transitions)))

    @property
    def bootstrap_mask(self) -> tuple[bool, ...]:
        # A truncated transition needs its final observation, not an auto-reset observation.
        return tuple(not t.terminated for t in self.transitions)

    @property
    def reset_after(self) -> tuple[bool, ...]:
        return tuple(t.ended for t in self.transitions)


class RecurrentReplayIndex:
    """Fixed-capacity global ring with independent per-environment episode lineage.

    A new environment starts at step zero. Episode IDs must strictly increase after
    an end flag, and symbols cannot change inside an episode. The owner supplies
    true final observations separately; this metadata cannot verify their content.
    """

    def __init__(self, capacity: int):
        _integer('capacity', capacity, 1)
        self.capacity = capacity
        self._slots: list[TransitionRef | None] = [None] * capacity
        self._last: dict[int, TransitionRef] = {}
        self._next = 0

    def append(self, *, environment: int, episode: int, step: int, symbol: str,
               terminated: bool = False, truncated: bool = False) -> TransitionRef:
        for name, value in [('environment', environment), ('episode', episode), ('step', step)]:
            _integer(name, value)
        if type(symbol) is not str or not symbol.strip():
            raise ValueError('symbol must be a nonempty string')
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError('end flags must be booleans')
        previous = self._last.get(environment)
        if previous is None or previous.ended:
            if step != 0 or (previous is not None and episode <= previous.episode):
                raise ValueError('new episode requires step zero and increasing episode ID')
        elif (episode != previous.episode or step != previous.step + 1
              or symbol != previous.symbol):
            raise ValueError('ongoing episode requires contiguous steps and unchanged symbol')
        ref = TransitionRef(self._next, environment, episode, step, symbol, terminated, truncated)
        self._slots[self._next % self.capacity] = ref
        self._last[environment] = ref
        self._next += 1
        return ref

    def resolve(self, sequence: SequenceRef) -> tuple[int, ...]:
        """Return physical slots only if every transition is still retained and valid."""
        if (not isinstance(sequence, SequenceRef) or type(sequence.transitions) is not tuple
                or not sequence.transitions):
            raise ValueError('nonempty SequenceRef required')
        _integer('burn_in', sequence.burn_in)
        if sequence.burn_in >= len(sequence.transitions):
            raise ValueError('at least one learning transition required')
        for i, ref in enumerate(sequence.transitions):
            if not isinstance(ref, TransitionRef) or type(ref.insertion_id) is not int:
                raise ValueError('invalid transition reference')
            if ref.insertion_id < 0 or self._slots[ref.insertion_id % self.capacity] != ref:
                raise ValueError('sequence was overwritten or does not match retained metadata')
            if i:
                prev = sequence.transitions[i - 1]
                if (prev.ended or ref.environment != prev.environment or ref.episode != prev.episode
                        or ref.symbol != prev.symbol or ref.step != prev.step + 1
                        or ref.insertion_id <= prev.insertion_id):
                    raise ValueError('sequence crosses an episode/environment/chronology boundary')
        return tuple(t.insertion_id % self.capacity for t in sequence.transitions)

    def windows(self, *, burn_in: int, unroll: int) -> tuple[SequenceRef, ...]:
        """Enumerate complete windows, preserving insertion order within each environment.

        Rebuild after writes; snapshot references must be resolved again before use.
        This correctness-first enumerator is not yet integrated into a learner.
        """
        _integer('burn_in', burn_in)
        _integer('unroll', unroll, 1)
        length = burn_in + unroll
        if length > self.capacity:
            raise ValueError('window exceeds total replay capacity')
        groups: dict[tuple[int, int], list[TransitionRef]] = {}
        for ref in sorted((s for s in self._slots if s is not None), key=lambda t: t.insertion_id):
            groups.setdefault((ref.environment, ref.episode), []).append(ref)
        result = []
        for refs in groups.values():
            for start in range(len(refs) - length + 1):
                sequence = SequenceRef(tuple(refs[start:start + length]), burn_in)
                self.resolve(sequence)
                result.append(sequence)
        return tuple(result)

    def sample(self, *, batch_size: int, burn_in: int, unroll: int,
               rng: np.random.Generator) -> tuple[SequenceRef, ...]:
        """Uniform complete-window sampling with replacement using caller-owned RNG."""
        _integer('batch_size', batch_size, 1)
        if not isinstance(rng, np.random.Generator):
            raise ValueError('caller-owned NumPy Generator required')
        windows = self.windows(burn_in=burn_in, unroll=unroll)
        if not windows:
            raise ValueError('no complete recurrent windows available')
        return tuple(windows[int(i)] for i in rng.integers(len(windows), size=batch_size))
