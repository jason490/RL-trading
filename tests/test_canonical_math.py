from decimal import Decimal, getcontext, localcontext
import math

import pytest

from rl_stocks.inference.canonical_math import bits32, certified_tanh, exact_dot, from_bits32, round32


def test_rounding_corrects_double_rounding_and_ties_even():
    with localcontext() as context:
        context.prec = 100
        midpoint = Decimal(1) + Decimal(2) ** -24
        assert round32(midpoint) == 1.
        assert round32(midpoint + Decimal(2) ** -90) == from_bits32(bits32(1.) + 1)
        assert round32(-midpoint) == -1.
        assert round32(-midpoint - Decimal(2) ** -90) == -from_bits32(bits32(1.) + 1)


def test_exact_dot_preserves_small_terms_and_context():
    before = getcontext().copy()
    assert exact_dot([1., 1., 1.], [100000000., 1., -100000000.]) == 1.
    assert exact_dot([1., -1.], [1., 1.]) == 0.
    assert getcontext().prec == before.prec
    assert getcontext().rounding == before.rounding
    for weights, values in [([], []), ([1.], []), ([float('nan')], [1.])]:
        with pytest.raises(ValueError):
            exact_dot(weights, values)


@pytest.mark.parametrize('value', [0., .01, .5, 1., 2., 8., 16., 100., from_bits32(1)])
def test_certified_tanh_is_odd_and_matches_reference_examples(value):
    result = certified_tanh(value)
    assert result == from_bits32(bits32(math.tanh(value)))
    assert certified_tanh(-value) == -result
    assert math.isfinite(result)


def test_nonfinite_and_overflow_rejected():
    with pytest.raises(ValueError):
        certified_tanh(float('inf'))
    with pytest.raises(ValueError):
        round32(Decimal('Infinity'))
    with pytest.raises(ValueError, match='overflow'):
        round32(Decimal(2) ** 128)


def test_rounding_uses_full_decimal_input_and_ignores_ambient_rounding():
    from decimal import ROUND_FLOOR
    with localcontext() as context:
        context.prec = 700
        value = Decimal(1) + Decimal(2) ** -24 + Decimal(10) ** -600
        expected = from_bits32(bits32(1.) + 1)
        context.prec = 6
        context.rounding = ROUND_FLOOR
        assert round32(value) == expected
        assert certified_tanh(1.) == from_bits32(bits32(math.tanh(1.)))


def test_reference_arithmetic_does_not_inherit_ambient_traps_or_exponent_limits():
    from decimal import Inexact, Rounded
    with localcontext() as context:
        context.traps[Inexact] = context.traps[Rounded] = True
        context.Emax = 2
        context.Emin = -2
        assert exact_dot([1., 1., 1.], [100000000., 1., -100000000.]) == 1.
        assert certified_tanh(1.) == from_bits32(bits32(math.tanh(1.)))
