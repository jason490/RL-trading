"""Reference binary32 arithmetic without native BLAS or platform tanh kernels.

Dots sum exact binary32 products before one nearest-even binary32 rounding.
Tanh uses correctly rounded Decimal exp and directed endpoint intervals, accepting
an answer only when both endpoints round to the same binary32 value.
"""
from decimal import (Context, Decimal, DivisionByZero, InvalidOperation, Overflow,
                     localcontext, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN)
import math
import struct


CONTRACT = 'binary32-exact-dot-certified-tanh-v1'
DOT_PRECISION = 512


def _context(precision):
    return Context(prec=precision, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999,
                   traps=[InvalidOperation, DivisionByZero, Overflow])


def bits32(value):
    return struct.unpack('!I', struct.pack('!f', value))[0]


def from_bits32(bits):
    return struct.unpack('!f', struct.pack('!I', bits))[0]


def input32(value):
    if not math.isfinite(value):
        raise ValueError('finite binary32 input required')
    try:
        result = from_bits32(bits32(value))
    except OverflowError as error:
        raise ValueError('finite binary32 input required') from error
    if not math.isfinite(result):
        raise ValueError('finite binary32 input required')
    return result


def round32(value):
    """Nearest-even binary32 from Decimal, correcting possible double rounding."""
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError('finite Decimal value required')
    with localcontext(_context(max(DOT_PRECISION, len(value.as_tuple().digits) + DOT_PRECISION))):
        maximum = Decimal.from_float(from_bits32(0x7f7fffff))
        overflow = maximum + Decimal(2) ** 103
        if abs(value) >= overflow:
            raise ValueError('binary32 overflow')
        if abs(value) > maximum:
            return from_bits32(0x7f7fffff if value > 0 else 0xff7fffff)
        candidate = bits32(float(value))
        candidates = {candidate}
        candidates.update(b for b in (candidate - 1, candidate + 1) if 0 <= b <= 0xffffffff)
        candidates = {b for b in candidates if b & 0x7f800000 != 0x7f800000}
        nearest = min(candidates, key=lambda b: (
            abs(value - Decimal.from_float(from_bits32(b))), b & 1, b))
        result = from_bits32(nearest)
        return 0.0 if result == 0 else result


def exact_dot(weights, values, bias=0.):
    if len(weights) == 0 or len(weights) != len(values) or len(weights) > 4096:
        raise ValueError('matching bounded nonempty dot vectors required')
    with localcontext(_context(DOT_PRECISION)):
        total = Decimal.from_float(input32(bias))
        for weight, value in zip(weights, values):
            total += Decimal.from_float(input32(weight)) * Decimal.from_float(input32(value))
        return round32(total)


def certified_tanh(value):
    value = input32(value)
    if value == 0:
        return 0.0
    if abs(value) >= 16:
        return 1.0 if value > 0 else -1.0
    x = Decimal.from_float(abs(value))
    with localcontext(_context(DOT_PRECISION)):
        twice_x = 2 * x
    for precision in (80, 160, 320, 640, 1280):
        with localcontext(_context(precision)):
            exponential = twice_x.exp()
        with localcontext(_context(precision + 20)) as context:
            half_ulp = Decimal(1).scaleb(exponential.adjusted() - precision + 1) / 2
            lower_exp, upper_exp = exponential - half_ulp, exponential + half_ulp
            # The additions/subtractions are exact at this guard precision.
            lower_num, lower_den = lower_exp - 1, lower_exp + 1
            upper_num, upper_den = upper_exp - 1, upper_exp + 1
            context.rounding = ROUND_FLOOR
            lower = lower_num / lower_den
            context.rounding = ROUND_CEILING
            upper = upper_num / upper_den
            lower32, upper32 = round32(lower), round32(upper)
        if bits32(lower32) == bits32(upper32):
            return lower32 if value > 0 else -lower32
    raise ValueError('tanh binary32 rounding could not be certified')
