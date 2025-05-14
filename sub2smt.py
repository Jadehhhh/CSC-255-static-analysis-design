# sub2smt.py -----------------------------------------------
from z3 import *

# 1. Declare 32-bit floating point variables
fp32 = Float32()
a, b = FP('a', fp32), FP('b', fp32)
rm = RNE()   # Default IEEE rounding mode (round-to-nearest-even)

# 2. Define constants
ZERO = FPVal(0.0, fp32)
FLT_MAX = FPVal(3.402823466e38, fp32)   # Max finite value for float32

# 3. Model the if / else logic
# cond1: a > 0 && b < 0
cond1 = And(fpGT(a, ZERO), fpLT(b, ZERO))

# cond2: a < 0 && b > 0
cond2 = And(fpLT(a, ZERO), fpGT(b, ZERO))

# Compute c
# If cond1: set c = 0
# Else if cond2: set c = 0
# Otherwise: c = a - b
c = If(cond1,
       ZERO,
       If(cond2,
          ZERO,
          fpSub(rm, a, b)))

# 4. (Optional) Input constraints: ensure a and b are not NaN or Inf
safe_in = [Not(fpIsInf(a)), Not(fpIsNaN(a)),
           Not(fpIsInf(b)), Not(fpIsNaN(b))]

# 5. Set up the goal: check if c becomes Inf or NaN
goal = Or(fpIsInf(c), fpIsNaN(c))
s = Solver()
s.add(safe_in)   # Optional: restrict a and b to normal values
s.add(goal)

print("Checking subtraction for Inf/NaN...")
if s.check() == sat:
    m = s.model()
    print("> Potential Inf/NaN detected!")
    print("  a =", m[a])
    print("  b =", m[b])
    print("  c =", m.eval(c, model_completion=True))
else:
    print("> Proven: no Inf/NaN can occur (UNSAT)")
