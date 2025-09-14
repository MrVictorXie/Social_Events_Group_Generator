from sympy import symbols, Eq, solve

# Define variables
a1, a2, b1, b2 = symbols('a1 a2 b1 b2')

# Define equations based on substitution
eq1 = Eq(-6*b1, 9*a1 + 9*a2 + 8)        # (A)
eq2 = Eq(6*a1, 9*b1 + 9*b2)             # (B)
eq3 = Eq(-6*b2, 3*a1 + 4*a2)            # (C)
eq4 = Eq(6*a2, 3*b1 + 4*b2 + 5)         # (D)

# Solve the system
solution = solve((eq1, eq2, eq3, eq4), (a1, a2, b1, b2))
# Print rounded results
for var, val in solution.items():
    print(f"{var} = {val.evalf():.4f}")