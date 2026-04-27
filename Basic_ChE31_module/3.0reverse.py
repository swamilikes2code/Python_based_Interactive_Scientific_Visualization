# ===================== REVERSE MODEL: RECYCLE ONLY =====================

def solve_with_recycle_inverse(m5):

    """
    Reverse model for the recycle case.

    Input:
      m5 = crystal production rate (kg/h)

    Internal concentrations are fixed from the problem statement.
    All other stream flow rates are calculated from m5.
    """

    # Given concentrations from the handout/problem
    x1K = 0.333
    x1W = 0.667
    x4K = 0.494
    x4W = 0.506
    x6K = 0.364
    x6W = 0.636
    x7K = 0.364
    x7W = 0.636
    cake_frac = 0.95

    if m5 <= 0:
        raise ValueError("m5 must be positive.")

    # Cake specification: m5 / (m5 + m6) = 0.95
    m6 = ((1 - cake_frac) / cake_frac) * m5

    # Overall K2CrO4 balance: m1*x1K = m5 + m6*x6K
    m1 = (m5 + m6 * x6K) / x1K

    # Overall mass balance: m1 = m3 + m5 + m6
    m3 = m1 - m5 - m6

    # Crystallizer K balance:
    # m4*x4K = m5 + (m6 + m7)*x6K, with m4 = m5 + m6 + m7
    # This reduces to m4*(x4K - x6K) = m5*(1 - x6K)
    m4 = m5 * (1 - x6K) / (x4K - x6K)

    # Evaporator overall balance: m2 = m3 + m4
    m2 = m3 + m4

    # Mixer overall balance: m2 = m1 + m7
    m7 = m2 - m1

    if m1 <= 0 or m2 <= 0 or m3 <= 0 or m4 <= 0 or m6 < 0 or m7 < 0:
        raise ValueError("Calculated a nonphysical negative flow rate.")

    # Mixed-feed composition
    x2K = (m1 * x1K + m7 * x7K) / m2
    x2W = 1 - x2K

    return {
        "m1_fresh_feed": m1,
        "x1K_fresh_feed": x1K,
        "x1W_fresh_feed": x1W,
        "m2_mixed_feed_to_evaporator": m2,
        "x2K_mixed_feed": x2K,
        "x2W_mixed_feed": x2W,
        "m3_evaporated_water": m3,
        "m4_to_crystallizer": m4,
        "x4K_after_evap": x4K,
        "x4W_after_evap": x4W,
        "m5_crystals": m5,
        "m6_solution_in_cake": m6,
        "x6K_solution_in_cake": x6K,
        "x6W_solution_in_cake": x6W,
        "m7_recycle": m7,
        "x7K_recycle": x7K,
        "x7W_recycle": x7W,
    }


# ===================== MAIN =====================

if __name__ == "__main__":

    m5 = 1470

    solution = solve_with_recycle_inverse(m5)

    print("===== REVERSE MODEL (RECYCLE ONLY) =====")
    for k, v in solution.items():
        print(k, round(v, 4))
