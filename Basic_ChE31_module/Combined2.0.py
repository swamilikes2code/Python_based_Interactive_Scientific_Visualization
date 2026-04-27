#New Combined model 2.0
#  ===================== NO RECYCLE =====================

def solve_no_recycle(m1, x1K, m2):

    """
    Inputs:
      m1   = fresh feed flow rate (kg/h)
      x1K  = K2CrO4 mass fraction in fresh feed
      m2   = evaporated water (kg/h)  <-- CONTROL VARIABLE
    """

    xsolK = 0.364
    cake_crystal_frac = 0.95

    # ---------- INPUT CHECKS ----------
    if m1 <= 0:
        raise ValueError("Throughput (m1) must be positive.")

    if not (0 < x1K < 1):
        raise ValueError("x1K must be between 0 and 1.")

    if not (0 < m2 < m1):
        raise ValueError("m2 must be between 0 and m1.")

    # ---------- EVAPORATOR ----------
    m3 = m1 - m2   # concentrated solution to crystallizer

    if m3 <= 0:
        raise ValueError("No liquid left after evaporation.")

    x3K = (m1 * x1K) / m3

    if x3K <= xsolK:
        raise ValueError("Not supersaturated → no crystals form.")

    # ---------- CRYSTALLIZER ----------
    d = (1 - cake_crystal_frac) / cake_crystal_frac

    m4 = m3 * (x3K - xsolK) / (1 - xsolK + d * xsolK)

    if m4 < 0:
        raise ValueError("Negative crystal rate.")

    m5 = d * m4
    m6 = m3 - m4 - m5

    if m6 < 0:
        raise ValueError("Negative filtrate.")

    return {
        "Streams": {
            "m1_fresh_feed": m1,
            "m2_evaporated_water": m2,
            "m3_to_crystallizer": m3,
            "x3K_after_evap": x3K,
            "m4_crystals": m4,
            "m5_solution_in_cake": m5,
            "m6_filtrate": m6
        }
    }


# ===================== WITH RECYCLE =====================

def solve_with_recycle(m1, x1K, m3, xsolK, cake_crystal_frac):

    if m1 <= 0:
        raise ValueError("m1 must be positive.")

    if not (0 < x1K < 1):
        raise ValueError("x1K must be between 0 and 1.")

    if not (0 < m3 < m1):
        raise ValueError("m3 must be between 0 and m1.")

    if not (0 < cake_crystal_frac < 1):
        raise ValueError("cake_crystal_frac must be between 0 and 1.")

    x4K = 0.494

    if x4K <= xsolK:
        raise ValueError("Not supersaturated.")

    d = (1 - cake_crystal_frac) / cake_crystal_frac

    # Example 4.5-2 assumes no filtrate leaves the system, so all mother
    # liquor not trapped in the cake is recycled.
    m6 = 0

    # Overall K2CrO4 balance with the cake specification.
    m5 = (m1 * x1K) / (1 + d * xsolK)
    m_solution_in_cake = d * m5

    # Use the crystallizer feed composition from the example to get the
    # crystallizer inlet flow directly.
    m4 = m5 * (1 - xsolK) / (x4K - xsolK)

    if m4 <= 0:
        raise ValueError("Negative crystallizer feed flow.")

    m2 = m4 + m3
    m7 = m2 - m1

    if m7 < 0:
        raise ValueError("Negative recycle flow.")

    x2K = (m1 * x1K + m7 * xsolK) / m2

    return {
        "m1_fresh_feed": m1,
        "m2_mixed_feed_to_evaporator": m2,
        "x2K_mixed_feed": x2K,
        "m3_evaporated_water": m3,
        "m4_to_crystallizer": m4,
        "x4K_after_evap": x4K,
        "m5_crystals": m5,
        "m6_filtrate": m6,
        "m_solution_in_cake": m_solution_in_cake,
        "m7_recycle": m7,
        "recycle_ratio": m7 / m1
    }


# ===================== COMPARISON =====================

def compare_systems():

    m1 = 4500
    x1K = 0.333
    m2 = 1466.6
    m3 = 2950
    xsolK = 0.364
    cake = 0.95

    noR = solve_no_recycle(m1, x1K, m2)
    R = solve_with_recycle(m1, x1K, m3, xsolK, cake)

    print("---- NO RECYCLE ----")
    for k, v in noR["Streams"].items():
        print(k, round(v, 4))

    print("\n---- WITH RECYCLE ----")
    for k, v in R.items():
        print(k, round(v, 4))

    print("\n==================== COMPARISON REPORT ====================")
    print(f"{'Variable':<30}{'No Recycle':<20}{'With Recycle':<20}")
    print("="*70)

    comparison_map = {
        "evaporated_water": ("m2_evaporated_water", "m3_evaporated_water"),
        "to_crystallizer": ("m3_to_crystallizer", "m4_to_crystallizer"),
        "crystals": ("m4_crystals", "m5_crystals"),
        "solution_in_cake": ("m5_solution_in_cake", "m_solution_in_cake"),
        "filtrate": ("m6_filtrate", "m6_filtrate"),
        "recycle": (None, "m7_recycle")
    }

    noR_streams = noR["Streams"]

    for label, (noR_key, R_key) in comparison_map.items():
        noR_val = noR_streams.get(noR_key, 0) if noR_key else 0
        R_val = R.get(R_key, 0)
        print(f"{label:<30}{round(noR_val,4):<20}{round(R_val,4):<20}")

    print("="*70)


# ===================== MAIN =====================

if __name__ == "__main__":
    compare_systems()
