
def probability_of_occurrence(return_interval, period_of_interest, pct=True):
    prob = 1.0 - (1.0 - 1.0 / return_interval) ** period_of_interest
    if prob < 0.0:
        prob = 0.0
    elif prob > 1.0:
        prob = 1.0

    if pct:
        prob *= 100.0
    return prob


def weibull_series(recurrence, years):
    """
    this came from Jim F.'s code. recurrence is a list of recurrence intervals. years is the number
    of years in the simulation. For each RI it determines the rank event index to estimate the return period.

    Not sure where Jim got it.
    """
    recurrence = sorted(recurrence)

    rec = {}
    i = 0
    rankind = years
    orgind = years + 1
    reccount = 0

    while i < len(recurrence) and rankind >= 2.5:
        retperiod = recurrence[i]
        rankind = float(years + 1) / retperiod
        intind = int(rankind) - 1

        if intind < orgind:
            rec[retperiod] = intind
            orgind = intind
            reccount += 1

        i += 1

    return rec