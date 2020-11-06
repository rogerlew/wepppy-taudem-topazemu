
def determine_wateryear(y, j=None, mo=None):
    if j is not None:
        mo = int((datetime(int(y), 1, 1) + timedelta(int(j))).month)

    if int(mo) > 9:
        return int(y) + 1

    return int(y)
