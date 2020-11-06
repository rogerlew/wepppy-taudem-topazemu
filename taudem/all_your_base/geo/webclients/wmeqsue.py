from urllib.request import urlopen

from ...all_your_base import isfloat


wmesque_url = 'https://wepp1.nkn.uidaho.edu/webservices/wmesque/'


def wmesque_retrieve(dataset, extent, fname, cellsize):
    global wmesque_url

    assert isfloat(cellsize)

    assert all([isfloat(v) for v in extent])
    assert len(extent) == 4

    extent = ','.join([str(v) for v in extent])

    if fname.lower().endswith('.tif'):
        fmt = 'GTiff'

    elif fname.lower().endswith('.asc'):
        fmt = 'AAIGrid'

    elif fname.lower().endswith('.png'):
        fmt = 'PNG'

    else:
        raise ValueError('fname must end with .tif, .asc, or .png')

    url = '{wmesque_url}{dataset}/?bbox={extent}&cellsize={cellsize}&format={format}'\
          .format(wmesque_url=wmesque_url, dataset=dataset,
                  extent=extent, cellsize=cellsize, format=fmt)

    try:
        output = urlopen(url, timeout=60)
        with open(fname, 'wb') as fp:
            fp.write(output.read())
    except Exception:
        raise Exception("Error retrieving: %s" % url)

    return 1
