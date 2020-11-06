from typing import Tuple, List, Dict, Union

import os
import sys
import shutil
import subprocess
import inspect
import json

from os.path import join as _join
from os.path import split as _split
from os.path import exists as _exists
import math

import operator

from collections import Counter

from osgeo import gdal, ogr, osr
import utm

import numpy as np
from scipy.ndimage import label

from pprint import pprint

import cv2

from wepppy.all_your_base import (
    read_tif,
    utm_srid,
    isfloat,
    GeoTransformer,
    wgs84_proj4,
    IS_WINDOWS,
    NCPU
)


# This also assumes that MPICH2 is properly installed on your machine and that TauDEM command line executables exist
# MPICH2.  Obtain from http://www.mcs.anl.gov/research/projects/mpich2/
# Install following instructions at http://hydrology.usu.edu/taudem/taudem5.0/downloads.html.
# It is important that you install this from THE ADMINISTRATOR ACCOUNT.

# TauDEM command line executables.

_thisdir = os.path.dirname(__file__)
_taudem_bin = _join(_thisdir, '../../bin')

_USE_MPI = True

_DEBUG = True


_outlet_template_geojson = """{{
"type": "FeatureCollection",
"name": "Outlet",
"crs": {{ "type": "name", "properties": {{ "name": "urn:ogc:def:crs:EPSG::{epsg}" }} }},
"features": [
{{ "type": "Feature", "properties": {{ "Id": 0 }}, 
   "geometry": {{ "type": "Point", "coordinates": [ {easting}, {northing} ] }} }}
]
}}"""


class WeppTopTranslator:
    """
    Utility class to translate between sub_ids, wepp_ids,
    and topaz_ids, and chn_enums.

    Conventions
        sub_id
            string in the "hill_%i" % top format


        chn_id
            string in the "chn_%i" % top  format

        wepp (a.k.a. wepp_id)
            integer wepp id (consecutive)

        top (a.k.a. topaz_id)
            integer topaz
            right hillslopes end with 3
            left hillslopes end with 2
            center hillslopes end with 1
            channels end with 4

        chn_enum
            integer = wepp - hillslope_n
    """

    def __init__(self, top_sub_ids, top_chn_ids):
        # need the sub_ids as integers sorted in ascending order
        top_sub_ids = sorted(top_sub_ids)

        # need the chn_ids as integers sorted in descending order
        top_chn_ids = sorted(top_chn_ids, reverse=True)

        # now we are going to assign wepp ids and build
        # lookup dictionaries from translating between
        # wepp and topaz ids
        top2wepp = {0: 0}

        i = 1
        for _id in top_sub_ids:
            assert _id not in top2wepp
            top2wepp[_id] = i
            i += 1

        for _id in top_chn_ids:
            assert _id not in top2wepp
            top2wepp[_id] = i
            i += 1

        wepp2top = dict([(v, k) for k, v in top2wepp.items()])

        self.sub_ids = ["hill_%i" % _id for _id in top_sub_ids]
        self.chn_ids = ["chn_%i" % _id for _id in top_chn_ids]
        self._wepp2top = wepp2top
        self._top2wepp = top2wepp
        self.hillslope_n = len(top_sub_ids)
        self.channel_n = len(top_chn_ids)
        self.n = self.hillslope_n + self.channel_n

    def top(self, wepp=None, sub_id=None, chn_id=None, chn_enum=None):
        assert sum([v is not None for v in [wepp, sub_id, chn_id, chn_enum]]) == 1
        _wepp2top = self._wepp2top

        if sub_id is not None:
            return int(sub_id.split('_')[1])

        if chn_id is not None:
            return int(chn_id.split('_')[1])

        if chn_enum is not None:
            wepp = self.wepp(chn_enum=int(chn_enum))

        if wepp is not None:
            return _wepp2top[int(wepp)]

        return None

    def wepp(self, top=None, sub_id=None, chn_id=None, chn_enum=None):
        assert sum([v is not None for v in [top, sub_id, chn_id, chn_enum]]) == 1
        _top2wepp = self._top2wepp
        hillslope_n = self.hillslope_n

        if chn_enum is not None:
            return int(chn_enum) + hillslope_n

        if sub_id is not None:
            top = self.top(sub_id=sub_id)

        if chn_id is not None:
            top = self.top(chn_id=chn_id)

        if top is not None:
            return _top2wepp[int(top)]

        return None

    def chn_enum(self, wepp=None, chn_id=None, top=None):
        assert sum([v is not None for v in [wepp, chn_id, top]]) == 1
        hillslope_n = self.hillslope_n

        if chn_id is not None:
            wepp = self.wepp(chn_id=chn_id)

        if top is not None:
            wepp = self.wepp(top=int(top))

        if wepp == 0:
            return 0

        assert self.is_channel(wepp=wepp), (wepp, top)

        if wepp is not None:
            return wepp - hillslope_n

        return None

    def is_channel(self, top=None, wepp=None):
        assert sum([v is not None for v in [top, wepp]]) == 1

        if top is not None:
            return str(top).endswith('4')
        else:
            return str(self.top(wepp=int(wepp))).endswith('4')

    def has_top(self, top):
        return top in self._top2wepp

    def __iter__(self):
        for sub_id in self.sub_ids:
            yield int(sub_id.split('_')[1])

        for chn_id in self.chn_ids:
            yield int(chn_id.split('_')[1])

    def iter_chn_ids(self):
        for chn_id in self.chn_ids:
            yield chn_id

    def iter_sub_ids(self):
        for sub_id in self.sub_ids:
            yield sub_id

    def iter_wepp_chn_ids(self):
        for chn_id in self.chn_ids:
            yield self.wepp(chn_id=chn_id)

    def iter_wepp_sub_ids(self):
        for sub_id in self.sub_ids:
            yield self.wepp(sub_id=sub_id)


def _cummnorm_distance(distance: List[float]) -> np.array:
    """
    builds and returns cumulative normalized distance array from an array
    of cell-to-cell distances
    """
    assert len(distance) > 0

    if len(distance) == 1:
        assert distance[0] > 0.0
        return np.array([0, 1])

    distance_p = np.cumsum(np.array(distance, np.float64))
    distance_p -= distance_p[0]
    distance_p /= distance_p[-1]
    return distance_p


def get_utm_zone(srs):
    """
    extracts the utm_zone from an osr.SpatialReference object (srs)

    returns the utm_zone as an int, returns None if utm_zone not found
    """

    if not isinstance(srs, osr.SpatialReference):
        raise TypeError('srs is not a osr.SpatialReference instance')

    if srs.IsProjected() != 1:
        return None

    projcs = srs.GetAttrValue('projcs')
    assert 'UTM' in projcs

    datum = None
    if 'NAD83' in projcs:
        datum = 'NAD83'
    elif 'WGS84' in projcs:
        datum = 'WGS84'
    elif 'NAD27' in projcs:
        datum = 'NAD27'

    # should be something like NAD83 / UTM zone 11N...

    if '/' in projcs:
        utm_token = projcs.split('/')[1]
    else:
        utm_token = projcs
    if 'UTM' not in utm_token:
        return None

    # noinspection PyBroadException
    try:
        utm_zone = int(''.join([k for k in utm_token if k in '0123456789']))
    except Exception:
        return None

    if utm_zone < 0 or utm_zone > 60:
        return None

    hemisphere = projcs[-1]
    return datum, utm_zone, hemisphere


def _centroid_px(indx, indy):
    """
    given a sets of x and y indices calulates a central [x,y] index
    """
    return (int(round(float(np.mean(indx)))),
            int(round(float(np.mean(indy)))))


def _compute_direction(head: List, tail: List) -> float:
    a = math.atan2(tail[1] - head[1],
                   head[0] - tail[0]) * (180.0 / math.pi) - 180.0

    if a < 0:
        return a + 360.0

    return a


def _representative_normalized_elevations(x: List[float], dy: List[float]) -> List[float]:
    """
    x should be a normed distance array between 0 and 1
    dy is an array of slopes

    returns normalized elevations (relative to the length of x)
    """
    assert len(x) == len(dy), (x, dy)
    assert x[0] == 0.0
    assert x[-1] == 1.0

    # calculate the positions, assume top of hillslope is 0 y
    y = [0.0]
    for i in range(len(dy) - 1):
        step = x[i+1] - x[i]
        y.append(y[-1] - step * dy[i])

    return y


def _weighted_slope_average(surface_lengths, slopes, lengths, max_points=19):
    """
    calculates weighted slopes based on the flowpaths contained on the hillslope

    eq. 3.3 in Thomas Cochrane's Dissertation
    """
    areas = np.ceil(surface_lengths)

    # determine longest flowpath
    i = int(np.argmax(lengths))
    longest = float(lengths[i])

    # determine number of points to define slope
    num_points = len(lengths)
    if num_points > max_points:
        num_points = max_points

    if num_points == 1:
        slope = float(slopes[i])
        return [slope, slope], [0.0, 1.0]

    # determine weights for each flowpath
    kps = np.array([L * a for L, a in zip(lengths, areas)])

    # build an array with equally spaced points to interpolate on
    distance_p = np.linspace(0, longest, num_points)

    # this will hold the weighted slope estimates
    eps = []

    # we will weight the slope at each distance away from the channel
    for d_p in distance_p:
        num = 0  # to hold numerator value
        kpsum = 0  # to hold k_p sum

        for slp, rcd, kp in zip(slopes, surface_lengths, kps):

            # we only want to interpolate where the slope is defined
            if d_p - 1e-6 > rcd:
                continue

            num += slp * kp
            kpsum += kp

        # store the weighted slope estimate
        eps.append(num / kpsum)

    # normalize distance_p array
    distance_p /= longest

    # reverse weighted slopes and return
    w_slopes = np.array(eps[::-1])

    return w_slopes.flatten().tolist(), distance_p.tolist()


def _rect_to_polar(d):
    point = d['point']
    origin = d['origin']
    refvec = d['refvec']

    # Vector between point and the origin: v = p - o
    vector = [point[0] - origin[0], point[1] - origin[1]]

    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])

    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi

    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]
    angle = math.atan2(diffprod, dotprod)

    angle %= 2*math.pi

    return angle


class Node:
    def __init__(self, tau_id, network):

        self.data = tau_id

        d = network[tau_id]

        self.top = top = d['top']
        self.bottom = bottom = d['bottom']

        links = d['links']

        if len(links) == 2:
            refvec = np.array(bottom, dtype=float) - np.array(top, dtype=float)
            links = sorted([dict(tau_id=_id, point=network[_id]['top'], origin=top, refvec=refvec)
                            for _id in links], key=lambda _d: _rect_to_polar(_d))
            links = [_d['tau_id'] for _d in links]

        if len(links) > 0:
            self.left = Node(links[0], network)
        else:
            self.left = None

        if len(links) > 1:
            self.right = Node(links[1], network)
        else:
            self.right = None


class TauDEMRunner:
    """
    Object oriented abstraction for running TauDEM

    For more infomation on taudem see the manual available here:
        https://hydrology.usu.edu/taudem/taudem5/documentation.html
    """
    def __init__(self, wd, dem, vector_ext='geojson'):
        """
        provide a path to a directory to store the taudem files a
        path to a dem
        """

        # verify the dem exists
        if not _exists(wd):
            raise Exception('working directory "%s" does not exist' % wd)

        self.wd = wd

        # verify the dem exists
        if not _exists(dem):
            raise Exception('file "%s" does not exist' % dem)

        self._dem_ext = _split(dem)[-1].split('.')[-1]
        shutil.copyfile(dem, self._z)

        self._vector_ext = vector_ext

        self.user_outlet = None
        self.outlet = None
        self._scratch = {}
        self._parse_dem()

    def _parse_dem(self):
        """
        reads metadata from the dem to get the projection, transform, bounds, resolution, and size
        """
        dem = self._z

        # open the dataset
        ds = gdal.Open(dem)

        # read and verify the num_cols and num_rows
        num_cols = ds.RasterXSize
        num_rows = ds.RasterYSize

        if num_cols <= 0 or num_rows <= 0:
            raise Exception('input is empty')

        # read and verify the _transform
        _transform = ds.GetGeoTransform()

        if abs(_transform[1]) != abs(_transform[5]):
            raise Exception('input cells are not square')

        cellsize = abs(_transform[1])
        ul_x = int(round(_transform[0]))
        ul_y = int(round(_transform[3]))

        lr_x = ul_x + cellsize * num_cols
        lr_y = ul_y - cellsize * num_rows

        ll_x = int(ul_x)
        ll_y = int(lr_y)

        # read the projection and verify dataset is in utm
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjectionRef())

        datum, utm_zone, hemisphere = get_utm_zone(srs)
        if utm_zone is None:
            raise Exception('input is not in utm')

        # get band
        band = ds.GetRasterBand(1)

        # get band dtype
        dtype = gdal.GetDataTypeName(band.DataType)

        if 'float' not in dtype.lower():
            raise Exception('dem dtype does not contain float data')

        # extract min and max elevation
        stats = band.GetStatistics(True, True)
        minimum_elevation = stats[0]
        maximum_elevation = stats[1]

        # store the relevant variables to the class
        self.transform = _transform
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.cellsize = cellsize
        self.ul_x = ul_x
        self.ul_y = ul_y
        self.lr_x = lr_x
        self.lr_y = lr_y
        self.ll_x = ll_x
        self.ll_y = ll_y
        self.datum = datum
        self.hemisphere = hemisphere
        self.epsg = utm_srid(utm_zone, datum, hemisphere)
        self.utm_zone = utm_zone
        self.srs_proj4 = srs.ExportToProj4()
        srs.MorphToESRI()
        self.srs_wkt = srs.ExportToWkt()
        self.minimum_elevation = minimum_elevation
        self.maximum_elevation = maximum_elevation


        del ds

    def data_fetcher(self, band, dtype=None):
        if dtype is None:
            dtype = np.int16

        if band not in self._scratch:
            _band = getattr(self, '_' + band)
            self._scratch[band], _, _ = read_tif(_band, dtype=dtype)

        return self._scratch[band]

    def get_elevation(self, easting, northing):
        z_data = self.data_fetcher('z', dtype=np.float64)
        x, y = self.utm_to_px(easting, northing)

        return z_data[x, y]

    def utm_to_px(self, easting, northing):
        """
        return the utm coords from pixel coords
        """

        # unpack variables for instance
        cellsize, num_cols, num_rows = self.cellsize, self.num_cols, self.num_rows
        ul_x, ul_y, lr_x, lr_y = self.ul_x, self.ul_y, self.lr_x, self.lr_y

        if isfloat(easting):
            x = int(round((easting - ul_x) / cellsize))
            y = int(round((northing - ul_y) / -cellsize))

            assert x >= 0 and x < num_rows, x
            assert y >= 0 and x < num_cols, y
        else:
            x = np.array(np.round((np.array(easting) - ul_x) / cellsize), dtype=np.int)
            y = np.array(np.round((np.array(northing) - ul_y) / -cellsize), dtype=np.int)

        return x, y

    def lnglat_to_px(self, long, lat):
        """
        return the x,y pixel coords of long, lat
        """

        # unpack variables for instance
        cellsize, num_cols, num_rows = self.cellsize, self.num_cols, self.num_rows
        ul_x, ul_y, lr_x, lr_y = self.ul_x, self.ul_y, self.lr_x, self.lr_y

        # find easting and northing
        x, y, _, _ = utm.from_latlon(lat, long, self.utm_zone)

        # assert this makes sense with the stored extent
        assert round(x) >= round(ul_x), (x, ul_x)
        assert round(x) <= round(lr_x), (x, lr_x)
        assert round(y) >= round(lr_y), (y, lr_y)
        assert round(y) <= round(y), (y, ul_y)

        # determine pixel coords
        _x = int(round((x - ul_x) / cellsize))
        _y = int(round((ul_y - y) / cellsize))

        # sanity check on the coords
        assert 0 <= _x < num_cols, str(x)
        assert 0 <= _y < num_rows, str(y)

        return _x, _y

    def px_to_utm(self, x, y):
        """
        return the utm coords from pixel coords
        """

        # unpack variables for instance
        cellsize, num_cols, num_rows = self.cellsize, self.num_cols, self.num_rows
        ul_x, ul_y, lr_x, lr_y = self.ul_x, self.ul_y, self.lr_x, self.lr_y

        assert 0 <= x < num_cols
        assert 0 <= y < num_rows

        easting = ul_x + cellsize * x
        northing = ul_y - cellsize * y

        return easting, northing

    def lnglat_to_utm(self, long, lat):
        """
        return the utm coords from lnglat coords
        """
        wgs2proj_transformer = GeoTransformer(src_proj4=wgs84_proj4, dst_proj4=self.srs_proj4)
        return wgs2proj_transformer.transform(long, lat)

    def px_to_lnglat(self, x, y):
        """
        return the long/lat (WGS84) coords from pixel coords
        """
        easting, northing = self.px_to_utm(x, y)
        proj2wgs_transformer = GeoTransformer(src_proj4=self.srs_proj4, dst_proj4=wgs84_proj4)
        return proj2wgs_transformer.transform(easting, northing)

    # dem
    @property
    def _z(self):
        return _join(self.wd, 'dem.%s' % self._dem_ext)

    @property
    def _z_args(self):
        return ['-z', self._z]

    # fel
    @property
    def _fel(self):
        return _join(self.wd, 'fel.tif')

    @property
    def _fel_args(self):
        return ['-fel', self._fel]

    # point
    @property
    def _fd8(self):
        return _join(self.wd, 'd8_flow.tif')

    @property
    def _fd8_args(self):
        return ['-p', self._fd8]

    _p_args = _fd8_args

    # slope d8
    @property
    def _sd8(self):
        return _join(self.wd, 'd8_slope.tif')

    @property
    def _sd8_args(self):
        return ['-sd8', self._sd8]

    # area d8
    @property
    def _ad8(self):
        return _join(self.wd, 'd8_area.tif')

    @property
    def _ad8_args(self):
        return ['-ad8', self._ad8]

    # stream raster
    @property
    def _src(self):
        return _join(self.wd, 'src.tif')

    @property
    def _src_args(self):
        return ['-src', self._src]

    # pk stream reaster
    @property
    def _pksrc(self):
        return _join(self.wd, 'pksrc.tif')

    @property
    def _pksrc_args(self):
        return ['-src', self._pksrc]

    # net
    @property
    def _net(self):
        return _join(self.wd, 'net.%s' % self._vector_ext)

    @property
    def _net_args(self):
        return ['-net', self._net]

    # user outlet
    @property
    def _uo(self):
        return _join(self.wd, 'user_outlet.%s' % self._vector_ext)

    @property
    def _uo_args(self):
        return ['-o', self._uo]

    # outlet
    @property
    def _o(self):
        return _join(self.wd, 'outlet.%s' % self._vector_ext)

    @property
    def _o_args(self):
        return ['-o', self._o]

    # stream source
    @property
    def _ss(self):
        return _join(self.wd, 'ss.tif')

    @property
    def _ss_args(self):
        return ['-ss', self._ss]

    # ssa
    @property
    def _ssa(self):
        return _join(self.wd, 'ssa.tif')

    @property
    def _ssa_args(self):
        return ['-ssa', self._ssa]

    # drop
    @property
    def _drp(self):
        return _join(self.wd, 'drp.txt')

    @property
    def _drp_args(self):
        return ['-drp', self._drp]

    # tree
    @property
    def _tree(self):
        return _join(self.wd, 'tree.tsv')

    @property
    def _tree_args(self):
        return ['-tree', self._tree]

    # coord
    @property
    def _coord(self):
        return _join(self.wd, 'coord.tsv')

    @property
    def _coord_args(self):
        return ['-coord', self._coord]

    # order
    @property
    def _ord(self):
        return _join(self.wd, 'order.tif')

    @property
    def _ord_args(self):
        return ['-ord', self._ord]

    # watershed
    @property
    def _w(self):
        return _join(self.wd, 'watershed.tif')

    @property
    def _w_args(self):
        return ['-w', self._w]

    # gord
    @property
    def _gord(self):
        return _join(self.wd, 'gord.tif')

    @property
    def _gord_args(self):
        return ['-gord', self._gord]

    # plen
    @property
    def _plen(self):
        return _join(self.wd, 'plen.tif')

    @property
    def _plen_args(self):
        return ['-plen', self._plen]

    # tlen
    @property
    def _tlen(self):
        return _join(self.wd, 'tlen.tif')

    @property
    def _tlen_args(self):
        return ['-tlen', self._tlen]

    # subwta
    @property
    def _subwta(self):
        return _join(self.wd, 'subwta.tif')

    # dinf angle
    @property
    def _dinf_angle(self):
        return _join(self.wd, 'dinf_angle.tif')

    @property
    def _dinf_angle_args(self):
        return ['-ang', self._dinf_angle]

    # dinf slope
    @property
    def _dinf_slope(self):
        return _join(self.wd, 'dinf_slope.tif')

    @property
    def _dinf_slope_args(self):
        return ['-slp', self._dinf_slope]

    # dinf contributing area
    @property
    def _dinf_sca(self):
        return _join(self.wd, 'dinf_sca.tif')

    @property
    def _dinf_sca_args(self):
        return ['-sca', self._dinf_sca]

    # dinf distance down output
    @property
    def _dinf_dd_horizontal(self):
        return _join(self.wd, 'dinf_dd_horizontal.tif')

    @property
    def _dinf_dd_vertical(self):
        return _join(self.wd, 'dinf_dd_vertical.tif')

    @property
    def _dinf_dd_surface(self):
        return _join(self.wd, 'dinf_dd_surface.tif')

    # subprocess methods

    @property
    def _mpi_args(self):
        global _USE_MPI

        if _USE_MPI:
            return ['mpiexec', '-n', NCPU]
        else:
            return []

    def _sys_call(self, cmd, verbose=True, intent_in=None, intent_out=None):
        # verify inputs exist
        if intent_in is not None:
            for product in intent_in:
                assert _exists(product), product

        # delete outputs if they exist
        if intent_out is not None:
            for product in intent_out:
                if _exists(product):
                    os.remove(product)

        cmd = [str(v) for v in cmd]
        caller = inspect.stack()[1].function
        log = _join(self.wd, caller + '.log')
        _log = open(log, 'w')

        if verbose:
            print(caller, cmd)

        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=_log, stderr=_log)
        p.wait()
        _log.close()

        if intent_out is None:
            return

        for product in intent_out:
            if not _exists(product):
                raise Exception('{} Failed: {} does not exist. See {}'.format(caller, product, log))

            if product.endswith('.tif'):
                p = subprocess.Popen(['gdalinfo', product, '-stats'], shell=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     close_fds=True)
                stdout = p.stdout.read().decode('utf-8')

                if 'no valid pixels found in sampling' in stdout:
                    raise Exception('{} Failed: {} does not contain any valid pixels.'.format(caller, product))

        if not _DEBUG:
            os.remove(log)

    @property
    def __pitremove(self):
        return self._mpi_args + [_join(_taudem_bin, 'pitremove')]

    @property
    def __d8flowdir(self):
        return self._mpi_args + [_join(_taudem_bin, 'd8flowdir')]

    @property
    def __aread8(self):
        return self._mpi_args + [_join(_taudem_bin, 'aread8')]

    @property
    def __gridnet(self):
        return self._mpi_args + [_join(_taudem_bin, 'gridnet')]

    @property
    def __threshold(self):
        return self._mpi_args + [_join(_taudem_bin, 'threshold')]

    @property
    def __moveoutletstostrm(self):
        return self._mpi_args + [_join(_taudem_bin, 'moveoutletstostrm')]

    @property
    def __peukerdouglas(self):
        return self._mpi_args + [_join(_taudem_bin, 'peukerdouglas')]

    @property
    def __dropanalysis(self):
        return self._mpi_args + [_join(_taudem_bin, 'dropanalysis')]

    @property
    def __streamnet(self):
        return self._mpi_args + [_join(_taudem_bin, 'streamnet')]

    @property
    def __gagewatershed(self):
        return self._mpi_args + [_join(_taudem_bin, 'gagewatershed')]

    @property
    def __dinfflowdir(self):
        return self._mpi_args + [_join(_taudem_bin, 'dinfflowdir')]

    @property
    def __areadinf(self):
        return self._mpi_args + [_join(_taudem_bin, 'areadinf')]

    @property
    def __dinfdistdown(self):
        return self._mpi_args + [_join(_taudem_bin, 'dinfdistdown')]

    # TauDEM wrapper methods

    def run_pitremove(self):
        """
        This function takes as input an elevation data grid and outputs a hydrologically correct elevation grid file
        with pits filled, using the flooding algorithm.

        in: dem
        out: fel
        """
        self._sys_call(self.__pitremove + self._z_args + self._fel_args,
                       intent_in=(self._z,),
                       intent_out=(self._fel,))

    def run_d8flowdir(self):
        """
        This function takes as input the hydrologically correct elevation grid and outputs D8 flow direction and slope
        for each grid cell. In flat areas flow directions are assigned away from higher ground and towards lower ground
        using the method of Garbrecht and Martz (Garbrecht and Martz, 1997).

        in: fel
        out: point, slope_d8
        """
        self._sys_call(self.__d8flowdir + self._fel_args + self._p_args + self._sd8_args,
                       intent_in=(self._fel,),
                       intent_out=(self._sd8, self._fd8))

    def run_aread8(self, no_edge_contamination_checking=False):
        """
        This function takes as input a D8 flow directions file and outputs the contributing area. The result is the
        number of grid cells draining through each grid cell. The optional command line argument for the outlet
        shapefile results in only the area contributing to outlet points in the shapefile being calculated. The optional
        weight grid input results in the output being the accumulation (sum) of the weights from upstream grid cells
        draining through each grid cell. By default the program checks for edge contamination. The edge contamination
        checking may be overridden with the optional command line.

        in: point
        out: area_d8
        """
        self._sys_call(self.__aread8 + self._p_args + self._ad8_args + ([], ['-nc'])[no_edge_contamination_checking],
                       intent_in=(self._fd8,),
                       intent_out=(self._ad8,))

    def run_gridnet(self):
        """
        in: p
        out: gord, plen, tlen
        """
        self._sys_call(self.__gridnet + self._p_args + self._gord_args + self._plen_args + self._tlen_args,
                       intent_in=(self._fd8,),
                       intent_out=(self._gord, self._plen, self._tlen))

    def _run_threshold(self, ssa, src, threshold=1000):
        """
        This function operates on any grid and outputs an indicator (1,0) grid of grid cells that have values >= the
        input threshold. The standard use is to threshold an accumulated source area grid to determine a stream raster.
        There is an option to include a mask input to replicate the functionality for using the sca file as an edge
        contamination mask. The threshold logic should be src = ((ssa >= thresh) & (mask >=0)) ? 1:0

        in: ssa
        out: src
        """
        self._sys_call(self.__threshold + ['-ssa', ssa, '-src', src, '-thresh', threshold],
                       intent_in=(ssa,),
                       intent_out=(src,))

    def run_src_threshold(self, threshold=1000):
        self._run_threshold(ssa=self._ad8, src=self._src, threshold=threshold)

    def _make_outlet(self, long=None, lat=None, dst=None, easting=None, northing=None):
        assert dst is not None

        if long is not None and lat is not None:
            easting, northing = self.lnglat_to_utm(long=long, lat=lat)

        assert isfloat(easting), easting
        assert isfloat(northing), northing

        with open(dst, 'w') as fp:
            fp.write(_outlet_template_geojson.format(epsg=self.epsg, easting=easting, northing=northing))

        assert _exists(dst), dst
        return dst

    def run_moveoutletstostrm(self, long, lat):
        """
        This function finds the closest channel location to the requested location

        :param long: requested longitude
        :param lat: requested latitude
        """
        self.user_outlet = long, lat
        self._make_outlet(long=long, lat=lat, dst=self._uo)
        self._sys_call(self.__moveoutletstostrm + self._p_args + self._src_args + ['-o', self._uo] + ['-om', self._o],
                       intent_in=(self._fd8, self._src, self._uo),
                       intent_out=(self._o,))

        with open(self._o) as fp:
            js = json.load(fp)
            if js['features'][0]['properties']['Dist_moved'] == -1:
                raise ValueError('Outlet location could not be processed')

            o_e, o_n = js['features'][0]['geometry']['coordinates']

        proj2wgs_transformer = GeoTransformer(src_proj4=self.srs_proj4, dst_proj4=wgs84_proj4)
        self.outlet = proj2wgs_transformer.transform(x=o_e, y=o_n)

    def run_peukerdouglas(self, center_weight=0.4, side_weight=0.1, diagonal_weight=0.05):
        """
        This function operates on an elevation grid and outputs an indicator (1,0) grid of upward curved grid cells
        according to the Peuker and Douglas algorithm. This is to be based on code in tardemlib.cpp/source.

        in: fel
        out: ss
        """
        self._sys_call(self.__peukerdouglas + self._fel_args + self._ss_args +
                       ['-par', center_weight, side_weight, diagonal_weight],
                       intent_in=(self._fel,),
                       intent_out=(self._ss,))

    @property
    def drop_analysis_threshold(self):
        """
        Reads the drop table and extracts the optimal value

        :return: optimimum threshold value from drop table
        """
        with open(self._drp) as fp:
            lines = fp.readlines()

        last = lines[-1]

        assert 'Optimum Threshold Value:' in last, '\n'.join(lines)
        return float(last.replace('Optimum Threshold Value:', '').strip())

    def run_peukerdouglas_stream_delineation(self, threshmin=5, threshmax=500, nthresh=10, steptype=0, threshold=None):
        """

        :param threshmin:
        :param threshmax:
        :param nthresh:
        :param steptype:
        :param threshold:

        in: p, o, ss
        out:
        """
        self._sys_call(self.__aread8 + self._p_args + self._o_args + ['-ad8', self._ssa] + ['-wg', self._ss],
                       intent_in=(self._fd8, self._o, self._ss),
                       intent_out=(self._ssa,))

        self._sys_call(self.__dropanalysis + self._p_args + self._fel_args +
                       self._ad8_args + self._ssa_args + self._drp_args +
                       self._o_args + ['-par', threshmin, threshmax, nthresh, steptype],
                       intent_in=(self._fd8, self._fel, self._ad8, self._o, self._ssa),
                       intent_out=(self._drp,))

        if threshold is None:
            threshold = self.drop_analysis_threshold

        self._run_threshold(self._ssa, self._pksrc, threshold=threshold)

    def run_streamnet(self, single_watershed=False, use_topaz_ids=True):
        """
        in: fel, p, ad8, pksrc, o
        out: w, ord, tree, net, coors
        """
        self._sys_call(self.__streamnet + self._fel_args + self._p_args + self._ad8_args +
                       self._pksrc_args + self._o_args + self._ord_args + self._tree_args + self._net_args +
                       self._coord_args + self._w_args + ([], ['-sw'])[single_watershed],
                       intent_in=(self._fel, self._fd8, self._ad8, self._pksrc, self._o),
                       intent_out=(self._w, self._ord, self._tree, self._net, self._coord))

        if not use_topaz_ids:
            return

        translator = self.tau2topaz_translator_factory()

        with open(self._net) as fp:
            js = json.load(fp)

        for i, feature in enumerate(js['features']):
            topaz_id = translator[feature['properties']['WSNO']]
            js['features'][i]['properties']['TopazID'] = int(str(topaz_id) + '4')

        with open(self._net, 'w') as fp:
            json.dump(js, fp)

    def _run_gagewatershed(self, **kwargs):
        """
        in: p
        out: gw
        """
        long = kwargs.get('long', None)
        lat = kwargs.get('lat', None)
        easting = kwargs.get('easting', None)
        northing = kwargs.get('northing', None)
        dst = kwargs.get('dst', None)

        point = self._make_outlet(long=long, lat=lat, easting=easting, northing=northing, dst=dst[:-4] + '.geojson')
        self._sys_call(self.__gagewatershed + self._p_args + ['-o', point] + ['-gw', dst],
                       intent_in=(point, self._fd8),
                       intent_out=(dst,))

    def run_dinfflowdir(self):
        """
        in: fel
        out: dinf_angle, dinf_slope
        """
        self._sys_call(self.__dinfflowdir + self._fel_args + self._dinf_angle_args + self._dinf_slope_args,
                       intent_in=(self._fel,),
                       intent_out=(self._dinf_angle, self._dinf_slope))

    def run_areadinf(self):
        """
        in: dinf_angle
        out: dinf_sca
        """
        self._sys_call(self.__areadinf + self._dinf_angle_args + self._o_args + self._dinf_sca_args,
                       intent_in=(self._o, self._dinf_angle),
                       intent_out=(self._dinf_sca,))

    def run_dinfdistdown(self, no_edge_contamination_checking=False):
        """

        in: dinf_angle, fel,
        out: dinf_dd_horizontal, dinf_dd_vertical, dinf_dd_surface
        """
        # method_statistic:
        #     ave = average of flowpath, min = minimum length of flowpath, max = maximum length of flowpath
        method_statistic = 'ave'

        # method_type:
        #     h = horizontal, v = vertical, p = Pythagoras, s = surface

        for method_type in ['horizontal', 'vertical', 'surface']:
            dst = _join(self.wd, 'dinf_dd_%s.tif' % method_type)

            self._sys_call(self.__dinfdistdown + self._dinf_angle_args + self._fel_args + self._pksrc_args +
                           ['-dd', dst] + ['-m', method_statistic, method_type[0]] +
                           ([], ['-nc'])[no_edge_contamination_checking],
                           intent_in=(self._dinf_angle, self._fel, self._pksrc),
                           intent_out=(dst,))

    def delineate_subcatchments(self, use_topaz_ids=True):
        """
        in: pksrc, net,
        out: subwta
        :return:
        """

        w_data = self.data_fetcher('w', dtype=np.int32)
        _src_data = self.data_fetcher('pksrc', dtype=np.int32)
        src_data = np.zeros(_src_data.shape, dtype=np.int32)
        src_data[np.where(_src_data == 1)] = 1

        subwta = np.zeros(w_data.shape, dtype=np.uint16)

        with open(self._net) as fp:
            js = json.load(fp)

        for _pass in range(2):
            for feature in js['features']:
                topaz_id = int(str(feature['properties']['TopazID'])[:-1])
                catchment_id = feature['properties']['WSNO']
                coords = feature['geometry']['coordinates']
                uslinkn01 = feature['properties']['USLINKNO1']
                uslinkn02 = feature['properties']['USLINKNO2']
                end_node = uslinkn01 == -1 and uslinkn02 == -1

                if end_node:
                    if _pass == 1:
                        continue  # this has already been processed

                else:
                    if _pass == 0:
                        continue  # don't process non end nodes on the first pass

                top = coords[-1]
                bottom = coords[0]

                top_px = self.utm_to_px(top[0], top[1])
                bottom_px = self.utm_to_px(bottom[0], bottom[1])

                # need a mask for the side subcatchments
                catchment_data = np.zeros(w_data.shape, dtype=np.int32)
                catchment_data[np.where(w_data == catchment_id)] = 1

                if end_node:
                    gw = _join(self.wd, 'wsno_%05i.tif' % catchment_id)
                    self._run_gagewatershed(easting=top[0], northing=top[1], dst=gw)

                    gw_data, _, _ = read_tif(gw, dtype=np.int16)  # gage watershed cells are 0 in the drainage area
                    gw_data += 1
                    gw_data = np.clip(gw_data, 0, 1)

                    # don't allow gw to extend beyond catchment
                    gw_data *= catchment_data

                    # identify top subcatchment cells
                    gw_indx = np.where(gw_data == 1)

                    # copy the top subcatchment to the subwta raster
                    if use_topaz_ids:
                        subwta[gw_indx] = int(str(topaz_id) + '1')
                    else:
                        subwta[gw_indx] = int(str(catchment_id) + '1')

                    if not _DEBUG:
                        os.remove(gw)
                        os.remove(gw[:-4] + '.geojson')

                # remove end subcatchments from the catchment mask
                catchment_data[np.where(subwta != 0)] = 0

                # remove channels from catchment mask
                catchment_data -= src_data
                catchment_data = np.clip(catchment_data, a_min=0, a_max=1)
                indx, indy = np.where(catchment_data == 1)

                # the whole catchment drains through the top of the channel
                if len(indx) == 0:
                    continue

                if _DEBUG:
                    driver = gdal.GetDriverByName('GTiff')
                    dst_ds = driver.Create(_join(self.wd, 'catchment_for_label_%05i.tif' % catchment_id),
                                           xsize=subwta.shape[0], ysize=subwta.shape[1],
                                           bands=1, eType=gdal.GDT_Int32,
                                           options=['COMPRESS=LZW', 'PREDICTOR=2'])
                    dst_ds.SetGeoTransform(self.transform)
                    dst_ds.SetProjection(self.srs_wkt)
                    band = dst_ds.GetRasterBand(1)
                    band.WriteArray(catchment_data.T)
                    dst_ds = None

                # we are going to crop the catchment for scipy.ndimage.label. It is really slow otherwise
                # to do this we identify the bounds and then add a pad
                pad = 1
                x0, xend = np.min(indx), np.max(indx)
                if x0 >= pad:
                    x0 -= pad
                if xend < self.num_cols - pad:
                    xend += pad

                y0, yend = np.min(indy), np.max(indy)

                if y0 >= pad:
                    y0 -= pad
                if yend < self.num_rows - pad:
                    yend += pad

                # crop to just the side channel catchments
                _catchment_data = catchment_data[x0:xend, y0:yend]

                # use scipy.ndimage.label to identify side subcatchments
                subcatchment_data, n_labels = label(_catchment_data)

                # isolated pixels in the channel can get misidentified as subcatchments
                # this gets rid of those
                subcatchment_data -= src_data[x0:xend, y0:yend]

                # we only want the two largest subcatchments. These should be the side subcatchments
                # so we need to identify which are the largest
                sub_d = []
                for i in range(n_labels):
                    s_indx, s_indy = np.where(subcatchment_data == i + 1)
                    sub_d.append(dict(rank=len(s_indx), s_indx=s_indx, s_indy=s_indy,
                                      point=(x0 + np.mean(s_indx), y0 + np.mean(s_indy)),
                                      origin=(float(bottom_px[0]), float(bottom_px[1])),
                                      refvec=np.array(top_px, dtype=float) - np.array(bottom_px, dtype=float)
                                      )
                                 )

                # sort clockwise
                sub_d = sorted(sub_d, key=lambda _d: _d['rank'], reverse=True)

                if len(sub_d) > 2:
                    sub_d = sub_d[:2]

                sub_d = sorted(sub_d, key=lambda _d: _rect_to_polar(_d))

                # assert len(sub_d) == 2

                k = 2
                for d in sub_d:
                    if use_topaz_ids:
                        subwta[x0:xend, y0:yend][d['s_indx'], d['s_indy']] = int(str(topaz_id) + str(k))
                    else:
                        subwta[x0:xend, y0:yend][d['s_indx'], d['s_indy']] = int(str(catchment_id) + str(k))
                    k += 1

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(self._subwta, xsize=subwta.shape[0], ysize=subwta.shape[1],
                               bands=1, eType=gdal.GDT_UInt16, options=['COMPRESS=LZW', 'PREDICTOR=2'])
        dst_ds.SetGeoTransform(self.transform)
        dst_ds.SetProjection(self.srs_wkt)
        band = dst_ds.GetRasterBand(1)
        band.WriteArray(subwta.T)
        band.SetNoDataValue(0)
        dst_ds = None

    @property
    def cellsize2(self):
        return self.cellsize ** 2

    @property
    def network(self):
        with open(self._net) as fp:
            js = json.load(fp)

        network = {}

        for feature in js['features']:
            tau_id = feature['properties']['WSNO']
            uslinkn01 = feature['properties']['USLINKNO1']
            uslinkn02 = feature['properties']['USLINKNO2']
            enz_coords = feature['geometry']['coordinates']
            bottom = enz_coords[0][0], enz_coords[0][1]
            top = enz_coords[-1][0], enz_coords[-1][1]

            links = [v for v in [uslinkn01, uslinkn02] if v != -1]
            network[tau_id] = dict(links=links, top=top, bottom=bottom)

        return network

    @property
    def topaz_network(self):
        tau2top = self.tau2topaz_translator_factory()

        network = self.network

        top_network = {}
        for tau_id, d in network.items():
            topaz_id = int(str(tau2top[tau_id]) + '4')
            links = [int(str(tau2top[_tau_id]) + '4') for _tau_id in d['links']]
            top_network[topaz_id] = links

        return top_network

    @property
    def outlet_tau_id(self):
        with open(self._net) as fp:
            js = json.load(fp)

        for feature in js['features']:
            tau_id = feature['properties']['WSNO']
            dslinkn0 = feature['properties']['DSLINKNO']

            if dslinkn0 == -1:
                return tau_id

    def tau2topaz_translator_factory(self):
        tree = Node(self.outlet_tau_id, self.network)

        def preorder_traverse(node):
            res = []
            if node:
                res.append(node.data)
                res.extend(preorder_traverse(node.left))
                res.extend(preorder_traverse(node.right))

            return res

        tau_ids = preorder_traverse(tree)

        if _DEBUG:
            print('network', tau_ids)

        d = {tau_id: i+2 for i, tau_id in enumerate(tau_ids)}

        return d

    def topaz2tau_translator_factory(self):
        d = self.tau2topaz_translator_factory()

        return {v: k for k, v in d.items()}

    def abstract_channels(self, use_topaz_ids=True):
        cellsize = self.cellsize
        cellsize2 = self.cellsize2

        slopes = self.data_fetcher('dinf_slope', dtype=np.float)

        with open(self._net) as fp:
            js = json.load(fp)

        chn_d = {}

        for feature in js['features']:
            topaz_id = int(str(feature['properties']['TopazID'])[:-1])
            catchment_id = feature['properties']['WSNO']
            uslinkn01 = feature['properties']['USLINKNO1']
            uslinkn02 = feature['properties']['USLINKNO2']
            dslinkn0 = feature['properties']['DSLINKNO']

            if use_topaz_ids:
                chn_id = int(str(topaz_id) + '4')
            else:
                chn_id = int(str(catchment_id) + '4')

            enz_coords = feature['geometry']['coordinates']  # listed bottom to top

            # need to identify unique pixels
            px_last, py_last = None, None
            indx, indy = [], []
            for e, n, z in enz_coords:
                px, py = self.utm_to_px(e, n)
                if px != px_last or py != py_last:
                    assert px >= 0 and px < slopes.shape[0], ((px, py), (e, n), slopes.shape)
                    assert py >= 0 and py < slopes.shape[1], ((px, py), (e, n), slopes.shape)

                    indx.append(px)
                    indy.append(py)
                    px_last, py_last = px, py

            # the pixels are listed bottom to top we want them top to bottom as if we walked downt the flowpath
            indx = indx[::-1]
            indy = indy[::-1]

            flowpath = np.array([indx, indy]).T
            _distance = flowpath[:-1, :] - flowpath[1:, :]
            distance = np.sqrt(np.power(_distance[:, 0], 2.0) +
                               np.power(_distance[:, 1], 2.0))

            slope = np.array([slopes[px, py] for px, py in zip(indx[:-1], indy[:-1])])

            assert distance.shape == slope.shape, (distance.shape, slope.shape)

            # need normalized distance_p to define slope
            distance_p = _cummnorm_distance(distance)
            if len(slope) == 1:
                slope = np.array([float(slope), float(slope)])

            # calculate the length from the distance array
            length = float(np.sum(distance) * cellsize)
            width = float(cellsize)
            # aspect = float(self._determine_aspect(indx, indy))
            isoutlet = dslinkn0 == -1

            head = [v * cellsize for v in flowpath[-1]]
            head = [float(v) for v in head]
            tail = [v * cellsize for v in flowpath[0]]
            tail = [float(v) for v in tail]

            direction = _compute_direction(head, tail)

            c_px, c_py = _centroid_px(indx, indy)
            centroid_lnglat = self.px_to_lnglat(c_px, c_py)

            elevs = _representative_normalized_elevations(distance_p, slope)
            slope_scalar = float(abs(elevs[-1]))

            chn_d[str(chn_id)] = dict(chn_id=int(chn_id),
                                       length=float(length),
                                       width=float(width),
                                       slopes=list(slope),
                                       direction=direction,
                                       distance_p=list(distance_p),
                                       centroid_lnglat=[float(v) for v in centroid_lnglat],
                                       elevs=list(elevs),
                                       slope_scalar=float(slope_scalar)
                                       )

        with open(_join(self.wd, 'channels.json'), 'w') as fp:
            json.dump(chn_d, fp, indent=2, sort_keys=True)

    @property
    def topaz_sub_ids(self):
        subwta = self.data_fetcher('subwta', dtype=np.uint16)
        sub_ids = sorted(list(set(subwta.flatten())))
        sub_ids.remove(0)

        return sub_ids

    @property
    def topaz_chn_ids(self):
        with open(self._net) as fp:
            js = json.load(fp)

        chn_ids = []
        for feature in js['features']:
            chn_ids.append(feature['properties']['TopazID'])

        return chn_ids

    @property
    def translator(self):
        return WeppTopTranslator(top_sub_ids=self.topaz_sub_ids, top_chn_ids=self.topaz_chn_ids)

    def abstract_subcatchments(self):
        """
        in: dinf_dd_horizontal, dinf_dd_vertical, dinf_dd_surface, dinf_slope, subwta
        :return:
        """
        cellsize = self.cellsize
        cellsize2 = self.cellsize2
        sub_ids = self.topaz_sub_ids

        assert _exists(self._dinf_dd_horizontal), self._dinf_dd_horizontal
        assert _exists(self._dinf_dd_vertical), self._dinf_dd_vertical
        assert _exists(self._dinf_dd_surface), self._dinf_dd_surface
        assert _exists(self._dinf_slope), self._dinf_slope
        assert _exists(self._subwta), self._subwta

        subwta = self.data_fetcher('subwta', dtype=np.uint16)

        lengths = self.data_fetcher('dinf_dd_horizontal', dtype=np.float)
        verticals = self.data_fetcher('dinf_dd_vertical', dtype=np.float)
        surface_lengths = self.data_fetcher('dinf_dd_surface', dtype=np.float)
        slopes = self.data_fetcher('dinf_slope', dtype=np.float)

        subs_d = {}

        for sub_id in sub_ids:
            # identify indicies of sub_id
            raw_indx, raw_indy = np.where(subwta == sub_id)
            area = float(len(raw_indx)) * cellsize2

            # qc for dinf statistics
            indx = []
            indy = []
            for x, y in zip(raw_indx, raw_indy):
                if lengths[x, y] > 0:
                    indx.append(x)
                    indy.append(y)

            # extract flowpath statistics
            fp_lengths = lengths[(indx, indy)]
            fp_verticals = verticals[(indx, indy)]
            fp_surface_lengths = surface_lengths[(indx, indy)]
            fp_slopes = slopes[(indx, indy)]

            # determine representative length and width
            # Cochrane dissertation eq 3.4
            length = float(np.sum(fp_lengths * fp_surface_lengths) / np.sum(fp_surface_lengths))
            width = area / length

            # determine representative slope profile
            w_slopes, distance_p = _weighted_slope_average(fp_surface_lengths, fp_slopes, fp_lengths)

            elevs = _representative_normalized_elevations(distance_p, w_slopes)
            slope_scalar = float(abs(elevs[-1]))

            # calculate centroid
            c_px, c_py = _centroid_px(indx, indy)
            centroid_lnglat = self.px_to_lnglat(c_px, c_py)

            # calculate longest flowpath statistics
            fp_longest = np.argmax(fp_lengths)
            fp_longest_vertical = fp_verticals[fp_longest]
            fp_longest_length = fp_lengths[fp_longest]
            fp_longest_slope = fp_longest_vertical / fp_longest_length

            subs_d[str(sub_id)] = dict(sub_id=int(sub_id),
                        area=float(area),
                        length=float(length),
                        width=float(width),
                        w_slopes=list(w_slopes),
                        distance_p=list(distance_p),
                        centroid_lnglat=[float(v) for v in centroid_lnglat],
                        elevs=list(elevs),
                        slope_scalar=float(slope_scalar),
                        fp_longest=float(fp_longest),
                        fp_longest_length=float(fp_longest_length),
                        fp_longest_slope=float(fp_longest_slope)
                        )

        with open(_join(self.wd, 'subcatchments.json'), 'w') as fp:
            json.dump(subs_d, fp, indent=2, sort_keys=True)

    def abstract_structure(self, verbose=False):
        translator = self.translator
        topaz_network = self.topaz_network
        top2tau = self.topaz2tau_translator_factory()

        # now we are going to define the lines of the structure file
        # this doesn't handle impoundments

        structure = []
        for chn_id in translator.iter_chn_ids():
            if verbose:
                print('abstracting structure for channel %s...' % chn_id)
            top = translator.top(chn_id=chn_id)
            chn_enum = translator.chn_enum(chn_id=chn_id)

            # right subcatchments end in 2
            hright = top - 2
            if not translator.has_top(hright):
                hright = 0

            # left subcatchments end in 3
            hleft = top - 1
            if not translator.has_top(hleft):
                hleft = 0

            # center subcatchments end in 1
            hcenter = top - 3
            if not translator.has_top(hcenter):
                hcenter = 0

            # define structure for channel
            # the first item defines the channel
            _structure = [chn_enum]

            # network is defined from the NETW.TAB file that has
            # already been read into {network}
            # the 0s are appended to make sure it has a length of
            # at least 3
            chns = topaz_network[top] + [0, 0, 0]

            # structure line with top ids
            _structure += [hright, hleft, hcenter] + chns[:3]

            # this is where we would handle impoundments
            # for now no impoundments are assumed
            _structure += [0, 0, 0]

            # and translate topaz to wepp
            structure.append([int(v) for v in _structure])

        with open(_join(self.wd, 'structure.tsv'), 'w') as fp:
            for row in structure:
                fp.write('\t'.join([str(v) for v in row]))
                fp.write('\n')


if __name__ == "__main__":
    wd = _join(_thisdir, 'test')
    dem = _join(_thisdir, 'dem.tif')

    taudem = TauDEMRunner(wd=wd, dem=dem)
    taudem.run_pitremove()
    taudem.run_d8flowdir()
    taudem.run_aread8()
    taudem.run_gridnet()
    taudem.run_src_threshold()
    taudem.run_moveoutletstostrm(long=-111.784228758779406, lat=41.743629188805421)  # Logan
    # taudem.run_moveoutletstostrm(long=-120.1652, lat=39.1079)  # Blackwood

    taudem.run_peukerdouglas()
    taudem.run_peukerdouglas_stream_delineation()  #threshold=10)
    taudem.run_streamnet()
    taudem.run_dinfflowdir()
    taudem.run_areadinf()
    taudem.run_dinfdistdown()

    taudem.delineate_subcatchments()
    taudem.abstract_channels()
    taudem.abstract_subcatchments()
    taudem.abstract_structure()

    import sys
    sys.exit()

