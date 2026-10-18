"""The local metric frame of a world and rasters on it (numpy + pyproj).

A world is built in a local transverse-Mercator frame centred on the take-off point: x east, y north, metres,
z = elevation above the take-off ground (so the drone starts at z ~ 0, where the rest of the drone scene was
built).  Within the ~10 km of a world the scale error of this projection is below 1e-6.

``Grid`` is a north-up raster on that frame: ``data[j, i]`` is the value at the centre of cell (i, j),
x = x0 + (i + 0.5) * res, y = y1 - (j + 0.5) * res.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    lat0: float
    lon0: float
    z0: float = 0.0                   # elevation (m above the DEM's datum) of local z = 0

    @property
    def proj4(self) -> str:
        return (f"+proj=tmerc +lat_0={self.lat0:.9f} +lon_0={self.lon0:.9f} +k=1 +x_0=0 +y_0=0 "
                "+ellps=WGS84 +units=m +no_defs")

    def _transformers(self):
        from pyproj import Transformer
        fwd = Transformer.from_crs("EPSG:4326", self.proj4, always_xy=True)
        inv = Transformer.from_crs(self.proj4, "EPSG:4326", always_xy=True)
        return fwd, inv

    def to_local(self, lon, lat):
        """(lon, lat) degrees -> (x, y) metres."""
        fwd, _ = self._transformers()
        return fwd.transform(np.asarray(lon, float), np.asarray(lat, float))

    def to_lonlat(self, x, y):
        """(x, y) metres -> (lon, lat) degrees."""
        _, inv = self._transformers()
        return inv.transform(np.asarray(x, float), np.asarray(y, float))

    def bbox_lonlat(self, rect, pad_m: float = 0.0) -> tuple[float, float, float, float]:
        """(west, south, east, north) degrees covering the local rect (x0, x1, y0, y1) plus pad_m."""
        x0, x1, y0, y1 = rect
        xs = np.linspace(x0 - pad_m, x1 + pad_m, 9)
        ys = np.linspace(y0 - pad_m, y1 + pad_m, 9)
        X, Y = np.meshgrid(xs, ys)
        border = (X == X.min()) | (X == X.max()) | (Y == Y.min()) | (Y == Y.max())
        lon, lat = self.to_lonlat(X[border], Y[border])
        return float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))


@dataclass
class Grid:
    data: np.ndarray
    x0: float
    y1: float
    res: float

    @classmethod
    def empty(cls, rect, res: float, fill=0.0, dtype=np.float32) -> "Grid":
        """A grid covering rect = (x0, x1, y0, y1) (rounded outwards to whole cells)."""
        x0, x1, y0, y1 = rect
        nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
        return cls(np.full((ny, nx), fill, dtype), float(x0), float(y0 + ny * res), float(res))

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[:2]

    @property
    def rect(self) -> tuple[float, float, float, float]:
        ny, nx = self.shape
        return self.x0, self.x0 + nx * self.res, self.y1 - ny * self.res, self.y1

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        """(X, Y) of every cell centre, each shaped like data[..., 0]."""
        ny, nx = self.shape
        xs = self.x0 + (np.arange(nx) + 0.5) * self.res
        ys = self.y1 - (np.arange(ny) + 0.5) * self.res
        return np.meshgrid(xs, ys)

    def index(self, x, y):
        """Fractional (i, j) cell coordinates of points (cell centres at integers)."""
        return (np.asarray(x, float) - self.x0) / self.res - 0.5, (self.y1 - np.asarray(y, float)) / self.res - 0.5

    def sample(self, x, y, method: str = "bilinear"):
        """Values at points; outside the grid the nearest edge cell is used."""
        fi, fj = self.index(x, y)
        ny, nx = self.shape
        if method == "nearest":
            i = np.clip(np.rint(fi).astype(int), 0, nx - 1)
            j = np.clip(np.rint(fj).astype(int), 0, ny - 1)
            return self.data[j, i]
        fi, fj = np.clip(fi, 0, nx - 1), np.clip(fj, 0, ny - 1)
        i0, j0 = np.minimum(np.floor(fi).astype(int), nx - 2), np.minimum(np.floor(fj).astype(int), ny - 2)
        i0, j0 = np.maximum(i0, 0), np.maximum(j0, 0)
        u, v = fi - i0, fj - j0
        d = self.data.astype(np.float64)
        if nx == 1 or ny == 1:
            return d[np.clip(np.rint(fj).astype(int), 0, ny - 1), np.clip(np.rint(fi).astype(int), 0, nx - 1)]
        return ((1 - u) * (1 - v) * d[j0, i0] + u * (1 - v) * d[j0, i0 + 1]
                + (1 - u) * v * d[j0 + 1, i0] + u * v * d[j0 + 1, i0 + 1])

    def resample(self, rect, res: float, method: str = "bilinear") -> "Grid":
        out = Grid.empty(rect, res, dtype=self.data.dtype if method == "nearest" else np.float32)
        X, Y = out.centres()
        out.data = self.sample(X, Y, method).astype(out.data.dtype)
        return out


# ---------------------------------------------------------------------------
# Sun position (NOAA solar calculator equations; ~0.01 deg for 1800-2100)
# ---------------------------------------------------------------------------

def sun_position(lat: float, lon: float, when_utc) -> tuple[float, float]:
    """(elevation_deg, azimuth_deg) of the sun, azimuth clockwise from north, at a UTC datetime."""
    import datetime as _dt

    if when_utc.tzinfo is not None:
        when_utc = when_utc.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    jd = (when_utc - _dt.datetime(2000, 1, 1, 12)).total_seconds() / 86400.0 + 2451545.0
    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + T * (36000.76983 + T * 0.0003032)) % 360
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    Mr = math.radians(M)
    C = (math.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T)) + math.sin(2 * Mr) * (0.019993 - 0.000101 * T)
         + math.sin(3 * Mr) * 0.000289)
    true_long = L0 + C
    omega = 125.04 - 1934.136 * T
    lam = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60
    eps = math.radians(eps0 + 0.00256 * math.cos(math.radians(omega)))
    decl = math.asin(math.sin(eps) * math.sin(math.radians(lam)))
    y = math.tan(eps / 2) ** 2
    L0r = math.radians(L0)
    eot = 4 * math.degrees(y * math.sin(2 * L0r) - 2 * e * math.sin(Mr) + 4 * e * y * math.sin(Mr) * math.cos(2 * L0r)
                           - 0.5 * y * y * math.sin(4 * L0r) - 1.25 * e * e * math.sin(2 * Mr))        # minutes
    minutes = when_utc.hour * 60 + when_utc.minute + when_utc.second / 60
    tst = (minutes + eot + 4 * lon) % 1440
    ha = math.radians(tst / 4 - 180 if tst / 4 >= 0 else tst / 4 + 180)
    latr = math.radians(lat)
    cz = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cz)))
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(latr) - math.tan(decl) * math.cos(latr))) + 180
    return 90.0 - math.degrees(zen), az % 360


def compass_to_ccw_from_east(azimuth_deg: float) -> float:
    """Compass azimuth (clockwise from north) -> angle counter-clockwise from +x (east), as SunCfg uses."""
    return (90.0 - azimuth_deg) % 360
