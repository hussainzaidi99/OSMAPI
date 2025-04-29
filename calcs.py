import math
import requests
import io

from PIL import Image, ImageDraw, ImageFont
from urllib.parse import urlencode
import numpy as np
import cv2

# ← YOUR KEYS ↓
GOOGLE_API_KEY = "AIzaSyDSgL71XaWTeDuDgdLHBeP_HRYVYXLnbdo"
MAPBOX_TOKEN   = "pk.eyJ1IjoiaHVzc2FpbjEyMzQ1IiwiYSI6ImNtOXR3ZmNsaDAzYm4ycXIyODFmazZtMWwifQ.HGtNa3Hr4_Bkg0IQXzBrdQ"

# map & image settings
ZOOM    = 20
IMG_W, IMG_H = 600, 600


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    φ1, φ2 = map(math.radians, (lat1, lat2))
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def latlng_to_world(lat, lng, zoom):
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    scale = 256 * 2**zoom
    x = (lng + 180) / 360 * scale
    y = (0.5 - math.log((1 + siny)/(1 - siny)) / (4 * math.pi)) * scale
    return x, y


def world_to_latlng(wx, wy, zoom):
    scale = 256 * 2**zoom
    lon = (wx / scale) * 360.0 - 180.0
    y_norm = 0.5 - (wy / scale)
    lat = math.degrees(2 * math.atan(math.exp(4*math.pi*y_norm)) - math.pi/2)
    return lat, lon


def latlng_to_image_px(lat, lng, center, zoom, img_size=(IMG_W, IMG_H)):
    wx, wy = latlng_to_world(lat, lng, zoom)
    cx, cy = latlng_to_world(center[0], center[1], zoom)
    return (wx - cx) + img_size[0]/2, (wy - cy) + img_size[1]/2


def image_px_to_latlng(px, py, center, zoom, img_size=(IMG_W, IMG_H)):
    cx, cy = latlng_to_world(center[0], center[1], zoom)
    wx = cx + (px - img_size[0]/2)
    wy = cy + (py - img_size[1]/2)
    return world_to_latlng(wx, wy, zoom)


def build_map(lat, lng, zoom=ZOOM, size=(IMG_W, IMG_H)):
    params = {
        "center":  f"{lat},{lng}",
        "zoom":    str(zoom),
        "size":    f"{size[0]}x{size[1]}",
        "maptype": "satellite",
        "key":     GOOGLE_API_KEY
    }
    url = f"https://maps.googleapis.com/maps/api/staticmap?{urlencode(params, safe=':,')}"
    r = requests.get(url); r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


def fetch_mapbox_building(lat, lng, radius=10):
    url = (
        f"https://api.mapbox.com/v4/"
        f"mapbox.mapbox-streets-v8/tilequery/{lng},{lat}.json"
        f"?layers=building&radius={radius}&limit=1"
        f"&access_token={MAPBOX_TOKEN}"
    )
    r = requests.get(url); r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        return None
    geom = feats[0]["geometry"]
    if geom["type"] == "Polygon":
        ring = geom["coordinates"][0]
    elif geom["type"] == "MultiPolygon":
        ring = geom["coordinates"][0][0]
    else:
        return None
    return [{"lat": pt[1], "lon": pt[0]} for pt in ring]


def fetch_osm_building(lat, lng, radii=(5,15,30)):
    overpass = "https://overpass-api.de/api/interpreter"
    for r in radii:
        q = f"""
        [out:json][timeout:25];
        way(around:{r},{lat},{lng})["building"];
        out geom;
        """
        res = requests.post(overpass, data={"data": q}); res.raise_for_status()
        elems = res.json().get("elements", [])
        if elems:
            best = max(elems, key=lambda e: len(e.get("geometry", [])))
            return [{"lat": v["lat"], "lon": v["lon"]} for v in best["geometry"]]
    return None


def fallback_box(lat, lng, meters=20):
    m_lat = 111320
    m_lng = 111320 * math.cos(math.radians(lat))
    d_lat, d_lng = meters/m_lat, meters/m_lng
    sw = (lat - d_lat, lng - d_lng)
    ne = (lat + d_lat, lng + d_lng)
    return [sw, (sw[0], ne[1]), ne, (ne[0], sw[1])]


def measure_and_annotate(lat, lng, zoom=ZOOM, size=(IMG_W, IMG_H)):
    # 1) fetch
    sat = build_map(lat, lng, zoom, size)
    # 2) footprint
    fp = fetch_mapbox_building(lat, lng) or fetch_osm_building(lat, lng)
    overlay = sat.convert("RGBA")
    draw    = ImageDraw.Draw(overlay, "RGBA")
    font    = ImageFont.load_default()

    length_m = width_m = None
    if fp and len(fp) >= 4:
        pts = np.array([
            latlng_to_image_px(v["lat"], v["lon"], (lat,lng), zoom, size)
            for v in fp
        ], dtype=np.float32)
        poly = [(int(x),int(y)) for x,y in pts]
        draw.polygon(poly, outline="red", width=2)

        rect = cv2.minAreaRect(pts)
        box  = cv2.boxPoints(rect).astype(int)
        e1, e2 = (tuple(box[0]),tuple(box[1])), (tuple(box[1]),tuple(box[2]))

        # draw lines
        draw.line(e1, fill="blue", width=4)
        draw.line(e2, fill="blue", width=4)

        # measure
        def m(e):
            (x1,y1),(x2,y2) = e
            la1,lo1 = image_px_to_latlng(x1,y1,(lat,lng),zoom,size)
            la2,lo2 = image_px_to_latlng(x2,y2,(lat,lng),zoom,size)
            return haversine(la1,lo1,la2,lo2)

        length_m, width_m = m(e1), m(e2)

        # annotate
        def an(e,txt,off=12):
            (x1,y1),(x2,y2) = e
            mx,my = (x1+x2)/2,(y1+y2)/2
            dx,dy = x2-x1,y2-y1
            L = math.hypot(dx,dy)
            nx,ny = -dy/L, dx/L
            tx,ty = int(mx+nx*off), int(my+ny*off)
            x0,y0,x1b,y1b = draw.textbbox((0,0), txt, font=font)
            tw,th = x1b-x0, y1b-y0
            draw.rectangle([(tx-tw//2-2,ty-th//2-2),(tx+tw//2+2,ty+th//2+2)],
                           fill=(0,0,0,160))
            draw.text((tx-tw//2,ty-th//2), txt, fill="white", font=font)

        an(e1, f"{length_m:.1f} m")
        an(e2, f"{width_m :.1f} m")

    else:
        box_geo = fallback_box(lat, lng)
        pxs = [latlng_to_image_px(a,b,(lat,lng),zoom,size) for a,b in box_geo]
        poly = [(int(x),int(y)) for x,y in pxs]
        draw.polygon(poly, outline="red", width=2)

    return length_m, width_m, overlay
