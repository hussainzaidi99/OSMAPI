from flask import Flask, request, jsonify
import io
import base64
from calcs import measure_and_annotate

app = Flask(__name__)

@app.route("/measure", methods=["GET"])
def measure():
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    if lat is None or lng is None:
        return jsonify({"error": "Missing lat/lng"}), 400

    length_m, width_m, img = measure_and_annotate(lat, lng)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    
    length_ft = length_m * 3.28084 if length_m is not None else None
    width_ft  = width_m  * 3.28084 if width_m  is not None else None

    return jsonify({
        "length_m":  round(length_m, 2) if length_m is not None else None,
        "width_m":   round(width_m,  2) if width_m  is not None else None,
        "length_ft": round(length_ft, 1) if length_ft is not None else None,
        "width_ft":  round(width_ft,  1) if width_ft  is not None else None,
        "image_base64": img_base64
    })

if __name__ == "__main__":
    app.run()
