# Pace Pal API — Koyeb deployment

Same backend as the main project (real video metadata, real quality heuristics, real YOLOv8 ball
tracking via ONNX Runtime — see the main README for details on what is/isn't implemented). This
folder just adds a `Dockerfile` set up for Koyeb's free tier specifically.

## Why Koyeb

- Free tier: 512MB RAM, no credit card required, always-on (no sleep/cold-start, unlike some
  free hosts). This backend measures ~160MB peak RAM, comfortably inside that.
- Trade-off: only 0.1 vCPU (shared) on the free instance, so processing may be slower than on a
  beefier host -- fine for occasional personal use, not for heavy concurrent traffic.

## Deploy steps

1. Push this folder to its own GitHub repo (or reuse the same repo as the Render version -- Koyeb
   just needs `Dockerfile`, `requirements.txt`, and `app/` including `app/models/yolov8n.onnx`).
2. Go to koyeb.com, sign up (no card should be required for the free tier).
3. Create a new Service → Deploy from GitHub → pick your repo.
4. Koyeb should detect the Dockerfile automatically. If asked:
   - Instance type: **Free**
   - Port: leave as detected, or set to match `$PORT` (the Dockerfile handles this automatically)
5. Deploy. Koyeb gives you a URL like `https://your-app-name.koyeb.app`.
6. Test: visit `https://your-app-name.koyeb.app/health` -- should return `{"status":"ok"}`.

## Connecting the frontend

Same as always: set `VITE_API_BASE_URL` to this Koyeb URL in Netlify's environment variables, then
trigger a redeploy.
