# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""Display-only parts of the page (the status strip, the training queue) refreshed without Gradio events.

Every Gradio event, however small, costs the browser about 0.3 s on LoRA Studio's big page, and the shared timer
made one every few seconds: clicks and typing stuttered. These parts only show HTML, so the page fetches them from one
small address (/ls_live, behind the same login as the page) every 2.5 seconds and swaps the HTML in itself; nothing
else on the page is touched. Gradio still redraws them after your own clicks as before.
"""
from __future__ import annotations
__version__ = "1.101126c"  # LoRA Studio version, see version.py

SOURCES: dict = {}            # elem_id -> function returning the HTML
ROUTE = "ls_live"

JS = """
<script>
(function(){
  var seen = {}, busy = false, was = false;
  function url(){ var p = location.pathname; return p.slice(0, p.lastIndexOf('/') + 1) + '%s'; }
  async function poll(){
    if (busy || document.hidden) return;
    busy = true;
    try {
      var r = await fetch(url(), {credentials: 'same-origin', cache: 'no-store'});
      if (r.ok) {
        var d = await r.json();
        if (d._busy && !was) { var w = document.getElementById('ls_wake'); if (w) w.click(); }
        was = !!d._busy; delete d._busy;
        for (var id in d) {
          var host = document.getElementById(id); if (!host) continue;
          var el = host.querySelector('.prose') || host;
          if (seen[id] !== d[id] || el.innerHTML.length === 0) { el.innerHTML = d[id]; seen[id] = d[id]; }
        }
      }
    } catch (e) {}
    busy = false;
  }
  setInterval(poll, 2500);
  document.addEventListener('visibilitychange', poll);
  // right after a click (a run or a download may just have started): look every second for a few seconds
  var quick = null;
  document.addEventListener('click', function(){
    var n = 0; clearInterval(quick);
    quick = setInterval(function(){ poll(); if (++n >= 6) clearInterval(quick); }, 1000);
  }, true);
})();
</script>
<style>#ls_wake { display: none !important; }</style>
""" % ROUTE


def register(elem_id: str, fn):
    SOURCES[elem_id] = fn


def snapshot() -> dict:
    import ticker
    out = {"_busy": ticker.is_busy()}
    for k, fn in SOURCES.items():
        try:
            out[k] = fn()
        except Exception:                                # noqa: BLE001 - one part failing must not stop the others
            pass
    return out


def mount(blocks) -> bool:
    """Add /ls_live to the running Gradio app (after launch(prevent_thread_lock=True)). False when not possible."""
    try:
        import fastapi
        api = getattr(blocks, "server_app", None) or blocks.app

        async def user_of(request):
            if getattr(api, "auth_dependency", None) is not None:
                u = api.auth_dependency(request)
                return await u if hasattr(u, "__await__") else u
            cid = getattr(api, "cookie_id", "")
            token = request.cookies.get(f"access-token-{cid}") or request.cookies.get(f"access-token-unsecure-{cid}")
            return getattr(api, "tokens", {}).get(token)

        async def live(request: fastapi.Request):
            if (getattr(api, "auth", None) or getattr(api, "auth_dependency", None)) and not await user_of(request):
                raise fastapi.HTTPException(status_code=401, detail="Not authenticated")
            import anyio
            return await anyio.to_thread.run_sync(snapshot)

        live.__annotations__ = {"request": fastapi.Request}   # (this file uses postponed annotations)
        api.add_api_route("/" + ROUTE, live, methods=["GET"])
        import ticker
        ticker.SLEEP["on"] = True                 # the page can wake the shared timer now
        # Gradio may have a catch-all route: put ours first
        api.router.routes.insert(0, api.router.routes.pop())
        return True
    except Exception as e:                               # noqa: BLE001
        print(f"[livepoll] not available: {e!r}")
        return False
