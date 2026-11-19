# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
"""The phone page, /m: one small HTML page (no NiceGUI, no websocket) that asks /m/status for a few hundred bytes of
JSON every 10 s while training runs and every 60 s otherwise, and not at all while the phone's screen or tab is off.
It shows the run, its progress and time left, a failure with the last log lines, the queue and the newest sample
picture (/m/sample, a small JPEG), with Stop now, Stop after this run and Start the queue. Behind the normal login
(login.allowed); open it at http://<this computer>:<port>/m with Settings › Start and network › Reachable on the
network. The data comes from Other/mobile.py."""
from __future__ import annotations
__version__ = "1.191126h"  # LoRA Studio version, see version.py

import hashlib
import io
import json

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from nicegui import app

import login
import mobile

_THUMB = {"key": None, "data": b""}

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#1C1F24">
<title>LoRA Studio</title>
<link rel="icon" href="/m/icon.png">
<style>
:root{--bg:#F4F2EE;--card:#fff;--ink:#1C1F24;--mute:#6B6F76;--line:#E3E0DA;--acc:#2F6FDB;--ok:#1F8A5B;--bad:#C2410C;--warn:#B7791F}
@media (prefers-color-scheme:dark){:root{--bg:#15171B;--card:#1F2228;--ink:#ECEDEF;--mute:#9AA0A8;--line:#2E3239;--acc:#6EA0FF;--ok:#4CC38A;--bad:#FF7A4D;--warn:#F2B84B}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:center;gap:8px;padding:12px 16px;background:#1C1F24;color:#fff}
header b{font-size:17px}header span{margin-left:auto;font-size:12px;opacity:.7}
main{padding:12px 16px 32px;display:flex;flex-direction:column;gap:12px;max-width:560px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.state{font-size:13px;font-weight:600;letter-spacing:.02em;text-transform:uppercase;color:var(--mute)}
.state.run{color:var(--acc)}.state.done{color:var(--ok)}.state.fail{color:var(--bad)}
.name{font-size:19px;font-weight:650;margin:2px 0 10px;word-break:break-word}
.bar{height:12px;border-radius:6px;background:var(--line);overflow:hidden}.bar i{display:block;height:100%;background:var(--acc);transition:width .6s}
.big{display:flex;justify-content:space-between;align-items:baseline;margin-top:8px}
.big b{font-size:28px;font-variant-numeric:tabular-nums}.big span{color:var(--mute)}
.facts{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin-top:10px;font-size:14px}
.facts dt{color:var(--mute)}.facts dd{margin:0;font-variant-numeric:tabular-nums}
.warn{color:var(--warn);font-weight:600;margin-top:8px}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.35 ui-monospace,Menlo,monospace;background:var(--bg);border-radius:8px;padding:8px;margin:8px 0 0;max-height:280px;overflow:auto}
img{width:100%;border-radius:8px;display:block;margin-top:8px}
h2{font-size:14px;margin:0 0 6px;color:var(--mute);font-weight:600}
ul{margin:0;padding-left:18px}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
button{font:inherit;font-weight:600;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:10px 14px;min-height:44px}
button.bad{border-color:var(--bad);color:var(--bad)}button.go{background:var(--acc);border-color:var(--acc);color:#fff}
.msg{font-size:13px;color:var(--mute);min-height:18px}
.foot{font-size:12px;color:var(--mute);text-align:center}
[hidden]{display:none!important}
</style></head><body>
<header><b>LoRA Studio</b><span id="ago"></span></header>
<main>
 <section class="card">
  <div id="state" class="state">…</div>
  <div id="name" class="name"></div>
  <div id="prog" hidden><div class="bar"><i id="fill" style="width:0"></i></div>
   <div class="big"><b id="pct"></b><span id="left"></span></div></div>
  <div id="line" class="msg"></div>
  <dl id="facts" class="facts"></dl>
  <div id="nan" class="warn" hidden>⚠ The loss is NaN: this run is not learning anymore.</div>
  <pre id="err" hidden></pre>
  <div class="btns" id="runbtns" hidden>
   <button class="bad" onclick="act('stop','Stop the training now?')">Stop now</button>
   <button id="after" onclick="act('after','Stop the queue after this run?')">Stop after this run</button>
  </div>
 </section>
 <section class="card" id="samplecard" hidden>
  <h2>Newest sample picture <span id="sname"></span></h2>
  <img id="sample" alt="newest sample picture" loading="lazy">
 </section>
 <section class="card">
  <h2>Queue</h2>
  <div id="queue" class="msg"></div>
  <ul id="next"></ul>
  <div class="btns"><button class="go" id="start" hidden onclick="act('start','Start the queue?')">Start the queue</button></div>
 </section>
 <div id="msg" class="msg"></div>
 <div class="foot">Updates every 10 s while training, every minute otherwise · <a href="/" style="color:inherit">full page</a></div>
</main>
<script>
const $=id=>document.getElementById(id);let last=null,etag="",timer=null,got=0;
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function show(d){
 const st={run:"Training",pics:"Making test pictures",done:"Finished",fail:"Failed",idle:"Idle"}[d.state]||d.state;
 $("state").textContent=st;$("state").className="state "+d.state;
 const L=d.last;
 $("name").textContent=d.state==="run"?d.name:(d.state==="pics"?d.pics:(L&&d.state!=="idle"?L.name:"Nothing is training"));
 const run=d.state==="run";$("prog").hidden=!(run&&d.total);$("runbtns").hidden=!run;
 if(run&&d.total){$("fill").style.width=d.pct+"%";$("pct").textContent=Math.floor(d.pct)+" %";$("left").textContent=d.eta?d.eta+" left":""}
 $("line").textContent=run?(d.stage||d.line||""):(d.state==="idle"&&L?("Last: "+L.name+(L.ok?" finished ":" failed ")+L.day+" "+L.when):"");
 let f=[];
 if(run){if(d.total)f.push(["Step",d.step.toLocaleString()+" / "+d.total.toLocaleString()]);
  if(d.epochs)f.push(["Epoch",d.epoch+" / "+d.epochs]);if(d.loss!==null)f.push(["Loss",d.loss]);
  if(d.elapsed)f.push(["Running for",d.elapsed])}
 else if(L&&(d.state==="done"||d.state==="fail")){f.push([L.ok?"Finished":"Failed",L.day+" "+L.when]);if(L.advice)f.push(["Note",L.advice])}
 $("facts").innerHTML=f.map(x=>"<dt>"+esc(x[0])+"</dt><dd>"+esc(x[1])+"</dd>").join("");
 $("nan").hidden=!(run&&d.nan);
 $("err").hidden=!(d.error&&d.error.length);$("err").textContent=(d.error||[]).join("\n");
 $("after").hidden=!d.queue.running||d.queue.stop_after;
 const q=d.queue;$("queue").textContent=(q.waiting?q.waiting+" waiting":"Nothing waiting")+(q.running?(q.stop_after?" · stops after this run":" · running"):"");
 $("next").innerHTML=q.next.map(n=>"<li>"+esc(n)+"</li>").join("");
 $("start").hidden=!(q.waiting&&!q.running&&!run);
 if(d.sample){$("samplecard").hidden=false;$("sname").textContent="· "+d.sample.when;
  const u="/m/sample?v="+d.sample.v;if($("sample").getAttribute("src")!==u)$("sample").src=u}else $("samplecard").hidden=true;
 last=d;
}
async function poll(){
 clearTimeout(timer);if(document.hidden)return;
 try{const r=await fetch("/m/status",{headers:etag?{"If-None-Match":etag}:{},cache:"no-store"});
  if(r.status===401){location.href="/login?next=/m";return}
  if(r.status===200){etag=r.headers.get("ETag")||"";show(await r.json())}
  got=Date.now();$("msg").textContent=""}
 catch(e){$("msg").textContent="LoRA Studio does not answer (is the computer on, and on this network?)"}
 timer=setTimeout(poll,last&&(last.state==="run"||last.state==="pics")?10000:60000)}
setInterval(()=>{if(got){const s=Math.round((Date.now()-got)/1000);$("ago").textContent=s<5?"just now":s+" s ago"}},1000);
async function act(a,q){if(!confirm(q))return;
 const r=await fetch("/m/"+a,{method:"POST"});const j=await r.json().catch(()=>({msg:"?"}));$("msg").textContent=j.msg;etag="";setTimeout(poll,800)}
document.addEventListener("visibilitychange",()=>{if(!document.hidden){etag="";poll()}});
poll();
</script></body></html>"""


def _no():
    return JSONResponse({"msg": "Log in first."}, status_code=401)


@app.get("/m")
def _page():
    if not login.allowed():
        return RedirectResponse("/login?next=/m")
    return HTMLResponse(PAGE, headers={"Cache-Control": "no-cache"})


@app.get("/m/status")
def _status(request: Request):
    if not login.allowed():
        return _no()
    d = mobile.status()
    body = json.dumps(d, separators=(",", ":"))
    tag = '"' + hashlib.md5(json.dumps({k: v for k, v in d.items() if k != "t"}, sort_keys=True).encode()).hexdigest()[:16] + '"'
    if request.headers.get("if-none-match") == tag:
        return Response(status_code=304, headers={"ETag": tag})
    return Response(body, media_type="application/json", headers={"ETag": tag, "Cache-Control": "no-store"})


@app.get("/m/sample")
def _sample():
    """The newest sample picture, at most 640 px wide, as JPEG (kept until a newer one appears)."""
    if not login.allowed():
        return _no()
    p = mobile.sample_file()
    if p is None:
        return Response(status_code=404)
    try:
        key = (str(p), p.stat().st_mtime_ns)
    except OSError:
        return Response(status_code=404)
    if _THUMB["key"] != key:
        from PIL import Image
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((640, 640))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=82)
        except OSError:
            return Response(status_code=404)
        _THUMB.update(key=key, data=buf.getvalue())
    return Response(_THUMB["data"], media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/m/icon.png")
def _icon():
    import lspaths
    f = lspaths.PROG / "logo" / "favicon.png"
    return Response(f.read_bytes() if f.is_file() else b"", media_type="image/png",
                    headers={"Cache-Control": "max-age=604800"})


def _act(fn):
    if not login.allowed():
        return _no()
    return JSONResponse({"msg": fn()})


@app.post("/m/stop")
def _stop():
    return _act(mobile.stop_now)


@app.post("/m/after")
def _after():
    return _act(mobile.stop_after)


@app.post("/m/start")
def _start():
    return _act(mobile.start_queue)
