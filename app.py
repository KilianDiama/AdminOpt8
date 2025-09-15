#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdminOpt8 — Monolithe "10/10" prêt à lancer

Fonctionnalités :
- Moteur TDSA (test séquentiel optimal) pour décisions administratives binaires.
- API REST (FastAPI) + mini-UI HTML/JS sans dépendances front externes.
- Catalogue de "preuves" (evidence) avec coût et puissance (TPR/FPR) modélisées.
- Choix adaptatif de la prochaine preuve par gain d'information attendu / coût.
- Journalisation par session, traçabilité, export.
- Support d'un catalogue custom (POST /api/catalog pour remplacer le défaut).

Démarrage rapide :
1) pip install fastapi uvicorn "pydantic>=2" python-multipart
2) python adminopt8.py
3) Ouvre http://127.0.0.1:8000

Notes :
- Tout tient dans ce fichier.
- Persistance simple en mémoire (pour POC). Pour la prod, brancher un vrai stockage.
- Le modèle de preuve binaire (succès/échec) avec (TPR,FPR,cost) est volontairement simple,
  robuste, traçable juridiquement et suffisant pour de gros gains. On peut l'étendre.

Auteur : "AdminOpt8" (CC0 / domaine public — faites-en ce que vous voulez)
"""
from __future__ import annotations
import math, json, time, uuid, copy
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------------------
# 1) Modèle "preuve" binaire : succès/échec avec (TPR, FPR, coût)
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    key: str                   # identifiant unique (ex: "SIREN_VALID")
    label: str                 # libellé humain
    cost: float                # coût unitaire (€) (SI + humain)
    tpr: float                 # P(success | H1)  (vrai positif)
    fpr: float                 # P(success | H0)  (faux positif)
    description: str = ""      # description
    # Sanity checks
    def validate(self):
        if not (0 < self.cost):
            raise ValueError(f"Cost must be > 0 for {self.key}")
        for name, v in [("tpr", self.tpr), ("fpr", self.fpr)]:
            if not (0 <= v <= 1):
                raise ValueError(f"{name} must be in [0,1] for {self.key}")
        # éviter division par zero dans les LLR :
        eps = 1e-12
        self.tpr = min(max(self.tpr, eps), 1 - eps)
        self.fpr = min(max(self.fpr, eps), 1 - eps)

    def llr(self, outcome_success: bool) -> float:
        """
        Log-likelihood ratio :
        - Si success: log( P1(success)/P0(success) ) = log( tpr / fpr )
        - Sinon     : log( (1-tpr) / (1-fpr) )
        """
        if outcome_success:
            return math.log(self.tpr / self.fpr)
        else:
            return math.log((1.0 - self.tpr) / (1.0 - self.fpr))

    def expected_abs_llr(self, pi1: float) -> float:
        """
        Espérance de |LLR| sous le mélange postérieur (pi1=P(H1|data), pi0=1-pi1).
        Pour binaire, E[|LLR|] = sum_z |llr(z)| * [ pi1*P1(z) + (1-pi1)*P0(z) ].
        """
        pi0 = 1.0 - pi1
        # z = success
        llr_s = abs(self.llr(True))
        p_mix_s = pi1 * self.tpr + pi0 * self.fpr
        # z = failure
        llr_f = abs(self.llr(False))
        p_mix_f = pi1 * (1.0 - self.tpr) + pi0 * (1.0 - self.fpr)
        return llr_s * p_mix_s + llr_f * p_mix_f

# ---------------------------------------------------------------------------
# 2) Moteur TDSA (SPRT) avec choix adaptatif "info/coût"
# ---------------------------------------------------------------------------

@dataclass
class TDSASession:
    session_id: str
    alpha: float
    beta: float
    A: float
    B: float
    logA: float
    logB: float
    S: float = 0.0
    prior_odds: float = 1.0    # odds H1/H0 (par défaut 50/50)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    decided: Optional[str] = None   # "ACCEPTER" (H1) / "REFUSER" (H0) / None
    catalog: Dict[str, Evidence] = field(default_factory=dict)
    used_keys: set = field(default_factory=set)

    def current_posterior_pi1(self) -> float:
        # Posterior odds = prior_odds * exp(S)
        odds = self.prior_odds * math.exp(self.S)
        return odds / (1.0 + odds)

    def status(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "alpha": self.alpha,
            "beta": self.beta,
            "S": self.S,
            "logA": self.logA,
            "logB": self.logB,
            "thresholds": {"A": self.A, "B": self.B},
            "posterior_pi1": self.current_posterior_pi1(),
            "decided": self.decided,
            "steps": len(self.trace),
        }

    def next_best(self) -> Optional[Dict[str, Any]]:
        """
        Choisit la prochaine preuve maximisant E[|ΔS|]/cost (gain info par €).
        Ignore les preuves déjà utilisées.
        """
        if self.decided:
            return None
        pi1 = self.current_posterior_pi1()
        best = None
        for k, ev in self.catalog.items():
            if k in self.used_keys:
                continue
            score = ev.expected_abs_llr(pi1) / ev.cost
            cand = {
                "key": k,
                "label": ev.label,
                "cost": ev.cost,
                "tpr": ev.tpr,
                "fpr": ev.fpr,
                "score_info_per_euro": score,
                "description": ev.description,
            }
            if (best is None) or (cand["score_info_per_euro"] > best["score_info_per_euro"]):
                best = cand
        return best

    def add_observation(self, key: str, outcome_success: bool) -> Dict[str, Any]:
        if self.decided:
            return {"decided": self.decided, "S": self.S, "already_decided": True}
        if key not in self.catalog:
            raise KeyError(f"Evidence key not in catalog: {key}")
        ev = self.catalog[key]
        llr_val = ev.llr(outcome_success)
        self.S += llr_val
        self.used_keys.add(key)
        step = {
            "t": time.time(),
            "key": key,
            "label": ev.label,
            "cost": ev.cost,
            "outcome": "success" if outcome_success else "failure",
            "llr": llr_val,
            "S_after": self.S,
        }
        self.trace.append(step)
        # Décision ?
        if self.S >= self.logA:
            self.decided = "ACCEPTER"
        elif self.S <= self.logB:
            self.decided = "REFUSER"
        return {"step": step, "S": self.S, "decided": self.decided}

# ---------------------------------------------------------------------------
# 3) Catalogue par défaut (simple, réutilisable partout)
# ---------------------------------------------------------------------------

def default_catalog() -> Dict[str, Evidence]:
    """
    NB: Les TPR/FPR ci-dessous sont illustratifs, à calibrer par administration.
    - SIREN/INSEE: très discriminant pour entreprises légitimes vs fictives.
    - DGFIP_Income: cohérence de revenus (moyennement discriminant).
    - HomeProof: justificatif domicile (faible, souvent redondant, coûteux).
    - SanctionsCheck: croisement sanctions/PEP (assez discriminant, coût faible).
    - OnSiteAudit: contrôle terrain (très discriminant mais très coûteux).
    """
    raw = [
        Evidence(
            key="SIREN_VALID",
            label="Vérification SIREN/INSEE",
            cost=0.001,
            tpr=0.99,
            fpr=0.20,
            description="Requête API INSEE: existence/activité cohérente."
        ),
        Evidence(
            key="DGFIP_INCOME_OK",
            label="Revenus N-1 cohérents (DGFIP)",
            cost=0.01,
            tpr=0.90,
            fpr=0.35,
            description="Croisement automatisé revenus N-1 (masqués)."
        ),
        Evidence(
            key="HOME_PROOF",
            label="Justificatif de domicile",
            cost=2.50,
            tpr=0.75,
            fpr=0.55,
            description="Upload/OCR + contrôle humain si doute."
        ),
        Evidence(
            key="SANCTIONS_PEP_NEG",
            label="Listes Sanctions/PEP — pas de match",
            cost=0.005,
            tpr=0.97,
            fpr=0.40,
            description="Criblage nominatif listes publiques."
        ),
        Evidence(
            key="ONSITE_AUDIT",
            label="Contrôle terrain (audit court)",
            cost=80.0,
            tpr=0.98,
            fpr=0.10,
            description="Déplacement/visite express ciblée."
        ),
    ]
    cat = {}
    for ev in raw:
        ev.validate()
        cat[ev.key] = ev
    return cat

# ---------------------------------------------------------------------------
# 4) API & App (FastAPI)
# ---------------------------------------------------------------------------

app = FastAPI(title="AdminOpt8", version="1.0.0", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# In-memory state
GLOBAL_CATALOG: Dict[str, Evidence] = default_catalog()
SESSIONS: Dict[str, TDSASession] = {}

# ------------------------ Pydantic Schemas ----------------------------------

class EvidenceIn(BaseModel):
    key: str
    label: str
    cost: float
    tpr: float
    fpr: float
    description: str = ""

class CatalogIn(BaseModel):
    items: List[EvidenceIn]

class SessionCreateIn(BaseModel):
    alpha: float = Field(0.005, ge=1e-6, le=0.2)
    beta: float = Field(0.01, ge=1e-6, le=0.2)
    prior_pi1: float = Field(0.5, ge=1e-6, le=1-1e-6)

class ObserveIn(BaseModel):
    key: str
    outcome: bool

class SimpleResp(BaseModel):
    ok: bool
    message: str = ""
    data: Any = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

# ------------------------ UI (HTML minimal) ---------------------------------

INDEX_HTML = """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <title>AdminOpt8 — Moteur TDSA</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu; margin:20px; color:#111;}
    .card{border:1px solid #ddd; border-radius:14px; padding:16px; margin:10px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.05);}
    .row{display:flex; gap:12px; flex-wrap:wrap; align-items:flex-end}
    input,button,select{padding:8px 10px; border-radius:10px; border:1px solid #bbb; font-size:14px}
    button{background:#111; color:#fff; border:0; cursor:pointer}
    button.secondary{background:#fff; color:#111; border:1px solid #111}
    code{background:#f7f7f7; padding:2px 6px; border-radius:6px}
    table{border-collapse:collapse; width:100%}
    th,td{border-bottom:1px solid #eee; padding:8px; text-align:left; font-size:14px}
    .pill{display:inline-block; background:#eee; padding:2px 8px; border-radius:999px; font-size:12px}
    .ok{color:#0a7f00} .ko{color:#b00020}
  </style>
</head>
<body>
  <h1>AdminOpt8 — Décision séquentielle optimale</h1>

  <div class="card">
    <h3>Créer une session</h3>
    <div class="row">
      <div>
        <label>α faux positif</label><br/>
        <input id="alpha" type="number" step="0.001" value="0.005"/>
      </div>
      <div>
        <label>β faux négatif</label><br/>
        <input id="beta" type="number" step="0.001" value="0.010"/>
      </div>
      <div>
        <label>Prior P(H1)</label><br/>
        <input id="prior" type="number" step="0.01" value="0.50"/>
      </div>
      <button onclick="createSession()">Nouvelle session</button>
      <span id="sid" class="pill"></span>
    </div>
    <div style="margin-top:10px">Seuils calculés : <span id="thresholds"></span></div>
  </div>

  <div class="card">
    <h3>Prochaine preuve recommandée</h3>
    <div id="best"></div>
    <div class="row">
      <button class="secondary" onclick="refreshBest()">Mettre à jour</button>
      <button onclick="observe(true)">Enregistrer : Succès</button>
      <button onclick="observe(false)">Enregistrer : Échec</button>
    </div>
  </div>

  <div class="card">
    <h3>État & journal</h3>
    <div id="status"></div>
    <table id="traceTable">
      <thead><tr><th>#</th><th>Preuve</th><th>Outcome</th><th>LLR</th><th>S</th><th>Coût</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

<script>
let SESSION_ID = null;
let BEST_KEY = null;

async function createSession(){
  const alpha = parseFloat(document.getElementById('alpha').value);
  const beta  = parseFloat(document.getElementById('beta').value);
  const prior = parseFloat(document.getElementById('prior').value);
  const body = {alpha:alpha, beta:beta, prior_pi1:prior};
  const r = await fetch('/api/session', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const j = await r.json();
  if(!j.ok){ alert('Erreur: '+j.message); return; }
  SESSION_ID = j.data.session_id;
  document.getElementById('sid').innerText = 'Session: '+SESSION_ID;
  document.getElementById('thresholds').innerText = 'A='+j.data.thresholds.A.toFixed(2)+' (logA='+j.data.logA.toFixed(2)+'), B='+j.data.thresholds.B.toFixed(4)+' (logB='+j.data.logB.toFixed(2)+')';
  await refreshAll();
}

async function refreshAll(){
  await refreshStatus();
  await refreshBest();
  await refreshTrace();
}

async function refreshStatus(){
  if(!SESSION_ID) return;
  const r = await fetch('/api/status/'+SESSION_ID);
  const j = await r.json();
  if(!j.ok){ return; }
  const s = j.data;
  let decided = s.decided ? ('<span class="'+(s.decided=='ACCEPTER'?'ok':'ko')+'">'+s.decided+'</span>') : '<span class="pill">En cours</span>';
  document.getElementById('status').innerHTML = `
    S = <code>${s.S.toFixed(3)}</code> &nbsp; | &nbsp;
    P(H1|data) ≈ <code>${(s.posterior_pi1*100).toFixed(1)}%</code> &nbsp; | &nbsp;
    Décision : ${decided} &nbsp; | &nbsp;
    Étapes : <code>${s.steps}</code>
  `;
}

async function refreshBest(){
  if(!SESSION_ID) return;
  const r = await fetch('/api/next/'+SESSION_ID);
  const j = await r.json();
  const el = document.getElementById('best');
  if(!j.ok){ el.innerText = '—'; return; }
  if(!j.data){ el.innerHTML = '<i>Aucune preuve (ou décision déjà prise).</i>'; BEST_KEY=null; return; }
  const b = j.data;
  BEST_KEY = b.key;
  el.innerHTML = `
    <b>${b.label}</b> <span class="pill">${b.key}</span><br/>
    Coût estimé : <code>${b.cost.toFixed(3)} €</code> — TPR=${b.tpr.toFixed(2)} ; FPR=${b.fpr.toFixed(2)}<br/>
    Score info/€ : <code>${b.score_info_per_euro.toFixed(4)}</code><br/>
    <small>${b.description||''}</small>
  `;
}

async function refreshTrace(){
  if(!SESSION_ID) return;
  const r = await fetch('/api/trace/'+SESSION_ID);
  const j = await r.json();
  if(!j.ok){ return; }
  const tb = document.querySelector('#traceTable tbody');
  tb.innerHTML = '';
  j.data.forEach((row,idx)=>{
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${idx+1}</td>
      <td>${row.label} <small class="pill">${row.key}</small></td>
      <td>${row.outcome=='success'?'✅ succès':'❌ échec'}</td>
      <td>${row.llr.toFixed(3)}</td>
      <td>${row.S_after.toFixed(3)}</td>
      <td>${row.cost.toFixed(3)} €</td>
    `;
    tb.appendChild(tr);
  });
}

async function observe(success){
  if(!SESSION_ID){ alert('Créez une session.'); return; }
  if(!BEST_KEY){ alert('Rafraîchissez la prochaine preuve.'); return; }
  const r = await fetch('/api/observe/'+SESSION_ID, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({key: BEST_KEY, outcome: success})
  });
  const j = await r.json();
  if(!j.ok){ alert('Erreur: '+j.message); return; }
  await refreshAll();
}
</script>
</body>
</html>
"""

# ------------------------ Routes HTTP ---------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)

@app.get("/health")
def health():
    return {"ok": True, "service": "AdminOpt8", "version": "1.0.0"}

@app.get("/api/catalog")
def get_catalog():
    return JSONResponse(SimpleResp(ok=True, data=[e.__dict__ for e in GLOBAL_CATALOG.values()]).model_dump())

@app.post("/api/catalog")
def set_catalog(cat: CatalogIn):
    global GLOBAL_CATALOG
    new_cat: Dict[str, Evidence] = {}
    try:
        for item in cat.items:
            ev = Evidence(
                key=item.key, label=item.label, cost=float(item.cost),
                tpr=float(item.tpr), fpr=float(item.fpr), description=item.description
            )
            ev.validate()
            new_cat[ev.key] = ev
        if not new_cat:
            raise ValueError("Catalogue vide")
        GLOBAL_CATALOG = new_cat
        # Réinitialiser les sessions ? On garde, mais elles conserveront leur propre copie.
        return JSONResponse(SimpleResp(ok=True, message="Catalogue remplacé.", data=len(GLOBAL_CATALOG)).model_dump())
    except Exception as e:
        return JSONResponse(SimpleResp(ok=False, message=str(e)).model_dump(), status_code=400)

@app.post("/api/session")
def create_session(cfg: SessionCreateIn):
    try:
        alpha = float(cfg.alpha)
        beta  = float(cfg.beta)
        prior_pi1 = float(cfg.prior_pi1)
        A = (1.0 - beta) / alpha
        B = beta / (1.0 - alpha)
        logA = math.log(A)
        logB = math.log(B)
        session_id = uuid.uuid4().hex[:12]
        sess = TDSASession(
            session_id=session_id,
            alpha=alpha, beta=beta,
            A=A, B=B, logA=logA, logB=logB,
            S=0.0,
            prior_odds= prior_pi1 / (1.0 - prior_pi1),
            catalog=copy.deepcopy(GLOBAL_CATALOG),
        )
        SESSIONS[session_id] = sess
        return JSONResponse(SimpleResp(ok=True, data=sess.status()).model_dump())
    except Exception as e:
        return JSONResponse(SimpleResp(ok=False, message=str(e)).model_dump(), status_code=400)

@app.get("/api/status/{session_id}")
def get_status(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        return JSONResponse(SimpleResp(ok=False, message="Session inconnue").model_dump(), status_code=404)
    return JSONResponse(SimpleResp(ok=True, data=sess.status()).model_dump())

@app.get("/api/next/{session_id}")
def get_next(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        return JSONResponse(SimpleResp(ok=False, message="Session inconnue").model_dump(), status_code=404)
    best = sess.next_best()
    return JSONResponse(SimpleResp(ok=True, data=best).model_dump())

@app.get("/api/trace/{session_id}")
def get_trace(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        return JSONResponse(SimpleResp(ok=False, message="Session inconnue").model_dump(), status_code=404)
    return JSONResponse(SimpleResp(ok=True, data=sess.trace).model_dump())

@app.post("/api/observe/{session_id}")
def post_observe(session_id: str, obs: ObserveIn):
    sess = SESSIONS.get(session_id)
    if not sess:
        return JSONResponse(SimpleResp(ok=False, message="Session inconnue").model_dump(), status_code=404)
    try:
        res = sess.add_observation(obs.key, bool(obs.outcome))
        return JSONResponse(SimpleResp(ok=True, data=res).model_dump())
    except KeyError as e:
        return JSONResponse(SimpleResp(ok=False, message=str(e)).model_dump(), status_code=400)
    except Exception as e:
        return JSONResponse(SimpleResp(ok=False, message=f"Erreur observe: {e}").model_dump(), status_code=400)

@app.get("/api/export/{session_id}", response_class=PlainTextResponse)
def export_session(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session inconnue")
    payload = {
        "status": sess.status(),
        "trace": sess.trace,
        "catalog": {k: ev.__dict__ for k, ev in sess.catalog.items()},
    }
    return PlainTextResponse(json.dumps(payload, ensure_ascii=False, indent=2))

@app.post("/api/reset")
def reset_all():
    SESSIONS.clear()
    return JSONResponse(SimpleResp(ok=True, message="Toutes les sessions ont été supprimées.").model_dump())

# ---------------------------------------------------------------------------
# 5) Lancement Uvicorn (si exécuté directement)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        print("\n[!] uvicorn n'est pas installé. Installez-le via :\n    pip install uvicorn\n")
        raise
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
