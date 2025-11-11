import streamlit as st

st.title("Omega-B")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jack.AI — Super Full (Client-only demo)</title>

<!--
  Jack.AI — Super Full (single-file)
  - Copy/paste into jack_ai_super.html and open in browser
  - All client-side: localStorage persistence, OCR, PDF extraction, TTS, STT,
    image gallery, markdown rendering, link analysis (fetch), transformers.js optional load
  - Simulated on-device fine-tune with progress and weight visualization
  - No server required
-->

<!-- Styles -->
<style>
:root{
  --bg: #071227;
  --panel: #0b1220;
  --muted: #9aa6bf;
  --accent: #4acbff;
  --accent2: #ff6b6b;
  --text: #e6eef8;
  --glass: rgba(255,255,255,0.02);
  --card-shadow: 0 10px 30px rgba(0,0,0,0.6);
  --rounded: 12px;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", "Courier New", monospace;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0;background:linear-gradient(180deg,var(--bg),#02040a);color:var(--text);font-family:Inter,system-ui,Segoe UI,Arial,sans-serif}
.container{max-width:1300px;margin:12px auto;padding:14px}
.header{display:flex;align-items:center;gap:12px}
.logo{width:56px;height:56px;border-radius:10px;background:linear-gradient(135deg,var(--accent),#3a86ff);display:flex;align-items:center;justify-content:center;font-weight:800;color:#001;font-size:22px}
.title{display:flex;flex-direction:column}
.title h1{margin:0;font-size:18px}
.title .subtitle{font-size:12px;color:var(--muted);margin-top:4px}
.layout{display:grid;grid-template-columns:1fr 360px;gap:12px;margin-top:14px}
@media(max-width:1000px){.layout{grid-template-columns:1fr}}
.card{background:var(--panel);border-radius:var(--rounded);padding:12px;box-shadow:var(--card-shadow);border:1px solid rgba(255,255,255,0.02)}
.chat-area{height:720px;display:flex;flex-direction:column}
.chat-log{flex:1;overflow:auto;padding:12px;border-radius:10px;background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01));border:1px solid rgba(255,255,255,0.02)}
.controls{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap}
.input-row{display:flex;gap:8px;margin-top:12px;align-items:flex-start}
#userInput{flex:1;min-height:64px;max-height:300px;padding:10px;border-radius:8px;border:1px solid rgba(255,255,255,0.03);background:rgba(255,255,255,0.02);color:var(--text);resize:vertical}
.small{font-size:13px;color:var(--muted)}
.btn{background:var(--accent);border:0;padding:8px 12px;border-radius:8px;color:#000;font-weight:700;cursor:pointer}
.btn.alt{background:transparent;border:1px solid rgba(255,255,255,0.06);color:var(--text);font-weight:600}
.btn.danger{background:#a63a3a;color:#fff}
.icon-btn{width:40px;height:40px;border-radius:8px;border:1px solid rgba(255,255,255,0.04);display:flex;align-items:center;justify-content:center;background:transparent;cursor:pointer;color:var(--muted)}
.msg{display:flex;margin-bottom:12px;align-items:flex-end}
.msg.user{justify-content:flex-end}
.bubble{max-width:78%;padding:12px;border-radius:12px;color:var(--text);line-height:1.4;white-space:pre-wrap}
.bubble.user{background:linear-gradient(180deg,#0f2133,#071226);border-radius:14px 14px 6px 14px}
.bubble.assistant{background:linear-gradient(180deg,#061426,#071427);border-radius:14px 14px 14px 6px;border:1px solid rgba(255,255,255,0.02)}
.meta{font-size:12px;color:var(--muted);margin-bottom:6px}
.sidebar .section{margin-bottom:12px}
.input-sm{padding:6px;border-radius:6px;border:1px solid rgba(255,255,255,0.03);background:rgba(255,255,255,0.01);color:var(--text)}
.file-list,.memory-list{max-height:180px;overflow:auto;padding:6px;border-radius:6px;background:rgba(255,255,255,0.01);border:1px solid rgba(255,255,255,0.02)}
.kv{font-size:13px;color:var(--muted);margin-top:6px}
.footer{margin-top:12px;font-size:12px;color:var(--muted);text-align:center}
#debug{height:160px;overflow:auto;background:#020812;border-radius:8px;padding:8px;border:1px solid rgba(255,255,255,0.02);font-family:var(--mono);font-size:12px;color:#9aa6bf}
.progressOuter{width:100%;background:#061426;padding:6px;border-radius:8px;border:1px solid rgba(255,255,255,0.02)}
.progressInner{height:12px;background:linear-gradient(90deg,var(--accent),var(--accent2));width:0%;border-radius:6px;transition:width 0.2s ease}
.weight-heat{display:grid;grid-template-columns:repeat(8,1fr);gap:4px}
.weight-cell{height:20px;border-radius:4px;background:#0e1624;text-align:center;font-size:10px;color:var(--muted);display:flex;align-items:center;justify-content:center}
.markdown{color:var(--text)}
.note{font-size:12px;color:var(--muted);margin-top:6px}
hr{border:none;border-top:1px solid rgba(255,255,255,0.02);margin:12px 0}
</style>

<!-- External libs loaded lazily when needed -->
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/tesseract.js@4.0.2/dist/tesseract.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/pdfjs-dist@3.8.162/build/pdf.min.js"></script>

</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo">J</div>
      <div class="title">
        <h1>Jack.AI — Super Full (Client)</h1>
        <div class="subtitle">Local-first • OCR • PDF • TTS/STT • Simulated fine-tune • Client models opt-in</div>
      </div>
    </div>

    <div class="layout">
      <!-- MAIN -->
      <div class="card chat-area" aria-live="polite">
        <div class="chat-log" id="chatLog" role="log" aria-atomic="false"></div>

        <div class="input-row" role="region" aria-label="Input area">
          <textarea id="userInput" aria-label="Message input" placeholder="Ask Jack something... (Shift+Enter = newline)"></textarea>

          <div style="display:flex;flex-direction:column;gap:8px">
            <button id="sendBtn" class="btn" title="Send message">Send</button>
            <button id="streamBtn" class="btn alt" title="Stream (simulate typing)">Stream</button>
            <button id="stopBtn" class="btn alt" style="display:none" title="Stop generation">Stop</button>
          </div>
        </div>

        <div class="controls">
          <button id="attachBtn" class="icon-btn" title="Attach file">📎</button>
          <input id="fileInput" type="file" accept="image/*,.txt,.md,.pdf" style="display:none">
          <button id="micBtn" class="icon-btn" title="Voice input">🎙</button>
          <button id="ttsToggle" class="icon-btn" title="Toggle TTS">🗣</button>

          <div style="flex:1"></div>

          <select id="modeSelect" class="input-sm" aria-label="Mode select">
            <option value="local">Local (no heavy models)</option>
            <option value="distilgpt2">distilgpt2 (Xenova)</option>
            <option value="gpt2">gpt2 (Xenova)</option>
            <option value="image-caption">image-caption (vit-gpt2)</option>
          </select>

          <button id="modelToggle" class="btn alt">Load / Unload Model</button>
          <label class="small" style="margin-left:8px"><input id="asiToggle" type="checkbox"> ASI (opt-in)</label>
        </div>

        <div style="margin-top:10px;display:flex;gap:8px;align-items:center">
          <div class="small">Tone:</div>
          <select id="toneSelect" class="input-sm" style="width:160px">
            <option value="balanced">Balanced</option>
            <option value="creative">Creative</option>
            <option value="concise">Concise</option>
            <option value="sarcastic">Sarcastic</option>
          </select>

          <div style="flex:1"></div>
          <div class="small">Status:</div>
          <div id="status" class="kv">Ready</div>
        </div>

        <div style="margin-top:12px;">
          <div class="small">Debug / Console</div>
          <div id="debug" aria-live="polite"></div>
        </div>
      </div>

      <!-- SIDEBAR -->
      <div class="card sidebar" role="complementary">
        <div class="section">
          <strong>Persona / Memory</strong>
          <div class="small">Customize persona and save memories/examples</div>
          <div style="margin-top:8px">
            <input id="personaName" class="input-sm" placeholder="Assistant name (Jack)" />
            <textarea id="personaPrompt" rows="3" style="margin-top:8px" class="input-sm" placeholder="System prompt / personality"></textarea>
            <div style="display:flex;gap:8px;margin-top:8px">
              <button id="savePersona" class="btn alt">Save Persona</button>
              <button id="evolvePersona" class="btn alt">Evolve Persona</button>
            </div>
          </div>
        </div>

        <div class="section">
          <strong>Files, OCR & PDF</strong>
          <div class="small">Upload images for OCR, PDFs for text extraction</div>
          <div style="margin-top:8px" id="fileArea"></div>
          <div style="display:flex;gap:6px;margin-top:8px">
            <button id="runOcr" class="btn alt">Run OCR</button>
            <button id="extractPdf" class="btn alt">Extract PDF</button>
          </div>
          <textarea id="extractedText" rows="6" style="margin-top:8px;display:none" class="input-sm"></textarea>
          <div class="note">OCR runs using Tesseract.js in the browser. PDF extraction uses pdf.js. Some large PDFs may be slow.</div>
        </div>

        <div class="section">
          <strong>Images & Captions</strong>
          <div class="small">Gallery and optional captioning via model</div>
          <div id="gallery" class="file-list" style="margin-top:8px"></div>
          <div style="display:flex;gap:6px;margin-top:8px">
            <button id="captionSelected" class="btn alt">Caption Selected</button>
            <button id="saveCaptionMem" class="btn alt">Save Caption to Memory</button>
          </div>
        </div>

        <div class="section">
          <strong>Memories & Examples</strong>
          <div class="small">Saved items and training examples</div>
          <div id="memoryList" class="memory-list"></div>
          <div style="display:flex;gap:6px;margin-top:8px">
            <button id="saveLastMem" class="btn alt">Save Last Reply</button>
            <button id="addExample" class="btn alt">Add Example</button>
          </div>
          <div style="display:flex;gap:6px;margin-top:8px">
            <button id="exportData" class="btn alt">Export JSON</button>
            <button id="importData" class="btn alt">Import JSON</button>
          </div>
        </div>

        <div class="section">
          <strong>Tiny Predictor & Training</strong>
          <div class="small">TF-IDF + linear predictor trained on examples</div>
          <div style="display:flex;gap:6px;margin-top:8px">
            <button id="initTiny" class="btn alt">Init Tiny</button>
            <button id="trainTiny" class="btn alt">Train Tiny</button>
          </div>
          <div style="display:flex;gap:6px;margin-top:8px">
            <label class="small">LR <input id="tinyLR" type="number" value="0.05" step="0.01" class="input-sm" style="width:100px;margin-left:6px" /></label>
          </div>
          <div id="tinyInfo" class="kv" style="margin-top:8px">Tiny: not initialized</div>
          <div class="progressOuter" style="margin-top:8px"><div id="tinyProgress" class="progressInner"></div></div>
          <div id="weightGrid" style="margin-top:8px"></div>
        </div>

        <div class="section">
          <strong>Tools</strong>
          <div style="display:flex;gap:6px;margin-top:8px">
            <button id="summarizeMem" class="btn alt">Summarize Memories</button>
            <button id="sentimentMem" class="btn alt">Sentiment</button>
          </div>
        </div>

        <div class="section">
          <strong>Safety & Data</strong>
          <div class="small">ASI is opt-in. Wipe local storage to remove all local data.</div>
          <button id="wipeAll" class="btn danger" style="margin-top:8px">Wipe All Local Data</button>
        </div>

        <div class="section">
          <strong>Help & Shortcuts</strong>
          <div class="small">Shortcuts: Enter=Send, Shift+Enter=Newline</div>
        </div>

      </div>
    </div>
  </div>

<script>
/*
  Jack.AI — Client single-file app
  Notes:
  - All data saved in localStorage (keys prefixed with 'jack_super_')
  - Model loading is optional and device-dependent. This script only attempts to load transformers.js on demand.
  - Simulated fine-tune is provided for visualization.
*/

/* -------------------------
   Storage keys & helpers
   ------------------------- */
const KEY_CHAT = 'jack_super_chat_v1';
const KEY_MEM = 'jack_super_mem_v1';
const KEY_EX = 'jack_super_examples_v1';
const KEY_TINY = 'jack_super_tiny_v1';
const KEY_PERSONA = 'jack_super_persona_v1';

function nowISO(){ return new Date().toISOString(); }
function debugLog(msg){ const d = document.getElementById('debug'); d.textContent = `${(new Date()).toLocaleTimeString()} - ${msg}\n` + d.textContent; }
function saveJSON(key, obj){ localStorage.setItem(key, JSON.stringify(obj)); }
function loadJSON(key, fallback){ try{ return JSON.parse(localStorage.getItem(key) || JSON.stringify(fallback)); } catch(e){ return fallback; } }

/* -------------------------
   App state
   ------------------------- */
let chat = loadJSON(KEY_CHAT, []);
let memories = loadJSON(KEY_MEM, []);
let examples = loadJSON(KEY_EX, []);
let tinyModel = loadJSON(KEY_TINY, null);
let persona = loadJSON(KEY_PERSONA, { name: 'Jack', system_prompt: 'You are Jack, a witty assistant.', politeness:1.0 });

let transformersLoaded = false;
let pipelines = {}; // pipelines cache
let galleryItems = []; // local gallery: {id,name,dataUrl,caption}
let lastPdfFile = null;

/* -------------------------
   DOM refs
   ------------------------- */
const chatLog = document.getElementById('chatLog');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const streamBtn = document.getElementById('streamBtn');
const stopBtn = document.getElementById('stopBtn');
const attachBtn = document.getElementById('attachBtn');
const fileInput = document.getElementById('fileInput');
const micBtn = document.getElementById('micBtn');
const ttsToggle = document.getElementById('ttsToggle');
const modelToggle = document.getElementById('modelToggle');
const modeSelect = document.getElementById('modeSelect');
const statusEl = document.getElementById('status');

const personaNameEl = document.getElementById('personaName');
const personaPromptEl = document.getElementById('personaPrompt');
const savePersonaBtn = document.getElementById('savePersona');
const evolvePersonaBtn = document.getElementById('evolvePersona');

const fileArea = document.getElementById('fileArea');
const runOcrBtn = document.getElementById('runOcr');
const extractPdfBtn = document.getElementById('extractPdf');
const extractedText = document.getElementById('extractedText');

const galleryEl = document.getElementById('gallery');
const captionSelectedBtn = document.getElementById('captionSelected');
const saveCaptionMemBtn = document.getElementById('saveCaptionMem');

const memoryListEl = document.getElementById('memoryList');
const saveLastMemBtn = document.getElementById('saveLastMem');
const addExampleBtn = document.getElementById('addExample');
const exportDataBtn = document.getElementById('exportData');
const importDataBtn = document.getElementById('importData');

const initTinyBtn = document.getElementById('initTiny');
const trainTinyBtn = document.getElementById('trainTiny');
const tinyLR = document.getElementById('tinyLR');
const tinyInfo = document.getElementById('tinyInfo');
const tinyProgress = document.getElementById('tinyProgress');
const weightGrid = document.getElementById('weightGrid');

const summarizeMemBtn = document.getElementById('summarizeMem');
const sentimentMemBtn = document.getElementById('sentimentMem');

const wipeAllBtn = document.getElementById('wipeAll');

const toneSelect = document.getElementById('toneSelect');
const asiToggle = document.getElementById('asiToggle');

/* -------------------------
   Utilities: escape, render
   ------------------------- */
function escapeHtml(s){ if(!s) return ''; return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function persistAll(){
  saveJSON(KEY_CHAT, chat);
  saveJSON(KEY_MEM, memories);
  saveJSON(KEY_EX, examples);
  saveJSON(KEY_TINY, tinyModel);
  saveJSON(KEY_PERSONA, persona);
  debugLog('State persisted');
}

function renderChat(){
  chatLog.innerHTML = '';
  for(const m of chat.slice(-500)){
    const wrapper = document.createElement('div');
    wrapper.className = 'msg ' + (m.role === 'user' ? 'user' : 'assistant');
    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + (m.role === 'user' ? 'user' : 'assistant');
    const who = m.role === 'user' ? 'You' : (persona.name || 'Jack');
    const meta = `<div class="meta"><strong>${escapeHtml(who)}</strong> <span class="small" style="margin-left:8px">${escapeHtml(m.time || '')}</span></div>`;
    if(m.role === 'assistant'){
      bubble.innerHTML = meta + `<div class="markdown">${marked.parse(m.text || '')}</div>`;
    } else {
      bubble.innerHTML = meta + `<div style="white-space:pre-wrap;">${escapeHtml(m.text || '')}</div>`;
    }
    wrapper.appendChild(bubble);
    chatLog.appendChild(wrapper);
  }
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderMemories(){
  memoryListEl.innerHTML = '';
  for(const mem of memories.slice(-200).reverse()){
    const d = document.createElement('div');
    d.style.padding = '6px 0';
    d.innerHTML = `<strong>${escapeHtml(mem.key)}</strong><div class="small" style="margin-top:4px">${escapeHtml((mem.value||'').slice(0,200))}</div>`;
    memoryListEl.appendChild(d);
  }
}

function renderFiles(){
  fileArea.innerHTML = '';
  if(lastPdfFile){
    const div = document.createElement('div');
    div.className = 'small';
    div.textContent = `PDF queued: ${lastPdfFile.name}`;
    fileArea.appendChild(div);
  }
}

/* -------------------------
   Local assistant heuristics
   ------------------------- */
function localHeuristicReply(userText){
  const tone = toneSelect.value || 'balanced';
  const low = (userText||'').toLowerCase();
  if(low.startsWith('/persona')){
    const arg = userText.slice('/persona'.length).trim();
    if(arg){
      persona.name = arg;
      persistAll();
      return `Persona updated to ${arg}.`;
    } else {
      return `Persona: ${persona.name || 'Jack'}`;
    }
  }
  if(low === '/clear'){
    chat = [];
    persistAll();
    renderChat();
    return 'Chat cleared.';
  }
  if(low.startsWith('/learn')){
    const content = userText.slice('/learn'.length).trim();
    if(content){
      memories.push({key:'learned_'+nowISO(), value:content, created_at:nowISO(), weight:1.0});
      persistAll();
      renderMemories();
      return 'Learned and saved to memory.';
    }
    return 'Supply content after /learn to save it.';
  }
  if(low.startsWith('http://') || low.startsWith('https://')){
    // try to fetch and summarize (CORS may block)
    return `Attempting to analyze link: ${userText}`;
  }
  if(low.includes('time')) return `Current time: ${new Date().toLocaleString()}`;
  if(low.includes('joke')) return 'Why do programmers prefer dark mode? Because light attracts bugs.';
  if(low.includes('summary')){
    const text = memories.map(m=>m.value).join('\\n');
    const s = simpleSummarize(text, 3);
    return s ? `Summary: ${s}` : 'No memory content to summarize.';
  }
  const replies = [
    'Probably fix X and Y. Or set it on fire.',
    'Here is a blunt take: check your settings, then breathe.',
    'Looks messy. Might be intentional.'
  ];
  const idx = Math.floor(Math.random()*replies.length);
  return replies[idx] + '\\n\\nContext: ' + userText.slice(0,300);
}

/* -------------------------
   TF-IDF tiny predictor + tiny model (client-side)
   ------------------------- */
function tokenize(text){
  return (text||'').toLowerCase().replace(/[^a-z0-9\\s]/g,' ').split(/\\s+/).filter(Boolean);
}

function buildVocab(docs){
  const map = {};
  let i=0;
  for(const d of docs){
    for(const t of tokenize(d)){
      if(!(t in map)){ map[t]=i++; }
    }
  }
  return map;
}

function computeIdf(docs, vocab){
  const N = docs.length;
  const df = new Array(Object.keys(vocab).length).fill(0);
  for(const d of docs){
    const seen = new Set();
    for(const t of tokenize(d)){
      if(t in vocab && !seen.has(t)){ df[vocab[t]] += 1; seen.add(t); }
    }
  }
  return df.map(x => Math.log((1+N)/(1+x))+1);
}

function toTfidf(text, vocab, idf){
  const vec = new Array(Object.keys(vocab).length).fill(0);
  const toks = tokenize(text);
  if(toks.length === 0) return vec;
  const tf = {};
  for(const t of toks) if(t in vocab) tf[vocab[t]] = (tf[vocab[t]]||0)+1;
  for(const k in tf) vec[k] = (tf[k]/toks.length) * (idf[k] || 1);
  const norm = Math.sqrt(vec.reduce((a,b)=>a+b*b,0)) || 1;
  return vec.map(v=>v/norm);
}

function buildVectorStore(){
  const docs = memories.map(m=>m.value).filter(Boolean);
  if(docs.length===0) return {vocab:{}, idf:[], embeddings:[], docs:[]};
  const vocab = buildVocab(docs);
  const idf = computeIdf(docs, vocab);
  const embeddings = docs.map(d=>toTfidf(d, vocab, idf));
  return {vocab, idf, embeddings, docs};
}

function initTiny(){
  const store = buildVectorStore();
  const dim = Object.keys(store.vocab).length;
  if(dim===0){
    tinyModel = null;
    tinyInfo.textContent = 'Tiny: no data to build vocab';
    persistAll();
    return;
  }
  const W = [];
  for(let i=0;i<dim;i++){
    const row = [];
    for(let j=0;j<dim;j++) row.push((Math.random()-0.5)*0.1);
    W.push(row);
  }
  const b = new Array(dim).fill(0);
  tinyModel = {W,b,inputDim:dim, vocab:store.vocab, idf:store.idf, lr: parseFloat(tinyLR.value||'0.05')||0.05};
  tinyInfo.textContent = `Tiny: initialized dim=${dim}`;
  persistAll();
  renderWeights();
  debugLog('Tiny initialized dim='+dim);
}

function tinyPredict(x){
  if(!tinyModel) return null;
  const out = new Array(tinyModel.inputDim).fill(0);
  for(let i=0;i<tinyModel.inputDim;i++){
    let s = 0;
    for(let j=0;j<tinyModel.inputDim;j++) s += (tinyModel.W[i][j]||0) * (x[j]||0);
    s += tinyModel.b[i]||0;
    out[i]=s;
  }
  const norm=Math.sqrt(out.reduce((a,b)=>a+b*b,0))||1;
  return out.map(v=>v/norm);
}

function tinyTrainStep(x,y){
  if(!tinyModel) return;
  const pred = tinyPredict(x);
  const lr = tinyModel.lr || 0.05;
  for(let i=0;i<tinyModel.inputDim;i++){
    const err = (pred[i]||0) - (y[i]||0);
    for(let j=0;j<tinyModel.inputDim;j++){
      tinyModel.W[i][j] -= lr * err * (x[j]||0);
    }
    tinyModel.b[i] -= lr * err;
  }
}

function trainTinyOnePass(simCallback, delay=0){
  const store = buildVectorStore();
  const dim = Object.keys(store.vocab).length;
  if(dim===0){ alert('No data to train on. Save memories/examples first.'); return; }
  if(!tinyModel || tinyModel.inputDim !== dim) initTiny();
  const pairs = [];
  for(const ex of examples){
    if(ex.user && ex.assistant) pairs.push({x: toTfidf(ex.user, store.vocab, store.idf), y: toTfidf(ex.assistant, store.vocab, store.idf)});
  }
  for(const mem of memories) {
    if(mem.value) pairs.push({x: toTfidf(mem.value, store.vocab, store.idf), y: toTfidf(mem.value, store.vocab, store.idf)});
  }
  if(pairs.length===0){ alert('No pairs to train on.'); return; }
  let step = 0;
  const total = pairs.length;
  tinyProgress.style.width = '0%';
  const doStep = (i) => {
    const p = pairs[i];
    tinyTrainStep(p.x, p.y);
    step++;
    const pct = Math.round(step/total*100);
    tinyProgress.style.width = pct+'%';
    if(simCallback) simCallback(step,total);
  };
  // synchronous loop with optional micro-delay to keep UI responsive
  let i=0;
  function loop(){
    const chunk = Math.min(4, pairs.length - i);
    for(let k=0;k<chunk;k++) doStep(i++);
    renderWeights();
    if(i < pairs.length){
      setTimeout(loop, delay);
    } else {
      tinyInfo.textContent = `Tiny: trained on ${pairs.length} pairs.`;
      persistAll();
      debugLog(`Tiny trained on ${pairs.length} pairs`);
    }
  }
  loop();
}

/* Show small heat grid of weights */
function renderWeights(){
  weightGrid.innerHTML = '';
  if(!tinyModel){ weightGrid.textContent = 'No tiny model.'; return; }
  const dim = tinyModel.inputDim;
  const sample = Math.min(8, dim);
  const grid = document.createElement('div');
  grid.className = 'weight-heat';
  // show small sample of W[0..sample-1][0..sample-1]
  for(let i=0;i<sample*sample;i++){
    const cell = document.createElement('div');
    cell.className = 'weight-cell';
    const r = Math.floor(i/sample);
    const c = i % sample;
    const val = tinyModel.W[r] && tinyModel.W[r][c] ? tinyModel.W[r][c] : 0;
    // color shading via inline style
    const shade = Math.min(1, Math.abs(val)/0.2);
    const hue = val >= 0 ? 200 : 340;
    cell.style.background = `linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.2))`;
    cell.style.border = `1px solid rgba(255,255,255,0.02)`;
    cell.textContent = val.toFixed(3);
    grid.appendChild(cell);
  }
  weightGrid.appendChild(grid);
}

/* -------------------------
   Summarize helper
   ------------------------- */
function simpleSummarize(text, maxSentences=3){
  if(!text) return '';
  const sentences = text.split(/(?<=[\\.\\?!])\\s+/).filter(Boolean);
  if(sentences.length <= maxSentences) return sentences.join(' ');
  const vocab = buildVocab(sentences);
  if(Object.keys(vocab).length === 0) return sentences.slice(0,maxSentences).join(' ');
  const idf = computeIdf(sentences, vocab);
  const scores = sentences.map(s => ({s,score: toTfidf(s,vocab,idf).reduce((a,b)=>a+b,0)}));
  scores.sort((a,b)=>b.score-a.score);
  const top = scores.slice(0,maxSentences).map(x=>x.s);
  return top.join(' ');
}

/* -------------------------
   Transformers dynamic load (browser) (Xenova)
   ------------------------- */
async function ensureTransformers(){
  if(transformersLoaded) return true;
  try{
    if(window.transformers) { transformersLoaded=true; return true; }
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.5.0/dist/transformers.min.js';
    document.head.appendChild(s);
    await new Promise((res,rej)=>{ s.onload=()=>res(true); s.onerror=()=>rej(new Error('Failed to load transformers.js')); });
    transformersLoaded=true;
    debugLog('transformers.js loaded');
    return true;
  } catch(err){
    debugLog('transformers load failed: ' + String(err));
    return false;
  }
}

async function getPipeline(task, modelName){
  const key = task + '|' + modelName;
  if(pipelines[key]) return pipelines[key];
  const ok = await ensureTransformers();
  if(!ok) throw new Error('transformers not available');
  statusEl.textContent = `Initializing ${task} (${modelName})`;
  try{
    const pipe = await window.transformers.pipeline(task, modelName);
    pipelines[key] = pipe;
    statusEl.textContent = `${task} ready: ${modelName}`;
    debugLog('Pipeline ready: ' + key);
    return pipe;
  } catch(err){
    debugLog('Pipeline creation error: ' + String(err));
    statusEl.textContent = 'Pipeline error';
    throw err;
  }
}

/* -------------------------
   Core generation flow (tries tiny predictor -> model -> local)
   ------------------------- */
let abortFlag = { abort:false };

async function generateReply(userText, stream=false){
  // push user
  chat.push({role:'user', text:userText, time: nowISO()});
  persistAll();
  renderChat();

  // assistant placeholder
  chat.push({role:'assistant', text:'Thinking...', time: nowISO()});
  persistAll();
  renderChat();

  abortFlag.abort = false;
  stopBtn.style.display = 'inline-block';

  // tiny predictor attempt
  try{
    const store = buildVectorStore();
    if(store.docs.length>0 && tinyModel && tinyModel.inputDim === Object.keys(store.vocab).length){
      const userVec = toTfidf(userText, store.vocab, store.idf);
      const pred = tinyPredict(userVec);
      let best=-1, bestSim=-1;
      for(let i=0;i<store.embeddings.length;i++){
        const s = cosine(pred, store.embeddings[i]);
        if(s>bestSim){ bestSim=s; best=i; }
      }
      if(best>=0 && bestSim>0.3){
        const reply = applyPersona(store.docs[best] + ` (tiny sim=${bestSim.toFixed(2)})`);
        replaceLastAssistant(reply);
        persistAll();
        stopBtn.style.display = 'none';
        return reply;
      }
    }
  } catch(e){ debugLog('Tiny predictor error: ' + String(e)); }

  // If a model is selected
  const mode = modeSelect.value || 'local';
  if(mode !== 'local'){
    try{
      // image caption special case
      if(mode === 'image-caption' && galleryItems.length>0){
        const it = galleryItems[0];
        const pipe = await getPipeline('image-to-text', 'Xenova/vit-gpt2-image-captioning');
        const out = await pipe(it.dataUrl);
        const caption = Array.isArray(out) && out[0] && out[0].generated_text ? out[0].generated_text : String(out||'');
        const reply = applyPersona('Image caption: ' + caption);
        replaceLastAssistant(reply);
        memories.push({key:'image_caption_'+nowISO(), value:caption, created_at: nowISO()});
        persistAll(); renderMemories();
        stopBtn.style.display = 'none';
        return reply;
      }

      const modelName = (mode === 'distilgpt2' ? 'Xenova/distilgpt2' : (mode === 'gpt2' ? 'Xenova/gpt2' : mode));
      const pipe = await getPipeline('text-generation', modelName);
      const personaLine = `You are ${persona.name || 'Jack'}. ${persona.system_prompt || ''}`;
      const asiNote = (asiToggle.checked ? 'Please provide detailed reasoning.' : '');
      const memoryContext = (memories.slice(-6).map(m=>m.value).join('\\n\\n')) || '';
      let prompt = personaLine + '\\n' + asiNote + '\\n' + memoryContext + '\\nUser: ' + userText + '\\nAssistant:';
      statusEl.textContent = 'Generating...';
      const opts = { max_new_tokens: (asiToggle.checked ? 200 : 100), do_sample:true, temperature: asiToggle.checked ? 0.9 : 0.7 };
      const out = await pipe(prompt, opts);
      let raw = '';
      if(Array.isArray(out) && out[0] && out[0].generated_text) raw = out[0].generated_text;
      else if(out && out.generated_text) raw = out.generated_text;
      else raw = String(out||'');
      if(raw.startsWith(prompt)) raw = raw.slice(prompt.length).trim();
      const reply = applyPersona(raw);
      replaceLastAssistant(reply);
      if(asiToggle.checked) memories.push({key:'interaction_'+nowISO(), value:reply, created_at:nowISO()});
      persistAll(); renderMemories();
      statusEl.textContent = 'Ready';
      stopBtn.style.display = 'none';
      return reply;
    } catch(err){
      debugLog('Model generation error: ' + String(err));
      statusEl.textContent = 'Model error, falling back';
    }
  }

  // fallback
  const fallback = localHeuristicReply(userText);
  replaceLastAssistant(applyPersona(fallback));
  persistAll();
  stopBtn.style.display = 'none';
  return fallback;
}

/* cosine similarity */
function cosine(a,b){
  if(!a || !b) return 0;
  let num=0, na=0, nb=0;
  for(let i=0;i<a.length;i++){ num += (a[i]||0)*(b[i]||0); na += (a[i]||0)*(a[i]||0); nb += (b[i]||0)*(b[i]||0); }
  if(na===0 || nb===0) return 0;
  return num / (Math.sqrt(na)*Math.sqrt(nb));
}

function replaceLastAssistant(text){
  for(let i=chat.length-1;i>=0;i--){
    if(chat[i].role === 'assistant'){ chat[i].text = text; chat[i].time = nowISO(); break; }
  }
  persistAll();
  renderChat();
}

function applyPersona(text){
  let out = text;
  const tone = toneSelect.value || 'balanced';
  if(tone === 'creative') out = '✨ ' + out;
  if(tone === 'sarcastic') out = out + ' (sarcastic)';
  if(tone === 'concise') out = out.split('.').slice(0,1).join('.') + '.';
  if(persona.politeness && persona.politeness < 0.7) out = out.replace(/please/ig,'');
  return out;
}

/* -------------------------
   File, OCR, PDF, Gallery
   ------------------------- */
fileInput.addEventListener('change', (ev) => {
  const f = ev.target.files[0];
  if(!f) return;
  const reader = new FileReader();
  if(f.type && f.type.indexOf('image') === 0){
    reader.onload = (e) => {
      const url = e.target.result;
      const id = 'img_' + Date.now();
      galleryItems.unshift({id, name:f.name, dataUrl:url, caption:''});
      renderGallery();
      fileArea.innerHTML = `<div class="small"><strong>${escapeHtml(f.name)}</strong> (image)</div>`;
    };
    reader.readAsDataURL(f);
  } else if(f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')){
    lastPdfFile = f;
    fileArea.innerHTML = `<div class="small">PDF queued: ${escapeHtml(f.name)}</div>`;
    extractedText.style.display = 'none';
    runOcrBtn.style.display = 'none';
    extractPdfBtn.style.display = 'inline-block';
  } else {
    reader.onload = (e) => {
      fileArea.innerHTML = `<pre style="white-space:pre-wrap;max-height:200px;overflow:auto">${escapeHtml((e.target.result||'').slice(0,2000))}</pre>`;
    };
    reader.readAsText(f);
  }
  fileInput.value = '';
});

function renderGallery(){
  galleryEl.innerHTML = '';
  galleryItems.forEach((it, idx) => {
    const d = document.createElement('div');
    d.style.display='flex'; d.style.gap='8px'; d.style.alignItems='center'; d.style.padding='6px 0';
    d.innerHTML = `<img src="${it.dataUrl}" alt="${escapeHtml(it.name)}" style="max-width:80px;border-radius:6px"/><div style="flex:1"><div style="font-weight:700">${escapeHtml(it.name)}</div><div class="small">${escapeHtml((it.caption||'').slice(0,120))}</div></div>`;
    d.onclick = () => {
      const els = galleryEl.querySelectorAll('div');
      els.forEach(e => e.style.background='transparent');
      d.style.background = 'rgba(255,255,255,0.01)';
      galleryEl.dataset.selectedIndex = idx;
    };
    galleryEl.appendChild(d);
  });
}

runOcrBtn.addEventListener('click', async () => {
  if(galleryItems.length===0){ alert('No image to OCR'); return; }
  const img = galleryItems[0];
  extractedText.style.display = 'block';
  extractedText.value = 'Running OCR...';
  try{
    const worker = Tesseract.createWorker();
    await worker.load();
    await worker.loadLanguage('eng');
    await worker.initialize('eng');
    const res = await worker.recognize(img.dataUrl);
    await worker.terminate();
    extractedText.value = res.data.text || '';
    memories.push({key:'ocr_'+nowISO(), value:res.data.text||'', created_at:nowISO()});
    persistAll(); renderMemories();
    alert('OCR complete and saved to memories.');
  } catch(err){
    debugLog('OCR error: ' + String(err));
    alert('OCR failed. See debug for details.');
  }
});

extractPdfBtn.addEventListener('click', async () => {
  if(!lastPdfFile){ alert('No PDF loaded.'); return; }
  extractedText.style.display='block';
  extractedText.value = 'Extracting PDF...';
  try{
    const arrayBuffer = await lastPdfFile.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({data: arrayBuffer}).promise;
    let combined = '';
    for(let p=1;p<=pdf.numPages;p++){
      const page = await pdf.getPage(p);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map(i=>i.str).join(' ');
      combined += pageText + '\\n\\n';
    }
    extractedText.value = combined;
    memories.push({key:'pdf_'+nowISO(), value:combined, created_at: nowISO()});
    persistAll(); renderMemories();
    alert('PDF extracted and saved to memories.');
  } catch(e){
    debugLog('PDF extract error: ' + String(e));
    alert('PDF extraction failed. See debug.');
  }
});

captionSelectedBtn.addEventListener('click', async () => {
  const idx = parseInt(galleryEl.dataset.selectedIndex || '-1', 10);
  if(idx < 0 || !galleryItems[idx]){ alert('Select an image from gallery first'); return; }
  const it = galleryItems[idx];
  try{
    const pipe = await getPipeline('image-to-text', 'Xenova/vit-gpt2-image-captioning');
    const out = await pipe(it.dataUrl);
    let caption = '';
    if(Array.isArray(out) && out[0] && out[0].generated_text) caption = out[0].generated_text;
    else caption = String(out || '');
    it.caption = caption;
    renderGallery();
    alert('Caption generated: ' + caption);
  } catch(e){
    debugLog('Caption error: ' + String(e));
    alert('Captioning failed. See debug.');
  }
});

saveCaptionMemBtn.addEventListener('click', () => {
  const idx = parseInt(galleryEl.dataset.selectedIndex || '-1', 10);
  if(idx < 0 || !galleryItems[idx]){ alert('Select an image first'); return; }
  const it = galleryItems[idx];
  if(!it.caption){ alert('No caption'); return; }
  memories.push({key:'image_caption_'+nowISO(), value:it.caption, created_at:nowISO()});
  persistAll(); renderMemories();
  alert('Caption saved to memories');
});

/* -------------------------
   Persona / examples / memory controls
   ------------------------- */
savePersonaBtn.addEventListener('click', () => {
  persona.name = personaNameEl.value || persona.name || 'Jack';
  persona.system_prompt = personaPromptEl.value || persona.system_prompt || '';
  persistAll();
  alert('Persona saved');
});

evolvePersonaBtn.addEventListener('click', () => {
  // simple heuristic to "evolve" persona from examples
  let polite=0, creative=0;
  for(const ex of examples.slice(-200)){
    const t = (ex.assistant||'').toLowerCase();
    if(t.includes('please') || t.includes('thank')) polite++;
    if(t.includes('imagine') || t.includes('creative')) creative++;
  }
  const total = Math.max(1, Math.min(examples.length, 200));
  persona.politeness = (persona.politeness||1)*0.7 + (polite/total)*0.3;
  if(creative/total > 0.2) persona.tone = 'creative';
  persistAll();
  alert('Persona evolved from examples');
});

saveLastMemBtn.addEventListener('click', () => {
  const last = chat.slice().reverse().find(c => c.role === 'assistant');
  if(!last){ alert('No assistant reply to save'); return; }
  memories.push({key:'assistant_'+nowISO(), value:last.text, created_at: nowISO()});
  persistAll(); renderMemories();
  alert('Saved last assistant reply to memories');
});

addExampleBtn.addEventListener('click', () => {
  for(let i=chat.length-1;i>0;i--){
    if(chat[i].role==='assistant' && chat[i-1] && chat[i-1].role==='user'){
      examples.push({user:chat[i-1].text, assistant: chat[i].text, ts: nowISO()});
      persistAll(); renderMemories();
      alert('Example added');
      return;
    }
  }
  alert('No recent user->assistant pair found');
});

/* Export / Import */
exportDataBtn.addEventListener('click', () => {
  const payload = {chat, memories, examples, persona, tinyModel, galleryItems};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'jack_export.json'; a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href), 1500);
});

importDataBtn.addEventListener('click', () => {
  const inp = document.createElement('input'); inp.type='file'; inp.accept='application/json';
  inp.onchange = (e) => {
    const f = e.target.files[0]; if(!f) return;
    const r = new FileReader();
    r.onload = () => {
      try{
        const data = JSON.parse(r.result);
        if(data.chat) chat = data.chat;
        if(data.memories) memories = data.memories;
        if(data.examples) examples = data.examples;
        if(data.persona) persona = data.persona;
        if(data.tinyModel) tinyModel = data.tinyModel;
        if(data.galleryItems) galleryItems = data.galleryItems;
        persistAll(); renderAll();
        alert('Import complete');
      }catch(err){
        alert('Import failed: ' + err);
      }
    };
    r.readAsText(f);
  };
  inp.click();
});

/* -------------------------
   Tiny predictor UI controls
   ------------------------- */
document.getElementById('initTiny').addEventListener('click', () => initTiny());
document.getElementById('trainTiny').addEventListener('click', () => trainTinyOnePass((step,total)=>{ /*progress handled inside*/ }, 5));

/* -------------------------
   Tools: summarize, sentiment (simulated via transformers when available)
   ------------------------- */
document.getElementById('summarizeMem').addEventListener('click', () => {
  const text = memories.map(m=>m.value).join('\\n');
  const s = simpleSummarize(text, 4) || 'No memories';
  alert('Summary:\\n\\n' + s);
});

document.getElementById('sentimentMem').addEventListener('click', async () => {
  try{
    await ensureTransformers();
    const pipe = pipelines['sentiment|Xenova/distilbert-base-uncased-finetuned-sst-2-english'] || await getPipeline('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
    const textSample = memories.map(m=>m.value).join('\\n').slice(0,1200);
    if(!textSample){ alert('No memory content'); return; }
    const res = await pipe(textSample);
    alert('Sentiment sample: ' + JSON.stringify(res[0]||res));
  } catch(err){
    debugLog('Sentiment analysis failed: ' + err);
    alert('Sentiment analysis failed; check debug.');
  }
});

/* -------------------------
   Wipe all
   ------------------------- */
document.getElementById('wipeAll').addEventListener('click', () => {
  if(!confirm('Wipe ALL local data? This cannot be undone.')) return;
  localStorage.removeItem(KEY_CHAT); localStorage.removeItem(KEY_MEM); localStorage.removeItem(KEY_EX);
  localStorage.removeItem(KEY_TINY); localStorage.removeItem(KEY_PERSONA);
  chat=[]; memories=[]; examples=[]; tinyModel=null; persona={name:'Jack', system_prompt:'You are Jack...'}; galleryItems=[];
  persistAll(); renderAll();
  alert('All local data wiped');
});

/* -------------------------
   Model load/unload
   ------------------------- */
modelToggle.addEventListener('click', async () => {
  if(modeSelect.value === 'local'){ alert('Choose a non-local model from dropdown to load'); return; }
  const key = 'text-generation|' + modeSelect.value;
  if(pipelines[key]){ delete pipelines[key]; statusEl.textContent='Model unloaded'; debugLog('Unloaded '+modeSelect.value); return; }
  try{
    await getPipeline('text-generation', modeSelect.value === 'distilgpt2' ? 'Xenova/distilgpt2' : (modeSelect.value === 'gpt2' ? 'Xenova/gpt2' : modeSelect.value));
    statusEl.textContent = 'Model loaded: ' + modeSelect.value;
  } catch(err){
    debugLog('Model load failed: ' + err);
    alert('Model load failed. See debug.');
  }
});

/* -------------------------
   TTS & STT
   ------------------------- */
ttsToggle.addEventListener('click', () => {
  const on = ttsToggle.classList.toggle('active');
  ttsToggle.style.background = on ? 'linear-gradient(135deg,var(--accent),#3a86ff)' : 'transparent';
});

let recognition = null;
micBtn.addEventListener('click', () => {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SpeechRec){ alert('SpeechRecognition not available'); return; }
  if(recognition){ try{ recognition.stop(); }catch(e){} recognition=null; micBtn.style.background=''; return; }
  recognition = new SpeechRec();
  recognition.lang = 'en-US';
  recognition.onresult = (e) => {
    const t = e.results[0][0].transcript;
    userInput.value = (userInput.value ? userInput.value + ' ' : '') + t;
  };
  recognition.onerror = (e) => { debugLog('STT error: ' + e.error); };
  recognition.onend = () => { recognition=null; micBtn.style.background=''; };
  try{ recognition.start(); micBtn.style.background='linear-gradient(135deg,var(--accent),#3a86ff)'; } catch(err){ debugLog('STT start failed: ' + err); }
});

function maybeSpeak(text){
  if(!ttsToggle.classList.contains('active')) return;
  if(!('speechSynthesis' in window)) return;
  try{
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'en-US';
    window.speechSynthesis.speak(u);
  } catch(e){ debugLog('TTS error: ' + String(e)); }
}

/* -------------------------
   Shortcuts & input handling
   ------------------------- */
userInput.addEventListener('keydown', (e) => {
  if(e.key === 'Enter' && !e.shiftKey){
    e.preventDefault();
    sendBtn.click();
  }
});

/* Send button */
sendBtn.addEventListener('click', async () => {
  const text = userInput.value.trim();
  if(!text) return;
  userInput.value='';
  await generateReply(text);
});

/* Stream simulate */
streamBtn.addEventListener('click', async () => {
  const text = userInput.value.trim();
  if(!text) return;
  userInput.value='';
  chat.push({role:'user', text, time: nowISO()});
  chat.push({role:'assistant', text:'Thinking...', time: nowISO()});
  persistAll(); renderChat();
  const reply = localHeuristicReply(text) + ' (streamed)';
  // streaming typing effect
  let cur='';
  for(let i=0;i<reply.length;i++){
    if(abortFlag.abort) break;
    cur = reply.slice(0,i+1);
    replaceLastAssistant(applyPersona(cur));
    await new Promise(res=>setTimeout(res,12));
  }
  replaceLastAssistant(applyPersona(reply));
  persistAll();
});

/* Stop */
stopBtn.addEventListener('click', () => { abortFlag.abort = true; stopBtn.style.display='none'; statusEl.textContent='Stopped'; });

/* -------------------------
   Boot / initialize UI
   ------------------------- */
function renderAll(){
  personaNameEl.value = persona.name || 'Jack';
  personaPromptEl.value = persona.system_prompt || '';
  renderChat(); renderMemories(); renderGallery(); renderFiles(); renderWeights();
  tinyInfo.textContent = tinyModel ? 'Tiny: loaded' : 'Tiny: not initialized';
  debugLog('App initialized');
}

renderAll();

/* Expose debug API to window for convenience */
window.jack = { getState: ()=>({chat,memories,examples,persona,tinyModel,pipelines,galleryItems}), persist: persistAll, gen: generateReply, log: debugLog };

</script>

</body>
</html>
