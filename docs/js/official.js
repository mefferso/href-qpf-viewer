const bounds={se:[[22.5,-98.5],[38.5,-72.5]],sp:[[20.5,-110.5],[40.5,-84.5]]};
const map=L.map('map').setView([30.1,-90.2],6);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png',{maxZoom:18,subdomains:'abcd'}).addTo(map);

const COUNTY_ZONE_AREAS=['LA','MS','AL'];
const STATE_BOUNDARY_URL="https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/0/query?where=STATE%20IN%20(%2701%27%2C%2722%27%2C%2728%27)&outFields=STATE%2CNAME&returnGeometry=true&outSR=4326&f=geojson";
const LIX_CWA_BOUNDARY_URL='data/lix_cwa_boundary.geojson';

function setupBoundaryPane(name,zIndex){
  if(!map.getPane(name))map.createPane(name);
  const pane=map.getPane(name);
  pane.style.zIndex=zIndex;
  pane.style.pointerEvents='none';
}

async function fetchBoundaryJson(url){
  const response=await fetch(url,{cache:'no-store',headers:{Accept:'application/geo+json, application/json'}});
  if(!response.ok)throw new Error(`HTTP ${response.status} loading ${url}`);
  return response.json();
}

async function fetchCountyZoneBoundaries(){
  const collections=await Promise.all(COUNTY_ZONE_AREAS.map(area=>fetchBoundaryJson(`https://api.weather.gov/zones?type=county&area=${area}`)));
  return {
    type:'FeatureCollection',
    features:collections.flatMap(collection=>(collection.features||[]).filter(feature=>feature.geometry))
  };
}

async function loadBoundaryLayers(){
  setupBoundaryPane('countyLinePane',560);
  setupBoundaryPane('stateLinePane',570);
  setupBoundaryPane('lixCwaPane',590);

  try{
    const counties=await fetchCountyZoneBoundaries();
    L.geoJSON(counties,{
      pane:'countyLinePane',
      interactive:false,
      style:()=>({color:'#000',weight:.75,opacity:.78,fill:false,fillOpacity:0,smoothFactor:.15,lineCap:'round',lineJoin:'round'})
    }).addTo(map);
  }catch(err){
    console.warn('Could not load land county/parish boundaries',err);
  }

  try{
    const states=await fetchBoundaryJson(STATE_BOUNDARY_URL);
    L.geoJSON(states,{
      pane:'stateLinePane',
      interactive:false,
      style:()=>({color:'#000',weight:1.35,opacity:.9,fill:false,fillOpacity:0,smoothFactor:.15,lineCap:'round',lineJoin:'round'})
    }).addTo(map);
  }catch(err){
    console.warn('Could not load state boundaries',err);
  }

  try{
    const lixCwa=await fetchBoundaryJson(LIX_CWA_BOUNDARY_URL);
    L.geoJSON(lixCwa,{
      pane:'lixCwaPane',
      interactive:false,
      style:()=>({color:'#000',weight:2.2,opacity:1,fill:false,fillOpacity:0,smoothFactor:.05,lineCap:'round',lineJoin:'round'})
    }).addTo(map);
  }catch(err){
    console.warn('Could not load LIX CWA boundary',err);
  }
}
loadBoundaryLayers();

const $=id=>document.getElementById(id);
const state={catalog:null,layer:null,img:null,sector:'se',opacity:1};

function fmtUTC(iso){if(!iso)return '-';const d=new Date(iso);if(isNaN(d))return '-';return `${d.getUTCMonth()+1}/${d.getUTCDate()} ${String(d.getUTCHours()).padStart(2,'0')}Z`}
function fmtGen(iso){if(!iso)return '-';const d=new Date(iso);if(isNaN(d))return '-';return `${d.getUTCMonth()+1}/${d.getUTCDate()} ${String(d.getUTCHours()).padStart(2,'0')}:${String(d.getUTCMinutes()).padStart(2,'0')}Z`}
function fmtRun(l){if(!l)return '-';const r=String(l.run||'');return /^\d{10}$/.test(r)?`${r.slice(0,4)}-${r.slice(4,6)}-${r.slice(6,8)} ${r.slice(8,10)}Z`:(l.runLabel||r||'-')}
function layers(){return(state.catalog?.layers||[]).filter(x=>x.available&&x.sector===state.sector)}
function fillTimes(prefer){const sel=$('timeSelect');sel.innerHTML='';const ls=layers();for(const l of ls){const o=document.createElement('option');o.value=l.id;o.textContent=`${l.periodLabel} ending ${fmtUTC(l.validTimeUTC)}`;sel.appendChild(o)}return ls.find(x=>x.id===prefer)?.id||ls.find(x=>x.forecastHour===12)?.id||ls[0]?.id}
function updateMeta(){const l=state.layer;$('runValue').textContent=fmtRun(l);$('validValue').textContent=l?fmtUTC(l.validTimeUTC):'-';$('periodValue').textContent=l?.periodLabel||'-';$('generatedValue').textContent=fmtGen(state.catalog?.generatedUTC);$('openPng').href=l?.imageUrl||'#';$('status').innerHTML=`Loaded ${layers().length} official SPC image periods • <b>PNG only: no hover/max values</b>`}
async function setLayer(id){const l=layers().find(x=>x.id===id);if(!l)throw new Error('Layer not found: '+id);state.layer=l;$('timeSelect').value=l.id;if(state.img)map.removeLayer(state.img);state.img=L.imageOverlay(l.imageUrl,bounds[l.sector]||bounds.se,{opacity:state.opacity}).addTo(map);updateMeta()}
async function refreshSector(){state.sector=$('sectorSelect').value;const id=fillTimes(state.layer?.id);await setLayer(id)}
function step(d){const ls=layers();const i=ls.findIndex(x=>x.id===state.layer.id);const n=Math.max(0,Math.min(ls.length-1,i+d));setLayer(ls[n].id)}

$('sectorSelect').onchange=refreshSector;
$('timeSelect').onchange=()=>setLayer($('timeSelect').value);
$('opacitySlider').oninput=()=>{state.opacity=Number($('opacitySlider').value);if(state.img)state.img.setOpacity(state.opacity)};
$('prevBtn').onclick=()=>step(-1);
$('nextBtn').onclick=()=>step(1);

(async()=>{
  try{
    state.catalog=await fetch('data/spc_official_catalog.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error('HTTP '+r.status);return r.json()});
    const id=fillTimes(state.catalog.defaultLayerId);
    await setLayer(id);
    $('loading').classList.add('hidden');
  }catch(e){
    $('loading').classList.add('hidden');
    $('status').innerHTML='<span style="color:#ff7b72;font-weight:900">Failed to load SPC catalog.</span><br>'+e.message;
    console.error(e);
  }
})();
