// ═══════════════════════════════════════════════════════════
//  FlexTime app.js — Full Dashboard Logic
// ═══════════════════════════════════════════════════════════

// ── STATE ────────────────────────────────────────────────────
let GS = null; // globalState from API
let currentRole = null;
let currentTask = 'task_medium';
let simStep = 0, simReward = 0;
let rewardHistory = [];
let leaveOptSel = 1;
let xaiLog = [];

const DAYS = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
const PERIODS = ['morning','afternoon','night'];
const PERIOD_LABEL = {morning:'🌅 Morning',afternoon:'🌞 Afternoon',night:'🌙 Night'};
const DEMAND_DATA = [
  {d:'Mon',v:17,p:85},{d:'Tue',v:19,p:95},{d:'Wed',v:14,p:68},
  {d:'Thu',v:17,p:85},{d:'Fri',v:21,p:100},{d:'Sat',v:12,p:60},{d:'Sun',v:8,p:38}
];

// ── CHARTS ───────────────────────────────────────────────────
const CHART_DEFAULTS = {
  color:'rgba(255,255,255,.7)',
  grid:{color:'rgba(255,255,255,.04)'},
  ticks:{color:'rgba(148,163,184,.7)',font:{size:9}}
};
let charts = {};

function mkChart(id, type, data, opts={}) {
  const el = document.getElementById(id);
  if (!el) return null;
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(el, {
    type,
    data,
    options: {
      responsive: true, maintainAspectRatio: false, animation: {duration:600},
      plugins: {legend:{display:opts.legend||false,labels:{color:'rgba(148,163,184,.8)',font:{size:9},boxWidth:10}}, tooltip:{bodyFont:{size:10},titleFont:{size:10}}},
      scales: opts.noScale ? undefined : {
        x:{grid:CHART_DEFAULTS.grid,ticks:CHART_DEFAULTS.ticks,border:{color:'rgba(255,255,255,.06)'}},
        y:{grid:CHART_DEFAULTS.grid,ticks:CHART_DEFAULTS.ticks,border:{color:'rgba(255,255,255,.06)'}, ...( opts.yOpts||{})}
      },
      ...opts.extra
    }
  });
  return charts[id];
}

function buildDemandChart() {
  mkChart('demandChart','bar',{
    labels: DEMAND_DATA.map(d=>d.d),
    datasets:[
      {label:'Demand',data:DEMAND_DATA.map(d=>d.v),backgroundColor:'rgba(59,130,246,.5)',borderColor:'rgba(59,130,246,.8)',borderWidth:1,borderRadius:3},
      {label:'Coverage',data:DEMAND_DATA.map(d=>Math.round(d.v*(GS?GS.metrics?.coverage_rate||0.7:0.7))),backgroundColor:'rgba(16,185,129,.4)',borderColor:'rgba(16,185,129,.8)',borderWidth:1,borderRadius:3}
    ]
  },{legend:true});
}

function buildCoverageChart() {
  const cov = GS ? DEMAND_DATA.map(d=>Math.round(d.v*(GS.metrics?.coverage_rate||0.7))) : DEMAND_DATA.map(d=>Math.round(d.v*0.7));
  mkChart('coverageChart','line',{
    labels:DEMAND_DATA.map(d=>d.d),
    datasets:[
      {label:'Demand',data:DEMAND_DATA.map(d=>d.v),borderColor:'rgba(59,130,246,.8)',backgroundColor:'rgba(59,130,246,.08)',fill:true,tension:.4,pointRadius:3},
      {label:'Staffed',data:cov,borderColor:'rgba(16,185,129,.8)',backgroundColor:'rgba(16,185,129,.08)',fill:true,tension:.4,pointRadius:3}
    ]
  },{legend:true});
}

function buildSkillChart() {
  const skills = GS ? [...new Set(GS.employees?.flatMap(e=>e.skills||[]))] : ['nurse','triage','ICU','supervisor'];
  const vals = skills.map(s => {
    if (!GS) return Math.random()*40+60;
    const need = GS.shifts?.filter(sh=>sh.required_skill===s).length||0;
    const have = GS.employees?.filter(e=>e.skills?.includes(s)).length||0;
    return need?Math.min(100,Math.round(have/need*100)):100;
  });
  mkChart('skillChart','bar',{
    labels:skills,
    datasets:[{label:'Coverage %',data:vals,backgroundColor:vals.map(v=>v>=80?'rgba(16,185,129,.55)':v>=50?'rgba(245,158,11,.55)':'rgba(239,68,68,.55)'),borderRadius:4}]
  },{yOpts:{min:0,max:100}});
}

function buildNightShiftChart() {
  const emps = GS?.employees?.slice(0,6)||[{name:'A'},{name:'B'},{name:'C'},{name:'D'}];
  mkChart('nightShiftChart','bar',{
    labels:emps.map(e=>e.name?.split(' ')[0]),
    datasets:[{label:'Night Shifts',data:emps.map(()=>Math.floor(Math.random()*4)),backgroundColor:'rgba(245,158,11,.45)',borderRadius:3}]
  },{});
}

function buildRewardChart() {
  const labels = rewardHistory.map((_,i)=>`${i+1}`);
  const data = {
    labels,
    datasets:[{label:'Reward',data:rewardHistory,borderColor:'rgba(16,185,129,.9)',backgroundColor:'rgba(16,185,129,.1)',fill:true,tension:.4,pointRadius:2,pointBackgroundColor:'rgba(16,185,129,.9)'}]
  };
  ['rewardChart','rpRewardChart'].forEach(id=>mkChart(id,'line',data,{yOpts:{}}));
}

// ── GAUGE ─────────────────────────────────────────────────────
function setGauge(score) {
  const pct = Math.min(Math.max(score,0),1);
  const offset = 188 - pct*188;
  const color = pct>=0.75?'var(--green)':pct>=0.5?'var(--amber)':'var(--rose)';
  const grade = pct>=0.9?'A+':pct>=0.8?'A':pct>=0.7?'B':pct>=0.6?'C':'D';
  ['gaugeArc','rp-gaugeArc'].forEach(id=>{
    const a=document.getElementById(id);
    if(a){a.style.strokeDashoffset=offset;a.style.stroke=color;}
  });
  ['gaugeVal','rp-score'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=pct.toFixed(2);});
  ['gaugeGrade','rp-grade'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=grade;});
}

// ── TOAST ─────────────────────────────────────────────────────
function toast(msg, icon='✅', dur=3000) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.innerHTML = `<span>${icon}</span><span>${msg}</span>`;
  document.getElementById('toastContainer').appendChild(el);
  setTimeout(()=>el.classList.add('show'),10);
  setTimeout(()=>{el.classList.remove('show');setTimeout(()=>el.remove(),400);},dur);
}

// ── LOGIN / LOGOUT ────────────────────────────────────────────
function quickLogin(role) {
  document.getElementById('loginUser').value = role;
  doLogin();
}

function doLogin() {
  const u = document.getElementById('loginUser').value.trim().toLowerCase();
  const roles = ['admin','manager','employee','ceo'];
  if (!roles.includes(u)) {
    document.getElementById('loginErr').textContent = 'Use: admin, manager, employee, or ceo';
    return;
  }
  currentRole = u;
  document.getElementById('loginPage').style.display = 'none';
  document.getElementById('appShell').classList.add('visible');
  document.getElementById('navUname').textContent = u.charAt(0).toUpperCase()+u.slice(1);
  document.getElementById('navUrole').textContent = u.toUpperCase();
  document.getElementById('navAvatar').textContent = u[0].toUpperCase();
  applyRoleVisibility(u);
  loadState();
}

function logout() {
  currentRole = null;
  document.getElementById('appShell').classList.remove('visible');
  document.getElementById('loginPage').style.display = 'flex';
  document.getElementById('loginUser').value = '';
  document.getElementById('loginPass').value = '';
  document.getElementById('loginErr').textContent = '';
}

function applyRoleVisibility(role) {
  const body = document.body;
  body.classList.remove('role-admin','role-manager','role-employee','role-ceo');
  body.classList.add(`role-${role}`);
  if (role === 'employee') showView('dashboard', document.getElementById('sb-dashboard'));
  else if (role === 'manager') showView('schedule', document.getElementById('sb-schedule'));
  else if (role === 'ceo') showView('analytics', null);
  else showView('dashboard', document.getElementById('sb-dashboard'));
}

// ── NAVIGATION ────────────────────────────────────────────────
function showView(id, el) {
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  const v = document.getElementById('view-'+id);
  if (v) v.classList.add('active');
  document.querySelectorAll('.sb-item').forEach(i=>i.classList.remove('active'));
  if (el) el.classList.add('active');
  else {
    const sb = document.getElementById('sb-'+id);
    if (sb) sb.classList.add('active');
  }
  // Lazy-render charts when tab becomes visible
  if (id==='analytics') { setTimeout(()=>{buildCoverageChart();buildSkillChart();buildHeatmap();},50); }
  if (id==='fairness')  { setTimeout(()=>{renderFairnessBarsDetail();buildNightShiftChart();},50); }
  if (id==='simulation'){ setTimeout(()=>{buildRewardChart();},50); }
}

// ── API ───────────────────────────────────────────────────────
async function loadState() {
  try {
    const res = await fetch('/state');
    if (!res.ok) throw new Error('no state');
    GS = await res.json();
    renderAll();
  } catch {
    const res = await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:currentTask,seed:42})});
    if (res.ok) { GS = await res.json(); renderAll(); }
  }
}

async function resetToTask(taskId) {
  currentTask = taskId;
  document.getElementById('taskSelect').value = taskId;
  toast(`Resetting to ${taskId}…`,'🔄');
  const res = await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:taskId,seed:42})});
  if (res.ok) { GS = await res.json(); simStep=0; simReward=0; rewardHistory=[]; renderAll(); toast(`Loaded ${taskId}`,'✅'); }
}

async function onTaskChange() {
  currentTask = document.getElementById('taskSelect').value;
  await resetToTask(currentTask);
}

function onWorkplaceChange() {
  toast(`Workplace type updated`,'🏥');
}

// ── AGENT STEP ────────────────────────────────────────────────
async function stepSim() {
  if (!GS) { toast('Reset first','⚠️'); return; }
  const emps = GS.employees||[];
  const unassigned = GS.unassigned_shifts||[];
  if (!unassigned.length) { toast('All shifts assigned!','✅'); return; }

  const shifts = Object.fromEntries((GS.shifts||[]).map(s=>[s.id,s]));
  let bestAction = {action_type:'noop'};
  for (const sid of unassigned) {
    const shf = shifts[sid]; if (!shf) continue;
    for (const emp of emps.sort((a,b)=>a.assigned_hours-b.assigned_hours)) {
      if (!emp.skills?.includes(shf.required_skill)) continue;
      if (!emp.availability?.[shf.day]) continue;  // shf.day is already an int 0-6
      if ((emp.assigned_hours||0)+(shf.duration_hours||8) > emp.max_hours_per_week) continue;
      bestAction = {action_type:'assign',employee_id:emp.id,shift_id:sid}; break;
    }
    if (bestAction.action_type!=='noop') break;
  }

  const res = await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(bestAction)});
  if (!res.ok) { toast('Step failed','❌'); return; }
  const result = await res.json();
  GS = result.observation;
  const r = result.reward?.total||0;
  simStep++; simReward=+(simReward+r).toFixed(3);
  rewardHistory.push(+r.toFixed(3));
  document.getElementById('sim-steps').textContent=simStep;
  document.getElementById('sim-reward').textContent=simReward.toFixed(2);
  document.getElementById('rp-steps').textContent=simStep;
  document.getElementById('rp-ret').textContent=simReward.toFixed(2);
  document.getElementById('sb-steps-badge').textContent=simStep;
  addSimLog(`Step ${simStep}: ${bestAction.action_type} → reward ${r>=0?'+':''}${r.toFixed(3)}`);
  addXaiLog(bestAction, emps, shifts);
  buildRewardChart();
  renderState();
  if (result.done || bestAction.action_type==='noop') {
    fetchGrade();
    toast(`Episode done! ${simStep} steps`,'🏁');
  }
}

async function runAgent() {
  if (!GS) { await loadState(); }
  document.getElementById('runAgentBtn').textContent='⏳ Running…';
  document.getElementById('runAgentBtn').disabled=true;
  let done = GS?.done||false, noop=0;
  const maxSteps = GS?.max_steps||60;
  while (!done && simStep < maxSteps && noop<5) {
    await stepSim();
    done = GS?.done||false;
    if ((GS?.unassigned_shifts||[]).length===0) noop++;
    await new Promise(r=>setTimeout(r,40));
  }
  document.getElementById('runAgentBtn').textContent='▶ Run Agent';
  document.getElementById('runAgentBtn').disabled=false;
  fetchGrade();
}

async function fetchGrade() {
  try {
    const res = await fetch('/grader');
    if (!res.ok) return;
    const g = await res.json();
    const score = g.score||0;
    setGauge(score);
    document.getElementById('sim-score').textContent=score.toFixed(3);
    document.getElementById('kpi-score').textContent=(score*100).toFixed(0)+'%';
    document.getElementById('kpi-score-sub').textContent=g.passed?'✅ Task passed':'❌ Below target';
    renderBreakdown(g.breakdown||{});
    toast(`Score: ${score.toFixed(3)} — ${g.passed?'PASS':'FAIL'}`,'🏆',4000);
  } catch {}
}

async function runBaseline() {
  toast('Running baseline on all 3 tasks…','⚡',8000);
  try {
    const res = await fetch('/baseline',{method:'POST'});
    if (!res.ok) throw new Error();
    const data = await res.json();
    toast(`Baseline mean: ${data.mean_score?.toFixed(3)}`,'🎯',5000);
    renderTaskResults(data.results||[]);
    showView('tasks',null);
  } catch { toast('Baseline error','❌'); }
}

async function runAllTasks() {
  await runBaseline();
}

// ── RENDER ALL ────────────────────────────────────────────────
function renderAll() {
  renderState();
  buildDemandChart();
  renderAlerts();
  renderFairnessBars();
  renderEmpTable();
  renderShiftsTable();
  renderUnassigned();
  renderConflicts();
  updateKPIs();
  updateRightPanel();
}

function renderState() {
  if (!GS) return;
  renderScheduleGrid('dashScheduleGrid', true);
  renderScheduleGrid('mainScheduleGrid', false);
  updateKPIs();
  updateRightPanel();
  renderAlerts();
}

function updateKPIs() {
  if (!GS) return;
  const m = GS.metrics||{};
  const cov = m.coverage_rate||0;
  const fair = m.fairness_score||0;
  const viol = (GS.conflicts||[]).length;
  setText('kpi-coverage', (cov*100).toFixed(0)+'%');
  setText('kpi-fairness', fair.toFixed(2));
  setText('kpi-violations', viol);
  setText('an-util', Math.round((m.avg_hours||0)) + ' / ' + Math.round(((GS.employees||[]).reduce((a,e)=>a+e.max_hours_per_week,0))) + ' hrs');
  setText('an-unmet', m.unmet_demand||0);
  setText('an-skill', (cov*100).toFixed(0)+'%');
  setText('an-ot', 0);
}

function updateRightPanel() {
  if (!GS) return;
  const m = GS.metrics||{};
  const cov = m.coverage_rate||0;
  const fair = m.fairness_score||0;
  const unmet = m.unmet_demand||0;
  const totalH = (GS.employees||[]).reduce((a,e)=>a+e.assigned_hours,0);
  const maxH = (GS.employees||[]).reduce((a,e)=>a+e.max_hours_per_week,0);
  const util = maxH ? totalH/maxH : 0;
  setText('rp-unmet', unmet);
  setText('rp-fairness', fair.toFixed(2));
  setText('rp-util', totalH.toFixed(0)+' / '+maxH+' hrs');
  setText('rp-skill', (cov*100).toFixed(0)+'%');
  setBar('rp-fairness-bar', fair*100);
  setBar('rp-util-bar', util*100);
  setBar('rp-skill-bar', cov*100);
  setText('sched-coverage-chip', (cov*100).toFixed(0)+'% filled');
  const conflicts = (GS.conflicts||[]).length;
  setText('sched-conflict-chip', `${conflicts} conflict${conflicts!==1?'s':''}`);
  setText('sb-conflicts-badge', conflicts||'');
  setText('unassigned-count', (GS.unassigned_shifts||[]).length);
}

function setText(id, v) { const e=document.getElementById(id); if(e) e.textContent=v; }
function setBar(id, pct) { const e=document.getElementById(id); if(e) e.style.width=Math.min(100,Math.max(0,pct))+'%'; }

// ── SCHEDULE GRID ─────────────────────────────────────────────
function renderScheduleGrid(containerId, mini) {
  const el = document.getElementById(containerId);
  if (!el || !GS) return;
  const emps = GS.employees||[];
  const shifts = GS.shifts||[];
  const days = mini ? DAYS.slice(0,5) : DAYS;
  const showEmps = mini ? emps.slice(0,4) : emps;

  let html = `<div class="sg"><div class="sg-hrow"><div class="sg-h">Employee</div>${days.map(d=>`<div class="sg-h">${d}</div>`).join('')}</div>`;
  for (const emp of showEmps) {
    html += `<div class="sg-row"><div class="sg-lbl">${empAvatar(emp)}${emp.name?.split(' ')[0]||emp.id}</div>`;
    for (const day of days) {
      const dayIdx = DAYS.indexOf(day);
      const dayShifts = shifts.filter(s=>s.day===dayIdx&&s.assigned_employee_id===emp.id);
      html += `<div class="sg-cell">${dayShifts.map(s=>`<div class="stag s-${s.period}" title="${s.id}" onclick="removeAssign('${s.id}')">${s.period?.slice(0,3)||'?'} <span style="opacity:.6;font-size:.55rem">✕</span></div>`).join('')}</div>`;
    }
    html += '</div>';
  }
  html += '</div>';
  el.innerHTML = html;
}

function empAvatar(emp) {
  const colors = ['var(--green)','var(--sky)','var(--amber)','var(--purple)','var(--rose)'];
  const c = colors[emp.id?.charCodeAt(emp.id?.length-1)%colors.length]||colors[0];
  return `<div style="width:18px;height:18px;border-radius:50%;background:${c};display:inline-flex;align-items:center;justify-content:center;font-size:.55rem;font-weight:800;color:#000;margin-right:.35rem;flex-shrink:0">${(emp.name||emp.id||'?')[0]?.toUpperCase()}</div>`;
}

async function removeAssign(shiftId) {
  if (!GS) return;
  const shf = GS.shifts?.find(s=>s.id===shiftId);
  if (!shf?.assigned_employee_id) return;
  const res = await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action_type:'remove',employee_id:shf.assigned_employee_id,shift_id:shiftId})});
  if (res.ok) { const r=await res.json(); GS=r.observation; renderState(); toast('Assignment removed','🗑'); }
}

// ── EMPLOYEE TABLE ────────────────────────────────────────────
function renderEmpTable() {
  const tbody = document.getElementById('empTableBody');
  if (!tbody || !GS) return;
  const emps = GS.employees||[];
  const isAdmin = currentRole==='admin';
  tbody.innerHTML = emps.map(emp=>{
    const wellness = Math.max(0,100 - ((emp.assigned_hours||0)/Math.max(1,emp.max_hours_per_week)*60));
    const wCol = wellness>70?'var(--green)':wellness>40?'var(--amber)':'var(--rose)';
    const rel = 4; // simulated reliability
    const avail = DAYS.map((_,i)=>`<div class="av-d ${emp.availability?.[i]?'av-y':'av-n'}">${emp.availability?.[i]?'✓':'·'}</div>`).join('');
    const skills = (emp.skills||[]).map(s=>`<span class="skill-tag">${s}</span>`).join('');
    return `<tr>
      <td class="font-mono text-xs text-muted">${emp.id}</td>
      <td><div class="flex items-center gap-1">${empAvatar(emp)}<span style="font-weight:600">${emp.name}</span></div></td>
      <td>${skills}</td>
      <td><div class="avail-row">${avail}</div></td>
      <td><span class="font-mono">${(emp.assigned_hours||0).toFixed(0)}</span><span class="text-muted text-xs">/${emp.max_hours_per_week}</span></td>
      <td><div class="wellness-bar"><div class="wb-track"><div class="wb-fill" style="width:${wellness}%;background:${wCol}"></div></div><span class="wb-txt" style="color:${wCol}">${wellness.toFixed(0)}%</span></div></td>
      <td><span class="rel-stars">${'⭐'.repeat(rel)}</span></td>
      <td><span class="chip chip-s">${emp.preferred_shift||'any'}</span></td>
      ${isAdmin?`<td><button class="btn btn-ghost btn-sm" onclick="toast('Edit coming soon','✏️')">Edit</button></td>`:'<td></td>'}
    </tr>`;
  }).join('');
}

// ── SHIFTS TABLE ──────────────────────────────────────────────
function renderShiftsTable() {
  const tbody = document.getElementById('shiftsTableBody');
  if (!tbody || !GS) return;
  const shifts = GS.shifts||[];
  const priMap = {high:'chip-r',medium:'chip-a',low:'chip-s'};
  tbody.innerHTML = shifts.slice(0,20).map(s=>`<tr>
    <td class="font-mono text-xs text-muted">${s.id}</td>
    <td>${s.day}</td>
    <td class="font-mono text-xs">${s.period==='morning'?'09:00-17:00':s.period==='afternoon'?'13:00-21:00':'21:00-05:00'}</td>
    <td><span class="chip chip-s">${s.period}</span></td>
    <td><span class="skill-tag">${s.required_skill}</span></td>
    <td>${s.min_staff||1}</td>
    <td class="font-mono">${s.duration_hours||8}h</td>
    <td><span class="chip ${priMap[s.priority]||'chip-s'}">${s.priority||'medium'}</span></td>
    <td>${s.assigned_employee_id?`<span class="chip chip-g">✓ ${s.assigned_employee_id}</span>`:`<span class="chip chip-a">Unassigned</span>`}</td>
  </tr>`).join('');
}

// ── ALERTS ────────────────────────────────────────────────────
function renderAlerts() {
  const el = document.getElementById('alertsList');
  if (!el) return;
  const alerts = [];
  const conflicts = GS?.conflicts||[];
  const unassigned = GS?.unassigned_shifts||[];

  if (conflicts.length) alerts.push({type:'r',icon:'⚠️',title:`${conflicts.length} constraint violation${conflicts.length>1?'s':''}`,msg:'Hard constraints detected — run agent to resolve.',time:'now'});
  if (unassigned.length) alerts.push({type:'a',icon:'📋',title:`${unassigned.length} unassigned shift${unassigned.length>1?'s':''}`,msg:'Run the AI agent to auto-schedule remaining shifts.',time:'now'});
  if (!alerts.length) alerts.push({type:'g',icon:'✅',title:'All constraints satisfied',msg:'Schedule is fully optimized and conflict-free.',time:'now'});
  if (GS) alerts.push({type:'s',icon:'ℹ️',title:'OpenEnv connected',msg:`Task: ${GS.task_id||currentTask} · Step ${GS.step_count||0}/${GS.max_steps||60}`,time:'live'});

  el.innerHTML = alerts.map(a=>`<div class="alert alert-${a.type}"><div class="a-icon">${a.icon}</div><div><div class="a-title">${a.title}</div><div class="a-msg">${a.msg}</div></div><div class="a-time">${a.time}</div></div>`).join('');
  setText('alertCount', alerts.filter(a=>a.type==='r'||a.type==='a').length||'');
}

// ── FAIRNESS BARS ─────────────────────────────────────────────
function renderFairnessBars() {
  const el = document.getElementById('fairnessBars');
  if (!el || !GS) return;
  const emps = (GS.employees||[]).slice(0,6);
  const maxH = Math.max(...emps.map(e=>e.assigned_hours||0),1);
  el.innerHTML = emps.map(e=>{
    const pct = (e.assigned_hours||0)/maxH*100;
    const col = pct>80?'var(--rose)':pct>50?'var(--green)':'var(--sky)';
    return `<div class="fair-row"><span class="fair-name">${e.name?.split(' ')[0]||e.id}</span><div class="fair-track"><div class="fair-fill" style="width:${pct}%;background:${col}"></div></div><span class="fair-hrs">${(e.assigned_hours||0).toFixed(0)}h</span></div>`;
  }).join('');
}

function renderFairnessBarsDetail() {
  ['fairnessBarsDetail','wellnessBars'].forEach((id,idx)=>{
    const el = document.getElementById(id);
    if (!el || !GS) return;
    const emps = GS.employees||[];
    const maxH = Math.max(...emps.map(e=>e.assigned_hours||0),1);
    el.innerHTML = emps.slice(0,8).map(e=>{
      const v = idx===0?(e.assigned_hours||0)/maxH*100:Math.max(0,100-((e.assigned_hours||0)/Math.max(1,e.max_hours_per_week)*60));
      const col = idx===0?(v>80?'var(--rose)':v>40?'var(--green)':'var(--sky)'):(v>70?'var(--green)':v>40?'var(--amber)':'var(--rose)');
      return `<div class="fair-row"><span class="fair-name">${e.name?.split(' ')[0]||e.id}</span><div class="fair-track"><div class="fair-fill" style="width:${v}%;background:${col}"></div></div><span class="fair-hrs">${idx===0?(e.assigned_hours||0).toFixed(0)+'h':v.toFixed(0)+'%'}</span></div>`;
    }).join('');
  });
}

// ── UNASSIGNED & CONFLICTS ────────────────────────────────────
function renderUnassigned() {
  const el = document.getElementById('unassignedList');
  if (!el || !GS) return;
  const unassigned = GS.unassigned_shifts||[];
  const shiftsMap = Object.fromEntries((GS.shifts||[]).map(s=>[s.id,s]));
  if (!unassigned.length) { el.innerHTML='<div class="text-sm text-muted" style="text-align:center;padding:1rem">✅ All shifts assigned!</div>'; return; }
  el.innerHTML = unassigned.slice(0,12).map(sid=>{
    const s=shiftsMap[sid]||{}; return `<div class="flex items-center gap-1" style="padding:.35rem 0;border-bottom:1px solid var(--border);font-size:.72rem">
      <span class="chip chip-a">${s.period||'?'}</span>
      <span>${s.day||'?'}</span>
      <span class="skill-tag" style="margin-left:.2rem">${s.required_skill||'?'}</span>
      <button class="btn btn-primary btn-sm" style="margin-left:auto" onclick="autoAssignShift('${sid}')">Assign</button>
    </div>`;
  }).join('');
}

async function autoAssignShift(shiftId) {
  if (!GS) return;
  const shifts = Object.fromEntries((GS.shifts||[]).map(s=>[s.id,s]));
  const shf = shifts[shiftId]; if (!shf) return;
  const emps = [...(GS.employees||[])].sort((a,b)=>a.assigned_hours-b.assigned_hours);
  let pick = null;
  for (const emp of emps) {
    if (!emp.skills?.includes(shf.required_skill)) continue;
    if (!emp.availability?.[shf.day]) continue;  // shf.day is int 0-6
    if ((emp.assigned_hours||0)+(shf.duration_hours||8)>emp.max_hours_per_week) continue;
    pick = emp; break;
  }
  if (!pick) { toast('No eligible employee found','⚠️'); return; }
  const res = await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action_type:'assign',employee_id:pick.id,shift_id:shiftId})});
  if (res.ok) { const r=await res.json(); GS=r.observation; renderState(); toast(`Assigned ${pick.name?.split(' ')[0]} → ${shiftId}`,'✅'); }
}

function renderConflicts() {
  const el = document.getElementById('conflictsList');
  if (!el || !GS) return;
  const conflicts = GS.conflicts||[];
  if (!conflicts.length) { el.innerHTML='<div class="text-sm text-muted" style="text-align:center;padding:1rem">✅ No conflicts!</div>'; return; }
  el.innerHTML = conflicts.slice(0,6).map(c=>`<div class="alert alert-r"><div class="a-icon">⚠️</div><div><div class="a-title">${c.type||'Conflict'}</div><div class="a-msg">${c.description||JSON.stringify(c)}</div></div></div>`).join('');
}

// ── REWARD BREAKDOWN ──────────────────────────────────────────
function renderBreakdown(bd) {
  const targets = ['rewardBreakdown','rp-breakdown'];
  const rows = Object.entries(bd).map(([k,v])=>`<div class="rw-row"><span class="rw-lbl">${k.replace(/_/g,' ')}</span><span class="rw-val ${v>=0?'pos':'neg'}">${v>=0?'+':''}${(+v).toFixed(2)}</span></div>`).join('');
  const total = Object.values(bd).reduce((a,v)=>a+(+v),0);
  const full = rows+`<div class="rw-total"><span>Total Score</span><span class="font-mono ${total>=0?'text-green':'text-rose'}">${total>=0?'+':''}${total.toFixed(2)}</span></div>`;
  targets.forEach(id=>{const e=document.getElementById(id);if(e)e.innerHTML=full;});
}

// ── TASK RESULTS ──────────────────────────────────────────────
function renderTaskResults(results) {
  const el = document.getElementById('taskResultsBody');
  if (!el) return;
  el.innerHTML = results.map(r=>`<div class="flex items-center gap-2" style="padding:.5rem 0;border-bottom:1px solid var(--border)">
    <span class="chip ${r.passed?'chip-g':'chip-r'}">${r.passed?'PASS':'FAIL'}</span>
    <span style="font-weight:600;font-size:.78rem">${r.task_id}</span>
    <span class="font-mono text-sm text-green" style="margin-left:auto">${(+r.score).toFixed(3)}</span>
    <span class="text-xs text-muted">${r.steps_used} steps</span>
  </div>`).join('');
}

// ── SIM LOG ───────────────────────────────────────────────────
function addSimLog(msg) {
  const el = document.getElementById('simLog');
  if (!el) return;
  el.innerHTML += `<div><span style="color:var(--text3)">[${String(simStep).padStart(3,'0')}]</span> <span style="color:var(--green)">${msg}</span></div>`;
  el.scrollTop = el.scrollHeight;
}

async function resetSim() {
  simStep=0; simReward=0; rewardHistory=[];
  setText('sim-steps',0); setText('sim-reward','0.00'); setText('sim-score','—');
  setText('rp-steps',0); setText('rp-ret','0.00');
  const el=document.getElementById('simLog'); if(el) el.innerHTML='Resetting environment…';
  buildRewardChart(); setGauge(0);
  // Also reset the backend environment
  const res = await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:currentTask,seed:42})});
  if(res.ok){GS=await res.json();renderAll();const el2=document.getElementById('simLog');if(el2)el2.innerHTML='Reset. Ready to simulate…';}
  toast('Simulation reset','↺');
}

// ── XAI ──────────────────────────────────────────────────────
function addXaiLog(action, emps, shifts) {
  const el = document.getElementById('xaiLogs');
  if (!el || action.action_type==='noop') return;
  const emp = emps.find(e=>e.id===action.employee_id);
  const shf = (shifts||{})[action.shift_id]||{};
  const now = new Date().toTimeString().slice(0,5);
  const entry = `<div class="xai-entry"><span class="xai-ts">[${now}]</span> <span class="xai-who">${emp?.name||action.employee_id}</span> → <strong>${action.shift_id}</strong> (${shf.day||'?'} ${shf.period||'?'})<br><span class="xai-why">✓ Skill match: ${shf.required_skill} | ✓ Available | ✓ Hours OK (${(emp?.assigned_hours||0).toFixed(0)}/${emp?.max_hours_per_week}h) | ✓ Fewest hours (fairness)</span></div>`;
  el.insertAdjacentHTML('afterbegin', entry);
}

function clearXai() { const el=document.getElementById('xaiLogs'); if(el) el.innerHTML='<div class="xai-entry text-muted text-xs">Log cleared.</div>'; }

// ── HEATMAP ───────────────────────────────────────────────────
function buildHeatmap() {
  const el = document.getElementById('demandHeatmap');
  if (!el) return;
  const periods = ['morning','afternoon','night'];
  const colors = [
    'rgba(239,68,68,','rgba(245,158,11,','rgba(16,185,129,','rgba(14,165,233,','rgba(59,130,246,','rgba(139,92,246,','rgba(255,255,255,'
  ];
  let html='';
  for (let p=0;p<periods.length;p++) {
    for (let d=0;d<7;d++) {
      const intensity=0.1+Math.random()*0.7;
      html+=`<div class="hm-cell" style="background:${colors[(d+p)%colors.length]}${intensity.toFixed(2)})" data-tip="${DAYS[d]||'?'} ${periods[p]}: ${Math.round(intensity*20)} staff needed"></div>`;
    }
  }
  el.innerHTML=html;
}

// ── LEAVE MODAL ───────────────────────────────────────────────
function openLeaveModal() { document.getElementById('leaveModal').classList.add('open'); }
function closeLeaveModal() { document.getElementById('leaveModal').classList.remove('open'); }
function selectLeaveOpt(n) {
  leaveOptSel=n;
  [1,2].forEach(i=>{const el=document.getElementById(`leaveOption${i}`);if(el)el.style.borderColor=i===n?'var(--green)':'var(--border)';});
}

async function confirmLeave() {
  closeLeaveModal();
  toast('Leave submitted — AI re-scheduling…','🏖',3000);
  await runAgent();
}

// ── ADD EMPLOYEE MODAL ────────────────────────────────────────
function openAddEmployee() { document.getElementById('addEmpModal').classList.add('open'); }
function closeAddEmpModal() { document.getElementById('addEmpModal').classList.remove('open'); }
function submitAddEmployee() {
  const name=document.getElementById('newEmpName').value.trim();
  if(!name){toast('Enter a name','⚠️');return;}
  toast(`${name} added (demo mode)`,'✅');
  closeAddEmpModal();
}

function openAddShift() { toast('Shift builder coming soon','✏️'); }

// ── SCENARIO ─────────────────────────────────────────────────
async function runScenario(type) {
  const scenarios = {
    shortage:{title:'🚨 Staff Shortage Simulation',desc:'Simulated 30% no-show rate. AI automatically re-assigned available employees to fill critical shifts.',color:'var(--rose)'},
    surge:{title:'📈 Demand Surge Simulation',desc:'Peak demand spike of +40%. Emergency on-call shifts triggered for 3 employees. Coverage maintained at 87%.',color:'var(--amber)'},
    holiday:{title:'🎄 Holiday Season Simulation',desc:'Reduced availability (25% staff on leave). Premium demand periods covered via overtime and preference overrides.',color:'var(--sky)'}
  };
  const s=scenarios[type];
  const el=document.getElementById('scenarioResult');
  if(!el)return;
  el.innerHTML=`<div class="card-head"><div class="card-title">${s.title}</div></div><div class="card-body"><div class="alert alert-s"><div class="a-icon">📊</div><div><div class="a-title">Scenario Complete</div><div class="a-msg">${s.desc}</div></div></div></div>`;
  toast(`Scenario: ${type}`,'🎲',3000);
}

// ── INIT ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Close modals on backdrop click
  document.querySelectorAll('.modal-bg').forEach(m=>{
    m.addEventListener('click', e=>{ if(e.target===m) m.classList.remove('open'); });
  });
});
