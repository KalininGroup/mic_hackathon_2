// Populate sections from JSON files (optional, safe to delete if you prefer to edit HTML directly)
async function fetchJSON(path){ 
  try{ const r = await fetch(path); if(!r.ok) throw new Error(path+" not found"); return await r.json(); }
  catch(e){ return null; }
}

function linkCell(url, label){
  if(!url) return "";
  const a = document.createElement("a");
  a.href = url; a.textContent = label || "Link"; a.target="_blank"; a.rel="noopener";
  return a.outerHTML;
}

(async function init(){
  const agenda = await fetchJSON("data/agenda.json");
  const projects = await fetchJSON("data/projects.json");
  const results = await fetchJSON("data/results.json");
  const media = await fetchJSON("data/media.json");

  // Agenda
  const agendaBody = document.getElementById("agenda-body");
  if(agendaBody && Array.isArray(agenda)){
    agenda.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${row.time||""}</td><td>${row.session||""}</td><td>${row.speaker||""}</td>`;
      agendaBody.appendChild(tr);
    });
  }

  // Projects
  const grid = document.getElementById("projects-grid");
  if(grid && Array.isArray(projects)){
    projects.forEach(p => {
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = `<h3>${p.title||"Untitled"}</h3>
        <p>${p.summary||""}</p>
        <p>${p.code?`<a href="${p.code}" target="_blank" rel="noopener">Code</a>`:""} ${p.data?` · <a href="${p.data}" target="_blank" rel="noopener">Data</a>`:""}</p>`;
      grid.appendChild(card);
    });
  }

  // Results
  const resultsBody = document.getElementById("results-body");
  if(resultsBody && Array.isArray(results)){
    results.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${r.rank||""}</td>
        <td>${r.team||""}</td>
        <td>${r.participants||""}</td>
        <td>${r.code?linkCell(r.code,"Code"):""}</td>
        <td>${r.drive?linkCell(r.drive,"Drive"):""}</td>`;
      resultsBody.appendChild(tr);
    });
  }

  // Media
  const mediaList = document.getElementById("media-list");
  if(mediaList && Array.isArray(media)){
    media.forEach(m => {
      const li = document.createElement("li");
      li.innerHTML = `<a href="${m.url}" target="_blank" rel="noopener">${m.title}</a>${m.note?` — ${m.note}`:""}`;
      mediaList.appendChild(li);
    });
  }
})();