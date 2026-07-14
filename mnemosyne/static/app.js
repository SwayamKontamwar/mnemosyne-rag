const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
}[char]));

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || 'Something went wrong.');
  return data;
}

async function loadLibrary() {
  const { documents, stats } = await request('/api/library');
  $('#stats').innerHTML = [
    `${stats.documents} source${stats.documents === 1 ? '' : 's'}`,
    `${stats.chunks} searchable passages`,
    `${Math.round(stats.characters / 1000)}k characters`
  ].map((value) => `<span class="stat">${escapeHtml(value)}</span>`).join('');
  $('#documents').innerHTML = documents.length ? documents.map((doc) => `
    <article class="document-row">
      <span class="file-icon">${escapeHtml(doc.type)}</span>
      <div><strong title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</strong><small>${doc.chunk_count} passages · ${Math.round(doc.character_count / 1000)}k characters</small></div>
      <time>${new Date(doc.indexed_at.replace(' ', 'T') + 'Z').toLocaleDateString()}</time>
    </article>`).join('') : '<p class="empty-state">Your library is empty. Add your first source to begin.</p>';
}

async function uploadFiles(files) {
  if (!files.length) return;
  const status = $('#upload-status');
  const form = new FormData();
  [...files].forEach((file) => form.append('files', file));
  status.textContent = `Indexing ${files.length} file${files.length === 1 ? '' : 's'}…`;
  try {
    const data = await request('/api/documents', { method: 'POST', body: form });
    const rejected = data.rejected.length ? ` ${data.rejected.length} unsupported.` : '';
    status.textContent = `Added ${data.indexed.length} source${data.indexed.length === 1 ? '' : 's'}.${rejected}`;
    await loadLibrary();
  } catch (error) {
    status.textContent = error.message;
  }
}

async function runSearch(query) {
  const panel = $('#search-results');
  panel.hidden = false;
  $('#results-title').textContent = `Results for “${query}”`;
  $('#results').innerHTML = '<p class="empty-state">Searching your knowledge…</p>';
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  try {
    const data = await request('/api/search', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query })
    });
    $('#results').innerHTML = data.results.length ? data.results.map((hit) => `
        <article class="result">
        <div class="result-head"><strong>${escapeHtml(hit.title)}</strong><span>${Math.max(0, Math.round(hit.score * 100))}% match</span></div>
        <p>${escapeHtml(hit.text.slice(0, 520))}${hit.text.length > 520 ? '…' : ''}</p>
        <div class="citation">${escapeHtml(hit.citation)}</div>
      </article>`).join('') : '<p class="empty-state">No matching passages found.</p>';
  } catch (error) {
    $('#results').innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p>`;
  }
}

$('#global-search').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && event.currentTarget.value.trim()) runSearch(event.currentTarget.value.trim());
});

document.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault(); $('#global-search').focus();
  }
});

$('#ask-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = $('#ask-input').value.trim();
  const output = $('#answer');
  output.hidden = false;
  output.innerHTML = '<p>Reading your sources…</p>';
  try {
    const data = await request('/api/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query })
    });
    output.innerHTML = `<h3>Answer</h3><p>${escapeHtml(data.answer).replace(/\n/g, '<br>')}</p><div class="sources">${data.sources.map((source) => `
      <div class="source"><strong>[${source.number}] ${escapeHtml(source.title)}</strong><div class="citation">${escapeHtml(source.citation)}</div></div>`).join('')}</div>`;
  } catch (error) {
    output.innerHTML = `<h3>Couldn’t answer yet</h3><p>${escapeHtml(error.message)}</p>`;
  }
});

const dropZone = $('#drop-zone');
['dragenter', 'dragover'].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault(); dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault(); dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', (event) => uploadFiles(event.dataTransfer.files));
$('#file-input').addEventListener('change', (event) => uploadFiles(event.target.files));
$('#refresh-library').addEventListener('click', loadLibrary);
$('.mobile-menu').addEventListener('click', () => $('.sidebar').classList.toggle('open'));
$$('.nav-item').forEach((button) => button.addEventListener('click', () => navigate(button.dataset.view)));

loadLibrary().catch((error) => { $('#documents').innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p>`; });

function navigate(view) {
  $$('.nav-item').forEach((button) => button.classList.toggle('active', button.dataset.view === view));
  if (view === 'search') {
    $('#global-search').focus();
    $('#search-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } else {
    $(`[data-section="${view}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  $('.sidebar').classList.remove('open');
}
