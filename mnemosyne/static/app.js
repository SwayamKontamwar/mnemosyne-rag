const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { tag: '', folder: '', fileType: '' };
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
}[char]));

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || 'Something went wrong.');
  return data;
}

function currentFilters() {
  return {
    tag: $('#tag-filter').value || null,
    folder: $('#folder-filter').value || null,
    file_type: $('#type-filter').value || null
  };
}

function queryString(filters = currentFilters()) {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => value && params.set(key, value));
  return params.toString() ? `?${params}` : '';
}

async function loadLibrary() {
  const { documents, stats, filters } = await request(`/api/library${queryString()}`);
  $('#stats').innerHTML = [
    `${stats.documents} source${stats.documents === 1 ? '' : 's'}`,
    `${stats.chunks} searchable passages`,
    `${Math.round(stats.characters / 1000)}k characters`,
    `${stats.tags} total tags`
  ].map((value) => `<span class="stat">${escapeHtml(value)}</span>`).join('');
  hydrateFilter($('#tag-filter'), filters.tags, state.tag, 'All tags');
  hydrateFilter($('#folder-filter'), filters.folders, state.folder, 'All folders');
  hydrateFilter($('#type-filter'), filters.types, state.fileType, 'All types');
  $('#documents').innerHTML = documents.length ? documents.map((doc) => `
    <article class="document-row">
      <span class="file-icon">${escapeHtml(doc.type)}</span>
      <div>
        <strong title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</strong>
        <small>${doc.chunk_count} passages · ${Math.round(doc.character_count / 1000)}k characters · ${escapeHtml(doc.folder)}</small>
        <div class="tag-row">${(doc.tags || []).map((tag) => `<button class="tag-chip" data-tag="${escapeHtml(tag)}">${escapeHtml(tag)}</button>`).join('')}</div>
      </div>
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
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, ...currentFilters() })
    });
    $('#results').innerHTML = data.results.length ? data.results.map((hit) => `
        <article class="result" data-chunk-id="${hit.id}">
        <div class="result-head"><strong>${escapeHtml(hit.title)}</strong><span>${Math.max(0, Math.round(hit.score * 100))}% match</span></div>
        <p>${escapeHtml(hit.text.slice(0, 520))}${hit.text.length > 520 ? '…' : ''}</p>
        <div class="tag-row">${(hit.tags || []).map((tag) => `<span class="mini-tag">${escapeHtml(tag)}</span>`).join('')}</div>
        <div class="citation">${escapeHtml(hit.citation)}</div>
        <button class="preview-link" data-chunk-id="${hit.id}">Open cited chunk</button>
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
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, ...currentFilters() })
    });
    output.innerHTML = `<div class="validation-strip ${escapeHtml(data.validation.verdict)}">
      <strong>${escapeHtml(validationLabel(data.validation.verdict))}</strong>
      <span>${escapeHtml(validationDetail(data.validation))}</span>
    </div>
    <h3>Answer</h3><p>${escapeHtml(data.answer).replace(/\n/g, '<br>')}</p><div class="sources">${data.sources.map((source) => `
      <div class="source"><strong>[${source.number}] ${escapeHtml(source.title)}</strong><div class="citation">${escapeHtml(source.citation)}</div><button class="preview-link" data-chunk-id="${source.chunk_id}">Preview source</button></div>`).join('')}</div>`;
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
['tag-filter', 'folder-filter', 'type-filter'].forEach((id) => {
  $(`#${id}`).addEventListener('change', async () => {
    state.tag = $('#tag-filter').value;
    state.folder = $('#folder-filter').value;
    state.fileType = $('#type-filter').value;
    await loadLibrary();
    await loadInsights();
  });
});
document.addEventListener('click', async (event) => {
  const previewButton = event.target.closest('.preview-link');
  if (previewButton) await showPreview(previewButton.dataset.chunkId);
  const tagChip = event.target.closest('.tag-chip');
  if (tagChip) {
    state.tag = tagChip.dataset.tag;
    $('#tag-filter').value = state.tag;
    await loadLibrary();
    await runSearch($('#global-search').value.trim() || state.tag);
  }
});

Promise.all([loadLibrary(), loadInsights()]).catch((error) => { $('#documents').innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p>`; });

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

async function loadInsights() {
  const [graph, clusters] = await Promise.all([
    request('/api/graph'),
    request('/api/clusters')
  ]);
  $('#graph').innerHTML = graph.edges.length ? graph.edges.map((edge) => `
    <article class="graph-edge">
      <div><strong>${escapeHtml(shortName(edge.source))}</strong><span>→</span><strong>${escapeHtml(shortName(edge.target))}</strong></div>
      <small>${Math.round(edge.weight * 100)}% related · ${escapeHtml(edge.reason)}</small>
    </article>`).join('') : '<p class="empty-state">Add more linked notes to see graph relationships.</p>';
  $('#clusters').innerHTML = clusters.clusters.length ? clusters.clusters.map((cluster) => `
    <article class="cluster-card">
      <strong>${escapeHtml(cluster.name)}</strong>
      <small>${cluster.documents.length} document${cluster.documents.length === 1 ? '' : 's'}</small>
      <div class="tag-row">${cluster.keywords.map((keyword) => `<span class="mini-tag">${escapeHtml(keyword)}</span>`).join('')}</div>
    </article>`).join('') : '<p class="empty-state">Clusters will appear as your notes gain stronger themes.</p>';
}

async function showPreview(chunkId) {
  const data = await request(`/api/chunks/${chunkId}`);
  $('#preview-panel').hidden = false;
  $('#preview-title').textContent = data.title;
  $('#preview-body').innerHTML = `
    <div class="preview-meta">
      <span class="mini-tag">${escapeHtml(data.citation)}</span>
      ${(data.tags || []).map((tag) => `<span class="mini-tag">${escapeHtml(tag)}</span>`).join('')}
    </div>
    <pre>${escapeHtml(data.text)}</pre>`;
  $('#preview-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function shortName(path) {
  return String(path).split('/').pop().replace(/\.[^.]+$/, '');
}

function hydrateFilter(select, values, current, fallback) {
  const unique = [...new Set(values || [])];
  select.innerHTML = `<option value="">${escapeHtml(fallback)}</option>${unique.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join('')}`;
  select.value = current || '';
}

function validationLabel(verdict) {
  return {
    grounded: 'Citations look grounded',
    'weak-support': 'Some claims may be weakly supported',
    'invalid-citations': 'Answer cited missing sources',
    'missing-citations': 'Answer needs citations'
  }[verdict] || 'Citation audit incomplete';
}

function validationDetail(validation) {
  if (validation.verdict === 'grounded') return `Referenced ${validation.cited_numbers.length} source${validation.cited_numbers.length === 1 ? '' : 's'}.`;
  if (validation.verdict === 'missing-citations') return 'The model answered without usable source markers.';
  if (validation.verdict === 'invalid-citations') return `Missing source numbers: ${validation.missing_numbers.join(', ')}.`;
  if (validation.verdict === 'weak-support') return `Potentially weak support for source numbers: ${validation.unsupported_numbers.join(', ')}.`;
  return 'The answer audit needs review.';
}
