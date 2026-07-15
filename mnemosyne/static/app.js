const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { tag: '', folder: '', fileType: '', lastQuery: '', currentReaderPath: '' };

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
    tag: $('#tag-filter')?.value || null,
    folder: $('#folder-filter')?.value || null,
    file_type: $('#type-filter')?.value || null,
    as_of: currentAsOf()
  };
}

function currentAsOf() {
  const value = $('#as-of-input')?.value;
  return value ? new Date(value).toISOString() : null;
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
    `${stats.tags} total tags`,
    `${stats.saved_searches} saved searches`,
    `${stats.watch_folders} watch folders`
  ].map((value) => `<span class="stat">${escapeHtml(value)}</span>`).join('');
  hydrateFilter($('#tag-filter'), filters.tags, state.tag, 'All tags');
  hydrateFilter($('#folder-filter'), filters.folders, state.folder, 'All folders');
  hydrateFilter($('#type-filter'), filters.types, state.fileType, 'All types');
  $('#documents').innerHTML = documents.length ? documents.map((doc) => `
    <article class="document-row" data-document-path="${escapeHtml(doc.path)}">
      <span class="file-icon">${escapeHtml(doc.type)}</span>
      <div>
        <strong title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</strong>
        <small>${doc.chunk_count} passages · ${Math.round(doc.character_count / 1000)}k characters · ${escapeHtml(doc.folder)}</small>
        <div class="tag-row">${(doc.tags || []).map((tag) => `<button class="tag-chip" data-tag="${escapeHtml(tag)}">${escapeHtml(tag)}</button>`).join('')}</div>
      </div>
      <time>${new Date(doc.indexed_at.replace(' ', 'T') + 'Z').toLocaleDateString()}</time>
    </article>`).join('') : '<p class="empty-state">No files indexed.</p>';
}

async function uploadFiles(files) {
  if (!files.length) return;
  const status = $('#upload-status');
  const form = new FormData();
  [...files].forEach((file) => form.append('files', file));
  status.textContent = `Indexing ${files.length} file${files.length === 1 ? '' : 's'}…`;
  try {
    const data = await request('/api/documents', { method: 'POST', body: form });
    const added = data.indexed.filter((item) => item.indexed).length;
    const skipped = data.indexed.filter((item) => !item.indexed);
    const rejected = data.rejected.length ? ` ${data.rejected.length} rejected.` : '';
    const reasons = [...skipped, ...data.rejected]
      .map((item) => item.diagnostics?.[0]?.message || item.reason)
      .filter(Boolean);
    status.textContent = [
      `Indexed ${added} source${added === 1 ? '' : 's'}.`,
      skipped.length ? ` ${skipped.length} had no searchable text.` : '',
      rejected,
      reasons.length ? ` ${reasons[0]}` : ''
    ].join('');
    await refreshAll();
  } catch (error) {
    status.textContent = error.message;
  }
}

async function runSearch(query) {
  state.lastQuery = query;
  const panel = $('#search-results');
  panel.hidden = false;
  const asOf = currentAsOf();
  $('#results-title').textContent = asOf ? `Results for “${query}” as of ${new Date(asOf).toLocaleString()}` : `Results for “${query}”`;
  $('#results').innerHTML = '<p class="empty-state">Searching…</p>';
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  try {
    const data = await request('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, ...currentFilters() })
    });
    $('#results').innerHTML = data.results.length ? data.results.map((hit) => `
      <article class="result" data-chunk-id="${hit.id}">
        <div class="result-head"><strong>${escapeHtml(hit.title)}</strong><span>${Math.max(0, Math.round(hit.score * 100))}% match</span></div>
        <p>${escapeHtml(hit.text.slice(0, 520))}${hit.text.length > 520 ? '…' : ''}</p>
        <div class="tag-row">${(hit.tags || []).map((tag) => `<span class="mini-tag">${escapeHtml(tag)}</span>`).join('')}</div>
        <div class="citation">${escapeHtml(hit.citation)} · rev ${escapeHtml(hit.revision || 1)}</div>
        <button class="preview-link" data-chunk-id="${hit.id}">Open cited chunk</button>
      </article>`).join('') : '<p class="empty-state">No matching passages found.</p>';
    await loadHistory();
  } catch (error) {
    $('#results').innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p>`;
  }
}

async function loadInsights() {
  const [graph, clusters] = await Promise.all([request('/api/graph'), request('/api/clusters')]);
  $('#graph').innerHTML = graph.edges.length ? graph.edges.map((edge) => `
    <article class="graph-edge">
      <div><strong>${escapeHtml(shortName(edge.source))}</strong><span>→</span><strong>${escapeHtml(shortName(edge.target))}</strong></div>
      <small>${Math.round(edge.weight * 100)}% related · ${escapeHtml(edge.reason)}</small>
    </article>`).join('') : '<p class="empty-state">No links yet.</p>';
  $('#clusters').innerHTML = clusters.clusters.length ? clusters.clusters.map((cluster) => `
    <article class="cluster-card">
      <strong>${escapeHtml(cluster.name)}</strong>
      <small>${cluster.documents.length} document${cluster.documents.length === 1 ? '' : 's'}</small>
      <div class="tag-row">${cluster.keywords.map((keyword) => `<span class="mini-tag">${escapeHtml(keyword)}</span>`).join('')}</div>
    </article>`).join('') : '<p class="empty-state">No clusters yet.</p>';
}

async function loadSavedSearches() {
  const data = await request('/api/saved-searches');
  $('#saved-searches').innerHTML = data.saved_searches.length ? data.saved_searches.map((search) => `
    <article class="graph-edge">
      <div><strong>${escapeHtml(search.name)}</strong></div>
      <small>${escapeHtml(search.query)}</small>
      <button class="preview-link run-saved-search"
        data-query="${escapeHtml(search.query)}"
        data-tag="${escapeHtml(search.tag || '')}"
        data-folder="${escapeHtml(search.folder || '')}"
        data-type="${escapeHtml(search.file_type || '')}">Run search</button>
    </article>`).join('') : '<p class="empty-state">No saved searches.</p>';
}

async function loadHistory() {
  const data = await request('/api/history');
  $('#history').innerHTML = data.history.length ? data.history.map((entry) => `
    <article class="graph-edge">
      <div><strong>${escapeHtml(entry.mode.toUpperCase())}</strong></div>
      <small>${escapeHtml(entry.query)}</small>
    </article>`).join('') : '<p class="empty-state">No history yet.</p>';
}

async function loadEvaluations() {
  const data = await request('/api/evaluations');
  const counts = Object.entries(data.counts || {});
  $('#evaluation-dashboard').innerHTML = counts.length ? `
    <div class="stats">${counts.map(([key, value]) => `<span class="stat">${escapeHtml(key)}: ${value}</span>`).join('')}</div>
    <div class="graph-list">${(data.recent || []).map((item) => `
      <article class="graph-edge">
        <div><strong>${escapeHtml(item.verdict)}</strong></div>
        <small>${escapeHtml(item.query)}</small>
    </article>`).join('')}</div>` : '<p class="empty-state">No data yet.</p>';
}

async function loadWatchFolders() {
  const data = await request('/api/watch-folders');
  $('#watch-folders').innerHTML = data.watch_folders.length ? data.watch_folders.map((watch) => `
    <article class="graph-edge">
      <div><strong>${escapeHtml(shortName(watch.path))}</strong></div>
      <small>${escapeHtml(watch.profile)} · ${escapeHtml(watch.path)}</small>
    </article>`).join('') : '<p class="empty-state">No watch folders.</p>';
}

async function loadSettings() {
  const data = await request('/api/settings');
  $('#setting-embed-provider').value = data.preferences.embed_provider || data.runtime.embed_provider;
  $('#setting-embed-model').value = data.preferences.embed_model || data.runtime.embed_model;
  $('#setting-vector-provider').value = data.preferences.vector_provider || data.runtime.vector_provider || 'sqlite';
  $('#setting-ollama-model').value = data.preferences.ollama_model || data.runtime.ollama_model;
  $('#setting-privacy-mode').value = data.preferences.privacy_mode || 'local-first';
}

async function loadHealth() {
  const data = await request('/api/health');
  const state = $('#model-state');
  if (data.ollama.available && data.ollama.embed_model_ready && data.ollama.answer_model_ready) {
    state.innerHTML = '<span></span>Ollama ready';
    state.title = `${data.embed_model} + ${data.ollama_model}`;
  } else if (data.ollama.available) {
    state.innerHTML = '<span></span>Models missing';
    state.title = 'Pull the configured Ollama models.';
  } else {
    state.innerHTML = '<span></span>Search only';
    state.title = 'Start Ollama for chat.';
  }
}

async function loadCollections() {
  const data = await request('/api/collections');
  $('#collections').innerHTML = data.collections.length ? data.collections.map((collection) => `
    <article class="graph-edge"><div><strong>${escapeHtml(collection.name)}</strong></div>
    <small>${escapeHtml((collection.tags || []).join(', ') || collection.query || 'Manual collection')}</small></article>`).join('') : '<p class="empty-state">No collections.</p>';
}

async function showPreview(chunkId) {
  const data = await request(`/api/chunks/${chunkId}`);
  $('#preview-panel').hidden = false;
  $('#preview-title').textContent = data.title;
  $('#preview-body').innerHTML = `
    <div class="preview-meta">
      <span class="mini-tag">${escapeHtml(data.citation)}</span>
      <span class="mini-tag">rev ${escapeHtml(data.revision || 1)}</span>
      ${data.valid_to ? `<span class="mini-tag">historical</span>` : '<span class="mini-tag">current</span>'}
      ${(data.tags || []).map((tag) => `<span class="mini-tag">${escapeHtml(tag)}</span>`).join('')}
    </div>
    <pre>${escapeHtml(data.text)}</pre>`;
  $('#preview-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function showReader(path) {
  state.currentReaderPath = path;
  const [data, history] = await Promise.all([
    request(`/api/reader?path=${encodeURIComponent(path)}`),
    request(`/api/revisions?path=${encodeURIComponent(path)}`)
  ]);
  $('#reader-panel').hidden = false;
  $('#reader-title').textContent = data.document?.title || shortName(path);
  const revisions = history.revisions || [];
  $('#reader-body').innerHTML = `
    <div class="preview-meta">${(data.document?.tags || []).map((tag) => `<span class="mini-tag">${escapeHtml(tag)}</span>`).join('')}</div>
    <div class="reader-grid">
      <div>${(data.chunks || []).map((chunk) => `
          <article class="reader-chunk">
          <div class="citation">${escapeHtml(chunk.citation)} · rev ${escapeHtml(chunk.document_version || 1)}</div>
          <pre>${escapeHtml(chunk.text)}</pre>
        </article>`).join('')}</div>
      <aside>
        <h4>History</h4>
        <div class="history-tools">
          <select id="diff-left">${revisions.map((revision) => `<option value="${revision.version}">v${revision.version}</option>`).join('')}</select>
          <select id="diff-right">${revisions.map((revision, index) => `<option value="${revision.version}" ${index === 0 ? 'selected' : ''}>v${revision.version}</option>`).join('')}</select>
          <button id="diff-revisions" class="quiet-button" type="button">Diff</button>
        </div>
        <div id="revision-diff" class="diff-output" hidden></div>
        <div class="graph-list">${revisions.map((revision) => `
          <article class="graph-edge">
            <div><strong>v${escapeHtml(revision.version)}</strong><span>${revision.tombstone ? 'Deleted' : 'Saved'}</span></div>
            <small>${new Date(revision.created_at).toLocaleString()} · ${escapeHtml(revision.digest.slice(0, 10))}</small>
            ${revision.tombstone ? '' : `<button class="preview-link restore-revision" data-version="${revision.version}">Restore this version</button>`}
          </article>`).join('') || '<p class="empty-state">No revisions yet.</p>'}</div>
        <h4>Related notes</h4>
        <div class="graph-list">${(data.related || []).map((item) => `
          <article class="graph-edge"><div><strong>${escapeHtml(shortName(item.path))}</strong></div><small>${Math.round(item.weight * 100)}% · ${escapeHtml(item.reason)}</small></article>`).join('') || '<p class="empty-state">No related notes.</p>'}</div>
        <h4>Entities</h4>
        <div class="tag-row">${(data.entities || []).map((entity) => `<span class="mini-tag">${escapeHtml(entity)}</span>`).join('')}</div>
        <h4>Timeline</h4>
        <div class="graph-list">${(data.timeline || []).map((item) => `
          <article class="graph-edge"><div><strong>${escapeHtml(item.date)}</strong></div><small>${escapeHtml(item.citation)}</small></article>`).join('') || '<p class="empty-state">No dates found.</p>'}</div>
        <h4>Contradictions</h4>
        <div class="graph-list">${(data.contradictions || []).map((item) => `
          <article class="graph-edge"><div><strong>${escapeHtml(item.left)}</strong></div><small>${escapeHtml(item.right)} · ${escapeHtml(item.shared_terms.join(', '))}</small></article>`).join('') || '<p class="empty-state">No contradictions flagged.</p>'}</div>
      </aside>
    </div>`;
  $('#reader-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
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
  return 'Check citations.';
}

async function refreshAll() {
  await Promise.all([loadLibrary(), loadInsights(), loadSavedSearches(), loadHistory(), loadEvaluations(), loadWatchFolders(), loadSettings(), loadHealth(), loadCollections()]);
}

$('#global-search').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && event.currentTarget.value.trim()) runSearch(event.currentTarget.value.trim());
});

document.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    $('#global-search').focus();
  }
});

$('#ask-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = $('#ask-input').value.trim();
  const output = $('#answer');
  output.hidden = false;
  output.innerHTML = '<p>Searching sources…</p>';
  try {
    const data = await request('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, ...currentFilters() })
    });
    output.innerHTML = `<div class="validation-strip ${escapeHtml(data.validation.verdict)}"><strong>${escapeHtml(validationLabel(data.validation.verdict))}</strong><span>${escapeHtml(validationDetail(data.validation))}</span></div>
    <h3>Answer</h3><p>${escapeHtml(data.answer).replace(/\n/g, '<br>')}</p><div class="sources">${data.sources.map((source) => `
      <div class="source"><strong>[${source.number}] ${escapeHtml(source.title)}</strong><div class="citation">${escapeHtml(source.citation)}</div><button class="preview-link" data-chunk-id="${source.chunk_id}">Preview source</button></div>`).join('')}</div>`;
    await Promise.all([loadEvaluations(), loadHistory()]);
  } catch (error) {
    output.innerHTML = `<h3>Answer unavailable</h3><p>${escapeHtml(error.message)}</p>`;
  }
});

const dropZone = $('#drop-zone');
['dragenter', 'dragover'].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', (event) => uploadFiles(event.dataTransfer.files));
$('#file-input').addEventListener('change', (event) => uploadFiles(event.target.files));
$('#refresh-library').addEventListener('click', refreshAll);
$('.mobile-menu').addEventListener('click', () => $('.sidebar').classList.toggle('open'));
$$('.nav-item').forEach((button) => button.addEventListener('click', () => navigate(button.dataset.view)));
['tag-filter', 'folder-filter', 'type-filter'].forEach((id) => {
  $(`#${id}`).addEventListener('change', async () => {
    state.tag = $('#tag-filter').value;
    state.folder = $('#folder-filter').value;
    state.fileType = $('#type-filter').value;
    await Promise.all([loadLibrary(), loadInsights()]);
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

  const saved = event.target.closest('.run-saved-search');
  if (saved) {
    state.tag = saved.dataset.tag;
    state.folder = saved.dataset.folder;
    state.fileType = saved.dataset.type;
    $('#tag-filter').value = state.tag;
    $('#folder-filter').value = state.folder;
    $('#type-filter').value = state.fileType;
    await loadLibrary();
    await runSearch(saved.dataset.query);
  }

  const documentRow = event.target.closest('.document-row');
  if (documentRow && !event.target.closest('.tag-chip')) {
    await showReader(documentRow.dataset.documentPath);
  }

  const diffButton = event.target.closest('#diff-revisions');
  if (diffButton && state.currentReaderPath) {
    const left = $('#diff-left').value;
    const right = $('#diff-right').value;
    const data = await request(`/api/revisions/diff?path=${encodeURIComponent(state.currentReaderPath)}&left=${left}&right=${right}`);
    $('#revision-diff').hidden = false;
    $('#revision-diff').innerHTML = `<pre>${escapeHtml(data.diff || 'No line changes.')}</pre>`;
  }

  const restoreButton = event.target.closest('.restore-revision');
  if (restoreButton && state.currentReaderPath) {
    if (!confirm(`Restore revision ${restoreButton.dataset.version}? This writes a new revision; history stays intact.`)) return;
    await request('/api/revisions/restore', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: state.currentReaderPath, version: Number(restoreButton.dataset.version) })
    });
    await refreshAll();
    await showReader(state.currentReaderPath);
  }
});

$('#saved-search-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const name = $('#saved-search-name').value.trim();
  const query = state.lastQuery || $('#global-search').value.trim();
  if (!name || !query) return;
  await request('/api/saved-searches', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, query, ...currentFilters() })
  });
  $('#saved-search-name').value = '';
  await loadSavedSearches();
});

$('#watch-submit').addEventListener('click', async () => {
  const path = $('#watch-path').value.trim();
  if (!path) return;
  await request('/api/watch-folders', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, profile: $('#watch-profile').value })
  });
  $('#watch-path').value = '';
  await refreshAll();
});

$('#watch-scan').addEventListener('click', async () => {
  await request('/api/watch-folders/scan', { method: 'POST' });
  await refreshAll();
});

$('#collection-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  await request('/api/collections', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: $('#collection-name').value.trim(),
      tags: $('#collection-tags').value.split(',').map((tag) => tag.trim()).filter(Boolean),
      query: state.lastQuery
    })
  });
  event.currentTarget.reset();
  await loadCollections();
});

$('#settings-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  await request('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      embed_provider: $('#setting-embed-provider').value,
      vector_provider: $('#setting-vector-provider').value,
      embed_model: $('#setting-embed-model').value.trim(),
      ollama_model: $('#setting-ollama-model').value.trim(),
      privacy_mode: $('#setting-privacy-mode').value
    })
  });
  await loadSettings();
});

$('.settings-button').addEventListener('click', () => navigate('settings'));

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

refreshAll().catch((error) => {
  $('#documents').innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p>`;
});
