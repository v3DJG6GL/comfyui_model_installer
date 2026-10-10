// Try to import app/api if the environment supports import maps; otherwise fall back to globals
let appRef, apiRef;
try {
  // dynamic import can fail on older bundles; we guard it
  ({ app: appRef } = await import('/scripts/app.js'));
  ({ api: apiRef } = await import('/scripts/api.js'));
} catch (e) {
  appRef = (typeof window !== 'undefined' ? window.app : undefined);
  apiRef = (typeof window !== 'undefined' ? window.api : undefined);
}

// Check if Model Installer server support is available
async function isModelInstallerAvailable() {
  try {
    const response = await fetch('/model_installer/health');
    const data = await response.json();
    return response.ok && data.ok;
  } catch {
    return false;
  }
}

// Minimal frontend extension that augments the Missing Models dialog entries
// Uses existing UI components, styles and fetch to call backend endpoints

// global app is provided by the ComfyUI frontend
if (appRef && typeof appRef.registerExtension === 'function') appRef.registerExtension({
  name: "comfyui_model_installer",
  async setup() {},
  async beforeRegisterNodeDef() {},
  async nodeCreated() {},
  async loadedGraph() {},
  async init() {
    // Check if server supports model installer
    const available = await isModelInstallerAvailable();
    if (!available) {
      console.log('[Model Installer] Server support not detected');
      return;
    }
    
    console.log('[Model Installer] Extension initialized');
    
    // augment missing model entries after UI loads. Observe DOM changes to attach buttons
    const observer = new MutationObserver(() => {
      augmentLegacyMissingModelsDialog();
      augmentMissingModelsPanel();
    });
    observer.observe(document.body, { childList: true, subtree: true });
  },
});

// Legacy frontend: "Missing Models" dialog with entries like "text_encoders / clip_l.safetensors"
function augmentLegacyMissingModelsDialog() {
  document.querySelectorAll('.comfy-missing-models li.p-listbox-option').forEach((li) => {
    const row = li.querySelector('div.flex.flex-row.items-center.gap-2');
    if (!row) return;
    if (row.querySelector('.model-install-buttons')) return;

    const labelSpan = row.querySelector('span[title]');
    const labelText = (labelSpan?.textContent || '').trim();
    const info = parseModelInfo(labelText);

    // Prefer URL from the Download button's title; avoid client-side CORS fetches
    const downloadBtn = row.querySelector('button[title]');
    let url = downloadBtn?.getAttribute('title') || '';
    // As a last resort, fallback to span title (may be a blob URL); we'll normalize it
    if (!url && labelSpan?.getAttribute('title')) url = labelSpan.getAttribute('title');
    url = normalizeDownloadUrl(url);

    const container = createInstallControls({
      directory: info?.directory,
      filename: info?.name,
      url,
      buttonClass: 'p-button p-component p-button-outlined p-button-sm',
      iconButtonClass: 'p-button p-component p-button-icon-only p-button-outlined p-button-sm',
    });
    row.appendChild(container);
  });
}

// Current frontend: "Missing Models" section in the right side panel (Workflow Overview > Issues).
// Rows only expose the model name and directory; the URL comes from the workflow's embedded model metadata.
function augmentMissingModelsPanel() {
  const nativeButtons = Array.from(document.querySelectorAll('button[data-testid="missing-model-download"]'))
    .filter((b) => !b.dataset.modelInstallerAttached);
  if (nativeButtons.length === 0) return;

  const modelsByName = collectWorkflowModels();

  nativeButtons.forEach((nativeBtn) => {
    nativeBtn.dataset.modelInstallerAttached = '1';
    const header = nativeBtn.parentElement;
    if (!header) return;

    // First titled element in the header is the model name (the link button after it also has a title)
    const filename = (header.querySelector('[title]')?.getAttribute('title') || '').trim();
    if (!filename) return;

    const candidates = modelsByName.get(filename) || [];
    // Metadata label reads "<directory> · <size>"
    const metaText = Array.from(header.querySelectorAll('span.block')).map((s) => s.textContent || '').find(Boolean) || '';
    const metaDirectory = metaText.split('·')[0].trim();
    const match = candidates.find((m) => m.directory === metaDirectory) || candidates[0];
    const directory = match?.directory || metaDirectory;
    const url = normalizeDownloadUrl(match?.url || '');
    if (!directory || !url) {
      console.debug('[Model Installer] No url/directory found for', filename);
      return;
    }

    const container = createInstallControls({
      directory,
      filename,
      url,
      buttonClass: nativeBtn.className,
      iconButtonClass: nativeBtn.className,
      onInstalled: () => document.querySelector('button[data-testid="missing-model-header-refresh"]')?.click(),
    });
    container.classList.add('pl-1', 'pb-1');
    header.insertAdjacentElement('afterend', container);
  });
}

// Gather { name, url, directory } entries embedded in the current workflow
// (top-level "models", node "properties.models", including subgraph definitions)
function collectWorkflowModels() {
  const byName = new Map();
  let data;
  try {
    data = appRef?.graph?.serialize?.();
  } catch (e) {
    console.warn('[Model Installer] Could not serialize workflow', e);
  }
  const seen = new Set();
  const visit = (obj) => {
    if (!obj || typeof obj !== 'object' || seen.has(obj)) return;
    seen.add(obj);
    if (Array.isArray(obj)) {
      obj.forEach(visit);
      return;
    }
    if (Array.isArray(obj.models)) {
      obj.models.forEach((m) => {
        if (m && typeof m.name === 'string' && typeof m.url === 'string') {
          if (!byName.has(m.name)) byName.set(m.name, []);
          byName.get(m.name).push({ name: m.name, url: m.url, directory: m.directory });
        }
      });
    }
    Object.values(obj).forEach(visit);
  };
  visit(data);
  return byName;
}

// Build the Install button + download path selector for one model entry
function createInstallControls({ directory, filename, url, buttonClass, iconButtonClass, onInstalled }) {
  const container = document.createElement('div');
  container.className = 'model-install-buttons flex gap-1 flex-row items-center';

  // Install button
  const btn = document.createElement('button');
  btn.className = buttonClass;
  btn.textContent = '...';

  // Wrapper for the folder icon and the transparent select element
  const folderWrapper = document.createElement('div');
  folderWrapper.style.position = 'relative';
  folderWrapper.style.display = 'inline-flex';
  folderWrapper.title = 'Select download path';

  // Folder button (visual only, not interactive)
  const folderBtn = document.createElement('button');
  folderBtn.className = iconButtonClass;
  folderBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-folder"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>`;
  folderBtn.setAttribute('tabindex', '-1'); // Make it non-focusable

  // The <select> element, styled as an invisible overlay
  const pathSelector = document.createElement('select');
  pathSelector.title = 'Select download path'; // Proper accessibility
  pathSelector.style.position = 'absolute';
  pathSelector.style.inset = '0';
  pathSelector.style.opacity = '0';
  pathSelector.style.cursor = 'pointer';

  folderWrapper.appendChild(folderBtn);
  folderWrapper.appendChild(pathSelector);

  container.appendChild(btn);
  container.appendChild(folderWrapper);

  const setBusy = (busy) => {
    btn.disabled = busy;
    btn.classList.toggle('p-disabled', busy);
  };

  const resetButtonStyle = () => {
    btn.style.backgroundColor = '';
    btn.style.color = '';
    btn.title = '';
  };

  const setLabel = (installed) => {
    btn.textContent = installed ? 'Uninstall' : 'Install';
    resetButtonStyle();
  };

  const query = async () => {
    try {
      let qs = '';
      if (directory && filename) {
        qs = `directory=${encodeURIComponent(directory)}&filename=${encodeURIComponent(filename)}`;
      } else if (url) {
        qs = `url=${encodeURIComponent(url)}`;
      }
      const res = await fetch(`/models/status?${qs}`);
      const js = await res.json();
      
      if (directory && filename) {
        updatePathSelector(pathSelector, folderBtn, directory, js.storage_info);
      }
      
      if (js.state === 'downloading') {
        btn.textContent = 'Downloading…';
        btn.disabled = true;
        pathSelector.disabled = true;
        folderBtn.disabled = true;
      } else if (js.state === 'failed') {
        btn.textContent = 'Retry Install';
        btn.disabled = false;
        btn.style.backgroundColor = '#ff6b6b';
        btn.style.color = 'white';
        btn.title = `Download failed: ${js.error || 'Unknown error'}. Click to retry.`;
        // Enable/disable based on path availability
        const isDisabled = (pathSelector.options.length <= 1);
        pathSelector.disabled = isDisabled;
        folderBtn.disabled = isDisabled;
      } else {
        setLabel(!!js.present);
        btn.disabled = false;
        btn.style.backgroundColor = '';
        btn.style.color = '';
        btn.title = '';
        const isDisabled = (pathSelector.options.length <= 1);
        pathSelector.disabled = isDisabled;
        folderBtn.disabled = isDisabled;
      }

      btn.onclick = async () => {
        const installing = (btn.textContent || '').trim().toLowerCase() !== 'uninstall';
        console.debug('[Model Installer] click', { installing, directory, filename, url });
        try {
          const ep = js.present ? '/models/uninstall' : '/models/install';
          let body = {};
          if (js.present) {
            body = { directory: js.folder, name: (js.path || '').split(/[\\/]/).pop() };
          } else if (url && directory && filename) {
            body = { 
              name: filename,
              directory: directory, 
              url: normalizeDownloadUrl(url),
              path: pathSelector.value // Always include selected path
            };
          } else {
            throw new Error('Insufficient information to install');
          }

          if (installing) {
            btn.textContent = 'Initiating...';
            btn.disabled = true;
            resetButtonStyle();
            pathSelector.disabled = true;
            folderBtn.disabled = true;
          } else {
            setBusy(true);
          }

          const r = await fetch(ep, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
          const jr = await r.json().catch(() => ({}));
          console.debug('[Model Installer] POST result', { status: r.status, jr });

          if (!r.ok) {
            if (r.status === 401 && jr && jr.error_code === 'auth_required') {
              const token = await openHfTokenDialog();
              if (token && token.trim().length > 0) {
                const lr = await fetch('/auth/hf_login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token: token.trim() }) });
                if ((await lr.json().catch(() => ({}))).ok) {
                  const rr = await fetch(ep, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
                  const rj = await rr.json().catch(() => ({}));
                  if (!rr.ok) throw new Error(rj.error || 'operation failed');
                  
                  btn.textContent = 'Downloading…';
                  btn.disabled = true;
                  const key = `${directory}/${filename}`;
                  startDownloadProgress({ directory, filename, url, expected: 0, onComplete: async () => { await query(); onInstalled?.(); } });
                  (async () => {
                    const exp = url ? await fetchExpected(url) : 0;
                    updateDownloadExpected(key, exp);
                  })();
                  return;
                } else {
                  alert('Login failed. Please run: hf auth login --token <token> --add-to-git-credential');
                }
              } else {
                alert('A Hugging Face access token is required. Visit https://huggingface.co/settings/tokens');
              }
            } else {
              alert(jr.error || 'operation failed');
            }

            if (installing) {
              btn.textContent = 'Retry Install';
              btn.disabled = false;
              btn.style.backgroundColor = '#ff6b6b';
              btn.style.color = 'white';
              btn.title = `Failed to start download: ${jr.error || 'Unknown error'}. Click to retry.`;
              const isDisabled = pathSelector.options.length <= 1;
              pathSelector.disabled = isDisabled;
              folderBtn.disabled = isDisabled;
            } else {
              setBusy(false);
            }
            return;
          }

          if (installing) {
            btn.textContent = 'Downloading…';
            const key = `${directory}/${filename}`;
            startDownloadProgress({ directory, filename, url, expected: 0, onComplete: async () => { await query(); onInstalled?.(); } });
            (async () => {
              const exp = url ? await fetchExpected(url) : 0;
              updateDownloadExpected(key, exp);
            })();
          } else {
            await query();
          }
        } catch (e) {
          alert(e.message || String(e));
          if (installing) {
            btn.textContent = 'Retry Install';
            btn.disabled = false;
            btn.style.backgroundColor = '#ff6b6b';
            btn.style.color = 'white';
            btn.title = `Failed to start download: ${e.message || String(e)}. Click to retry.`;
            const isDisabled = pathSelector.options.length <= 1;
            pathSelector.disabled = isDisabled;
            folderBtn.disabled = isDisabled;
          } else {
            setBusy(false);
          }
        }
      };
    } catch (e) {
      setLabel(false);
    }
  };

  query();
  return container;
}

// Helper functions for path selection
function parseModelInfo(labelText) {
  // Parse format: "text_encoders / clip_l.safetensors"
  const match = labelText.match(/^(.*?)\s*\/\s*([^\s]+\.[a-z0-9]+)\b/i);
  if (match) {
    return {
      directory: match[1].trim(),  // "text_encoders"
      name: match[2].trim()        // "clip_l.safetensors"
    };
  }
  return null;
}

// Get storage info for path selection
async function getStorageInfo() {
  try {
    const response = await fetch('/model_installer/health');
    const data = await response.json();
    return data.storage_info || {};
  } catch {
    return {};
  }
}

// Format bytes to GB
function formatGB(bytes) {
  return (bytes / (1024**3)).toFixed(1);
}

// Update path selector with available paths for specific folder type
function updatePathSelector(pathSelector, folderBtn, directory, storageInfo) {
  const storage = storageInfo || {};
  const paths = storage[directory] || [];
  
  pathSelector.innerHTML = '';
  if (paths.length > 0) {
      paths.forEach((pathInfo, index) => {
        const option = document.createElement('option');
        option.value = pathInfo.path;
        option.textContent = `${pathInfo.path} (${formatGB(pathInfo.available_bytes)} GB free)`;
        if (index === 0) option.selected = true;
        pathSelector.appendChild(option);
      });
  }

  const isDisabled = paths.length <= 1;
  pathSelector.disabled = isDisabled;
  folderBtn.disabled = isDisabled; // Visually disable the button underneath
  pathSelector.style.cursor = isDisabled ? 'not-allowed' : 'pointer';
}

function openHfTokenDialog() {
  return new Promise((resolve) => {
    // Overlay
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.inset = '0';
    overlay.style.background = 'rgba(0,0,0,0.35)';
    overlay.style.zIndex = '1000000';

    // Dialog
    const dlg = document.createElement('div');
    dlg.className = 'p-dialog p-component';
    dlg.style.position = 'fixed';
    dlg.style.top = '50%';
    dlg.style.left = '50%';
    dlg.style.transform = 'translate(-50%, -50%)';
    dlg.style.minWidth = '420px';
    dlg.style.background = 'var(--surface-card, #1e1e1e)';
    dlg.style.borderRadius = '8px';
    dlg.style.boxShadow = '0 10px 30px rgba(0,0,0,0.4)';
    dlg.style.zIndex = '1000001';
    dlg.setAttribute('role', 'dialog');
    dlg.setAttribute('aria-modal', 'true');

    dlg.innerHTML = `
      <div class="p-dialog-header p-3">
        <span class="p-dialog-title">Hugging Face Authentication</span>
      </div>
      <div class="p-dialog-content p-3" style="display:flex;flex-direction:column;gap:.75rem;">
        <div class="text-sm">Enter your Hugging Face access token. You can create one at
          <a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</a>.
        </div>
        <input type="password" class="p-inputtext p-component" placeholder="hf_xxx token" style="width:100%;" />
        <div class="p-message p-message-error" style="display:none"><span class="p-message-text"></span></div>
      </div>
      <div class="p-dialog-footer p-3" style="display:flex;justify-content:flex-end;gap:.5rem;">
        <button class="p-button p-component p-button-text" data-action="cancel">Cancel</button>
        <button class="p-button p-component" data-action="ok">Authenticate</button>
      </div>
    `;

    const input = dlg.querySelector('input');
    const msg = dlg.querySelector('.p-message');
    const msgText = dlg.querySelector('.p-message-text');
    const destroy = () => { document.body.removeChild(overlay); document.body.removeChild(dlg); };
    const submit = async () => {
      const val = (input.value || '').trim();
      if (!val) {
        msg.style.display = '';
        msgText.textContent = 'Token is required.';
        return;
      }
      destroy();
      resolve(val);
    };
    dlg.querySelector('[data-action="cancel"]').onclick = () => { destroy(); resolve(null); };
    dlg.querySelector('[data-action="ok"]').onclick = submit;
    input.onkeydown = (e) => { if (e.key === 'Enter') submit(); };

    document.body.appendChild(overlay);
    document.body.appendChild(dlg);
    input.focus();
  });
}

async function fetchExpected(url) {
  if (!url) return 0;
  try {
    const r = await fetch(`/models/expected_size?url=${encodeURIComponent(url)}`);
    const j = await r.json();
    return j.expected || 0;
  } catch {
    return 0;
  }
}

function startDownloadProgress({ directory, filename, url, expected, onComplete }) {
  try {
    createOrUpdateProgress({ directory, filename, url, expected, onComplete });
  } catch (e) {
    console.warn('[Model Installer] progress error', e);
  }
}

function updateDownloadExpected(key, expected) {
  try {
    const panel = document.getElementById('model-install-progress-panel');
    if (!panel) return;
    const row = panel.querySelector(`[data-key="${key}"]`);
    if (!row) return;
    row.dataset.expected = String(expected || 0);
  } catch {}
}

function createOrUpdateProgress({ directory, filename, url, expected, onComplete }) {
  let panel = document.getElementById('model-install-progress-panel');
  if (!panel) {
    panel = document.createElement('div');
    panel.id = 'model-install-progress-panel';
    panel.style.position = 'fixed';
    // Bottom-right with an opaque background so it doesn't bleed into the top bar / side panel
    panel.style.bottom = '16px';
    panel.style.right = '16px';
    panel.style.width = '320px';
    panel.style.maxHeight = '50vh';
    panel.style.overflowY = 'auto';
    panel.style.zIndex = '100000';
    panel.style.padding = '8px';
    panel.style.background = 'var(--comfy-menu-bg, #1e1e1e)';
    panel.style.color = 'var(--fg-color, #ddd)';
    panel.style.border = '1px solid var(--border-color, #444)';
    panel.style.borderRadius = '8px';
    panel.style.boxShadow = '0 8px 24px rgba(0,0,0,0.5)';
    panel.className = 'p-card p-component flex flex-col gap-2';
    document.body.appendChild(panel);
  }

  const key = `${directory}/${filename}`;
  let row = panel.querySelector(`[data-key="${key}"]`);
  if (!row) {
    row = document.createElement('div');
    row.dataset.key = key;
    row.className = 'p-card p-component';
    row.style.padding = '6px';
    row.style.background = 'transparent';
    row.style.boxShadow = 'none';
    row.innerHTML = `
      <div class="flex justify-between text-sm mb-1">
        <span style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${directory} / ${filename}">${directory} / ${filename}</span>
        <button class="p-button p-component p-button-text p-button-sm" data-action="close" style="flex-shrink:0;background:none;border:none;color:inherit;cursor:pointer">×</button>
      </div>
      <div class="p-progressbar p-component" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" style="height:8px;border-radius:4px;overflow:hidden;background:var(--comfy-input-bg, #333)">
        <div class="p-progressbar-value" style="width:0%;height:100%;background:var(--p-primary-color, #3b82f6);transition:width .3s"></div>
      </div>
      <div class="flex justify-between text-xs mt-1">
        <span class="sizes">0 MB / ? MB</span>
        <span class="percent">0%</span>
      </div>
      <div class="flex justify-between text-xs mt-1">
        <span class="eta">ETA —</span>
        <span class="rate">0 MB/s</span>
      </div>
    `;
    panel.appendChild(row);
    row.querySelector('button[data-action="close"]').onclick = () => {
      row.remove();
      // Clean up panel if no more downloads
      const panel = document.getElementById('model-install-progress-panel');
      if (panel && panel.querySelectorAll('[data-key]').length === 0) {
        panel.remove();
      }
    };
  }

  const progressValue = row.querySelector('.p-progressbar-value');
  const progressBar = row.querySelector('.p-progressbar');
  const rateEl = row.querySelector('.rate');
  const percentEl = row.querySelector('.percent');
  const sizesEl = row.querySelector('.sizes');
  const etaEl = row.querySelector('.eta');

  let lastBytes = 0;
  let lastTime = Date.now();

  const timer = setInterval(async () => {
    try {
      const qs = `directory=${encodeURIComponent(directory)}&filename=${encodeURIComponent(filename)}`;
      const r = await fetch(`/models/status?${qs}`);
      const j = await r.json();
      const bytes = j.size || 0;
      const now = Date.now();
      const dt = Math.max(1, now - lastTime) / 1000;
      const db = Math.max(0, bytes - lastBytes);
      const mbps = db / (1024 * 1024) / dt;
      lastBytes = bytes;
      lastTime = now;

      const dynamicExpected = Number(row.dataset.expected || '0') || 0;
      const total = dynamicExpected || expected || 0;
      const pct = total > 0 ? Math.min(100, Math.round((bytes / total) * 100)) : 0;
      progressValue.style.width = `${pct}%`;
      if (progressBar) progressBar.setAttribute('aria-valuenow', String(pct));

      rateEl.textContent = `${mbps.toFixed(2)} MB/s`;
      percentEl.textContent = `${pct}%`;

      // Sizes and ETA
      const toMB = (n) => (n / (1024 * 1024));
      const downloadedMB = toMB(bytes);
      const totalMB = total > 0 ? toMB(total) : 0;
      sizesEl.textContent = total > 0
        ? `${downloadedMB.toFixed(2)} MB / ${totalMB.toFixed(2)} MB`
        : `${downloadedMB.toFixed(2)} MB / ? MB`;

      if (total > 0 && mbps > 0) {
        const remainingBytes = Math.max(0, total - bytes);
        const remainingSec = remainingBytes / (mbps * 1024 * 1024);
        const mm = Math.floor(remainingSec / 60);
        const ss = Math.max(0, Math.floor(remainingSec % 60));
        etaEl.textContent = `ETA ${String(mm).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
      } else {
        etaEl.textContent = 'ETA —';
      }

      if (j.present && total > 0 && bytes >= total) {
        clearInterval(timer);
        progressValue.style.width = '100%';
        percentEl.textContent = '100%';
        setTimeout(() => {
          row.remove();
          // Clean up panel if no more downloads
          const panel = document.getElementById('model-install-progress-panel');
          if (panel && panel.querySelectorAll('[data-key]').length === 0) {
            panel.remove();
          }
        }, 2500);
        try { if (typeof onComplete === 'function') onComplete(); } catch {}
      }
    } catch (e) {
      console.warn('[Model Installer] progress poll error', e);
    }
  }, 2000);
}

function normalizeDownloadUrl(u) {
  if (!u) return '';
  try {
    const url = new URL(u);
    if (url.hostname.endsWith('huggingface.co')) {
      // Convert blob/tree links to resolve links
      if (url.pathname.includes('/blob/')) url.pathname = url.pathname.replace('/blob/', '/resolve/');
      if (url.pathname.includes('/tree/')) url.pathname = url.pathname.replace('/tree/', '/resolve/');
      // Add download hint when missing
      if (!url.searchParams.has('download')) url.searchParams.set('download', 'true');
      return url.toString();
    }
    return u;
  } catch {
    return u;
  }
}
