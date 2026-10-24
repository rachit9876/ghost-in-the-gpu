const dot = document.getElementById('dot');
const box = document.getElementById('infoBox');
const domainField = document.getElementById('domainField');
const domainSelect = document.getElementById('domainSelect');

function refreshStatus() {
  chrome.runtime.sendMessage({ action: "check_server_status" }, (res) => {
    if (res && res.live && res.data) {
      dot.classList.add('online');
      const d = res.data;
      const activeId = res.activeDomainId || 1;
      box.innerHTML = `Status: <b>Online (Port 8770)</b><br>Domain Images: <b>${d.logged_count || 0}</b><br>Brain: <b>${d.is_trained ? 'Trained & Active' : 'Untrained'}</b>`;

      if (d.domains && d.domains.length > 0) {
        domainField.style.display = 'block';
        domainSelect.innerHTML = '';
        for (const dom of d.domains) {
          const opt = document.createElement('option');
          opt.value = dom.id;
          opt.textContent = dom.name;
          if (dom.id === activeId) opt.selected = true;
          domainSelect.appendChild(opt);
        }
      }
    } else {
      dot.classList.remove('online');
      box.innerHTML = `Status: <b style="color:#FF4D6D">Offline</b><br><span style="font-size:10.5px;">Launch <code>START.bat</code> to start Ghost</span>`;
    }
  });
}

domainSelect.onchange = () => {
  const chosenId = parseInt(domainSelect.value, 10) || 1;
  chrome.storage.local.set({ activeDomainId: chosenId }, () => {
    refreshStatus();
  });
};

refreshStatus();
