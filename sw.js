const CACHE = 'apexscan-v27';

// App shell – alles was sich selten ändert
const SHELL = [
  '/ApexTradeHub/dashboard.html',
  '/ApexTradeHub/claude_picks.html',
  'https://cdn.jsdelivr.net/npm/chart.js'
];

// JSON-Daten – immer frisch versuchen, Fallback auf Cache
const DATA_FILES = [
  '/ApexTradeHub/apex_signals.json',
  '/ApexTradeHub/apex_equity_results.json',
  '/ApexTradeHub/apex_equity_top2.json',
  '/ApexTradeHub/apex_market.json',
  '/ApexTradeHub/apex_positions.json',
  '/ApexTradeHub/apex_open_positions.json',
  '/ApexTradeHub/apex_trade_log.json',
  '/ApexTradeHub/claude_picks.json'
];

// ── Install: Shell cachen ──
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

// ── Activate: alten Cache löschen ──
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// ── Fetch-Strategie ──
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // JSON + Claude's-Picks-Board: Network-first → Cache-fallback
  // (claude_picks.html wird haeufig iteriert → immer frisch, kein Versions-Bump noetig)
  const isData = DATA_FILES.some(f => e.request.url.includes(f.split('/').pop()));
  const isLiveHtml = e.request.url.includes('claude_picks.html');
  if (isData || isLiveHtml) {
    e.respondWith(
      fetch(e.request)
        .then(res => {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
          return res;
        })
        .catch(() => caches.match(e.request))
    );
    return;
  }

  // Alles andere: Cache-first → Network-fallback
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(res => {
        if (res && res.status === 200) {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return res;
      });
    })
  );
});