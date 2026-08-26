# STATO IMPLEMENTAZIONE (aggiornato)
Legenda: ✅ fatto · ⏳ aperto · ❌ testato e scartato

Fatte:
- ✅ §1 (1a/1b/1c) sicurezza read-only — denylist ricorsiva nel validator + nel runner (non disattivabile)
- ✅ §2 sessioni per-utente nel server (correlati CORS/auth/SSE-error: ancora aperti)
- ✅ §3a $limit su aggregate · ✅ §3b maxTimeMS · ✅ §3c cap su distinct
- ✅ §3e coercizione date schema-aware + fix timezone · ✅ §3f coercizione ObjectId (+ forma Extended JSON $oid)
- ✅ §3d roundtrip DataFrame → execute_query ritorna list[dict] (assente≠null, int≠float); validato replay +7/7 e A/B LLM +7/7 vs 0/7
- ✅ §6 describe_collection: errore + suggerimenti su nome inesistente
- ✅ **P1 text-memory retrieval loop** — `search_text()` ora ha call-site reale in `_prepare_turn()`; framing non-fidato; provenance minima (`source`/`verified`); config sicura; test + harness A/B in `examples/ab_text_memory/`

Testate e scartate (verifica a posteriori negativa):
- ❌ §8 join schema pre-loading (is_reference): implementata e SCARTATA — +0 accuratezza su 2 modelli (qwen3.6-27b e qwen3.5-9b), solo costo token in più

Aperte (prossime): §4a-e loop/error-handling · §5a-c memoria (auto-save unconfirmed, training threshold, multi-tenant) · §6 (contratto args nei tool) · §7 astrazione backend · §8 prompt · §10 osservabilità · correlati §2 · **P1-followup** (sotto)

---

## P1 — Text memory retrieval loop (chiusura del gap save→retrieve→prompt) — ✅ IMPLEMENTATO

### Diagnosi verificata
`MemoryService.search_text()` + `SaveTextMemoryTool` esistevano, ma `_prepare_turn()` recuperava solo training + tool-memory. Le note testuali non raggiungevano mai il prompt dinamico → glossario/domain knowledge morto a runtime. Collegato a §5c (text memory senza filtro) e §5 (poisoning/trust): iniettare testo arbitrario nel system prompt senza framing difensivo sarebbe un vettore di prompt-injection.

### Politica adottata
| source | verified default | retrieval |
|--------|------------------|-----------|
| `manual` / `import` | True | recuperabile |
| `llm` (`save_text_memory`) | False | recuperabile, mai trattata come verificata |
| `legacy` (metadata assente) | False | recuperabile, conservativo |

Tutte le note iniettate usano framing **reference data, not instructions** con delimitatori `<<<DOMAIN_NOTE…>>>` / `<<<END_DOMAIN_NOTE>>>`. Lo score è etichettato `retrieval_score`, non confidence. Config default: `enable_text_memory=True`, `top_k=2`, soglia `0.55` (calibrata su MiniLM: a `0.75` i match Q↔definition tipici ~0.49–0.67 non passavano → retrieval morto), cap 500/1200 char, `text_memory_include_unverified=True`. Errori backend → warning + turno prosegue. `auto_save_memory=False` supporta A/B frozen.

### Follow-up aperti (non in P1)
- **P1b / §5a:** escludere di default le note `llm` non verificate (`include_unverified=False` come default futuro, dopo A/B).
- **P1c / §5c:** namespace text-memory per fingerprint DB (oggi condivise cross-tenant come le altre).
- **P1d / §5:** promozione `verified` via feedback utente / endpoint dedicato.
- Non migrare distruttivamente i record legacy.

### Relazione con §5
P1 chiude il loop di retrieval ma **non** risolve self-poisoning (§5a) né la soglia permissiva dei training (§5b). Il framing non-fidato mitiga il rischio di trattare note (incluse LLM-generated) come istruzioni — difesa necessaria una volta che le note entrano nel prompt.

### Verifica
Test: `tests/test_text_memory_retrieval.py` (+ smoke in `test_prompt_builder.py`). A/B: `examples/ab_text_memory/` (glossario congelato + harness; richiede credenziali DEV).

---

✅ FATTO — 1. SICUREZZA — read-only guarantee: BUCATA (bloccante)
Auto-clarity: qui esco dal modo compresso perché è materiale di sicurezza.

1a. $out e $merge sono esplicitamente nell'allowlist delle pipeline stage
In validator.py:57-65, _PIPELINE_STAGES contiene "$merge" e "$out". L'allowlist delle operazioni (find/aggregate/count/distinct in mongodb.py:31) dà una garanzia solo nominale: aggregate è permesso, e una pipeline [{"$match": ...}, {"$out": "victims"}] passa la validazione e viene eseguita da _execute_aggregate, che fa collection.aggregate(pipeline) senza ispezionare le stage. Risultato: l'agente "read-only" può scrivere e sovrascrivere collezioni intere ($out rimpiazza la collezione target). $merge può addirittura scrivere in un altro database.

Il vettore non è teorico: l'input è linguaggio naturale non fidato. "Copia i risultati nella collezione X" o un prompt-injection dentro un documento del DB (i sample documents finiscono nel contesto!) bastano perché l'LLM generi la stage. Il commento in mongodb.py — "No write operations are possible" — è falso.

Per ironia, _describe_stage in explain_query sa descrivere $out/$merge come "Write results to..." — il codice conosce la semantica di scrittura ma non la blocca.

1b. Esecuzione JavaScript server-side permessa
_FILTER_OPERATORS include "$where" (validator.py:43); _ACCUMULATOR_OPERATORS include "$accumulator" e "$function" (validator.py:73). Tutti e tre eseguono JS arbitrario su mongod (abilitato di default). Vettore di DoS (while(true){}), e amplia enormemente la superficie di attacco rispetto a MQL dichiarativo.

1c. Stage amministrative/diagnostiche permesse
$collStats, $indexStats, $planCacheStats, $listSessions, $listLocalSessions, $changeStream sono in allowlist. $changeStream apre un cursore infinito (hang del thread); le altre leakano metadati operativi. $unionWith e $lookup permettono lettura di qualunque collezione del DB anche se in futuro volessi ACL per-collezione.

Fix proposto
In MQLValidator, aggiungi denylist hard (errore, non warning) per: $out, $merge, $where, $function, $accumulator, $changeStream, $listSessions, $listLocalSessions, $planCacheStats. La ricorsione di _collect_unknown_operators esiste già — riusala per cercare operatori vietati a qualunque profondità (serve: $function può annidarsi in $expr dentro $match, in $project, in $facet, in sub-pipeline di $lookup).
Difesa in profondità in _execute_aggregate: stesso check nel runner, indipendente dal flag validate=True di RunMQLTool (oggi validate=False disattiva TUTTO, incluso il check $out). La garanzia read-only non può essere opzionale.
Documenta (README + connect()): usare utente MongoDB con ruolo read — unica garanzia vera. Il codice applicativo è il secondo anello, non il primo.
Trade-off: perdi $where per query legittime rarissime (ricostruibili con $expr); $collStats/$indexStats potrebbero servire a un futuro tool di ottimizzazione — esponili come tool dedicato, non via pipeline. Complessità: bassa. Priorità: bloccante.

✅ FATTO (sessioni) / ⏳ correlati aperti — 2. Server FastAPI — sessioni finte, stato condiviso (bloccante)
Torno terso.

routes.py:54-60: session_id generato e restituito al client, ma mai usato. Tutti i client parlano con lo stesso app.state.agent, quindi stessa _conversation. Conseguenze:

Leak cross-utente: utente B vede contesto (domande, dati) di utente A nella history. Con max_turns=5 la conversazione di A condiziona le risposte a B.
Race: due richieste concorrenti fanno append interleaved su self._conversation (agent.py:414, 544, 616) → history con tool_use orfani → 400 dal provider, o risposte incrociate.
new_session() (agent.py:198) esiste ed è cheap — ma il server non lo chiama.
Fix: mappa session_id → agent (dict con TTL/LRU) in app.state; new_session() per sessioni nuove; lock per-sessione per serializzare richieste sulla stessa sessione. Trade-off: gestione eviction (memoria per sessioni abbandonate). Bloccante per qualunque deploy multi-utente.

Correlati (importanti, stesso file):

CORS allow_origins=["*"] + allow_credentials=True (main.py:67-73): combinazione vietata dalla spec CORS (browser la rifiuta) e comunque insicura come default.
Zero autenticazione. /memory/import e /memory/train aperti = poisoning remoto della memoria (chiunque inietta training entry che il prompt tratta come verità: vedi §5). Endpoint di scrittura memoria senza auth è un'escalation diretta.
str(exc) degli errori interni sparato nel SSE (routes.py:63) — può leakare path, nomi host del DB.
3. run_mql — vincoli mancanti, cap dei risultati bugato (bloccante/importante)
✅ FATTO — 3a. limit non applicato ad aggregate
mongo_tools.py:392 calcola il cap, ma _execute_aggregate ignora req.limit. Pipeline senza $limit (es. $match largo + $project) → intero result set in RAM come DataFrame → serializzato con indent=2 → nel contesto LLM. La descrizione del tool dichiara "Results are capped at 100 rows by default": falso per aggregate, che è l'operazione consigliata dal prompt stesso ("Prefer aggregate pipelines"). Milioni di righe = OOM del processo o esplosione di token/costi. Fix: appendi {"$limit": cap} se la pipeline non termina già con $limit/$count/$sortByCount più restrittivo. Bloccante (DoS accidentale banale).

✅ FATTO — 3b. Nessun maxTimeMS
Nessuna query ha timeout server-side. count_documents senza filtro su collezione enorme, $group full-scan, $graphLookup profondo: thread bloccato per minuti, evento SSE muto, mongod sotto carico. Fix: maxTimeMS configurabile (default es. 30s) su find/aggregate/count/distinct in MongoRunner. Costo: quasi zero. Importante.

✅ FATTO — 3c. distinct senza cap
_execute_distinct (mongodb.py:146) su campo ad alta cardinalità → array gigante (e limite 16MB BSON con errore criptico). Meglio pipeline $group + $limit interno.

✅ FATTO — 3d. Roundtrip DataFrame lossy (design, importante)
[Implementato: execute_query ritorna list[dict] (via _docs_to_rows/_json_safe), niente pandas nel percorso caldo; ABC NoSQLRunner aggiornato (rottura accettata, Mango → Mongo-only). Validato: replay offline deterministico NEW 7/7 vs OLD 0/7, e A/B LLM end-to-end ON 7/7 vs OFF 0/7 su un subset di find eterogenei. Gold del benchmark verificato pulito → ship sicuro. _docs_to_dataframe tenuto come utility opzionale.]
docs → DataFrame → to_json → json.loads (mongodb.py:357-360, mongo_tools.py:420). Problemi teorici concreti:

Documenti eterogenei → unione colonne, campi mancanti diventano null (l'LLM non distingue più "campo assente" da "campo null" — semanticamente diverso in Mongo).
Colonna int con un solo mancante → float (42 → 42.0 nella risposta all'utente).
pandas non serve a niente qui: i tool consumano JSON records. Il DataFrame esiste solo come tassa di conversione.
Fix: execute_query ritorna list[dict] (bson → JSON-safe diretto); pandas opzionale per l'API pubblica Python se vuoi tenerla. Trade-off: cambio dell'interfaccia NoSQLRunner (breaking per backend terzi — ma il progetto è giovane). Importante.

✅ FATTO — 3e. _coerce_dates troppo aggressiva + bug timezone (importante)
[Implementata coercizione schema-aware: converte solo se il campo ha tipo univoco date; fix timezone (astimezone UTC); rimossa la vecchia _coerce_dates cieca. Validata con replay ablation: +8/10 causale.]
mongo_tools.py:555-605:

Contraddice il prompt: la regola in prompt_builder.py:82-84 dice "le date possono essere stringhe, matcha il formato esatto". Ma se l'LLM obbedisce e genera {"created": {"$gte": "2024-11-22T08:29:28.225"}} su campo stringa, _coerce_dates lo converte in datetime → il confronto stringa-vs-date non matcha nulla → zero risultati, silenziosamente. Il layer Python sabota l'istruzione data all'LLM. Contraddizione di design pura.
Bug timezone: dt.replace(tzinfo=None) (mongo_tools.py:602) tronca l'offset invece di convertire a UTC: 2024-01-01T10:00:00+02:00 → naive 10:00 interpretata come UTC = errore di 2 ore. Fix: dt.astimezone(timezone.utc).replace(tzinfo=None).
Converte QUALUNQUE stringa ISO-like ovunque: un filtro su campo version: "2024-01-01" (stringa legittima) si rompe. E un pattern $regex che somiglia a una data verrebbe convertito.
Fix di design: la coercizione deve essere schema-aware — converti solo se il campo target ha tipo date nello schema introspettato (l'informazione c'è già in FieldInfo.types). Fallback: coercizione attuale solo per {"$date": ...} (Extended JSON, intento esplicito). Trade-off: serve risolvere il path del campo dal filtro (nesting, $and/$or) — complessità media ma è l'unico modo corretto. Importante.

✅ FATTO — 3f. ObjectId mai coercizzato
[Coercizione ObjectId schema-aware: hex 24-char su campo di tipo ObjectId → ObjectId(...); gestita anche la forma Extended JSON {"$oid": "..."} annidata in $in.]
_stringify_bson mostra _id come stringa hex all'LLM; se l'utente poi chiede "dammi il documento con id X", l'LLM genera {"_id": "68a3..."} (stringa) → zero risultati, nessun errore. Serve coercizione simmetrica: stringa 24-hex su campo di tipo ObjectId → ObjectId(...). Stesso principio schema-aware del punto sopra. Importante.

4. Loop dell'agente — error handling e integrità della conversazione
✅ FATTO — 4a. _is_retryable è substring-matching fragile (importante, bug concreto)
[Implementato: ToolResult.error_kind = type(exc).__name__ marcato nel registry; runner splitta ConnectionFailure→BackendError (fatale) vs PyMongoError→QueryError (retryable, include timeout); _is_retryable classifica per _FATAL_ERROR_KINDS invece che substring. Validation-error tornano con kind=None → retryable (path "did you mean" recuperato). Fix collaterale: timeout maxTimeMS (§3b) ora retryable→LLM semplifica invece di arrendersi. Test: +2 (regression network_events + registry stamping), suite 36/36 verde.]
agent.py:41-56: pattern come "network", "ssl", "timed out" cercati nel testo dell'errore. Ma l'errore del validator contiene il nome della collezione: "Collection 'network_events' does not exist. Did you mean...?" → contiene "network" → classificato FATALE, l'LLM riceve "Do not retry" e il suggerimento Did you mean diventa inutilizzabile. Stessa cosa per collezioni/campi con ssl, certificate. E "timed out" è spesso retryable (query troppo pesante → l'LLM può semplificarla). Classificare su testo libero è intrinsecamente fragile.

Fix: classifica per tipo di eccezione, non per stringa: ValidationError/QueryError → retryable; BackendError/ConnectionFailure/AutoReconnect → fatale. Serve che ToolResult porti una error_kind (il registry oggi appiattisce tutto a str(exc) in base.py:131). Complessità bassa. Importante.

4b. Crash o disconnect ⇒ conversazione corrotta permanentemente
_prepare_turn appende il messaggio user prima della chiamata LLM (agent.py:414); il loop appende assistant con tool_use prima di eseguire i tool (agent.py:544). Se self._llm.chat lancia (rate limit, 529), o il client SSE si disconnette a metà (il generator di ask_stream viene abbandonato), la history resta con un tool_use senza tool_result → ogni chiamata successiva della sessione fallisce con 400 dal provider. Nessun try/finally ripara lo stato. Fix: costruisci i messaggi del turno in una lista transazionale, committa in _conversation solo a turno completato; oppure repair-pass che chiude i tool_use orfani con tool_result sintetico "interrupted". Importante (in produzione SSE i disconnect sono la norma).

4c. Chiamata LLM sincrona dentro codice async
self._llm.chat(...) (agent.py:504) è bloccante (SDK sync) dentro async def → blocca l'event loop per l'intera latenza LLM (secondi). Nel server: tutte le altre richieste SSE congelate. Fix minimo: await asyncio.to_thread(self._llm.chat, ...). Fix vero: client async nei provider. Importante per il server, irrilevante per CLI.

4d. Tool call parallele rompono l'adapter Anthropic
Il loop appende un Message(role="tool") separato per ogni tool call (agent.py:616); _to_anthropic_messages li traduce in messaggi user separati. L'API Anthropic esige che tutti i tool_result di un turno assistant stiano nel singolo messaggio user immediatamente successivo. Se Claude emette 2+ tool_use in parallelo (lo fa spesso), la richiesta successiva → 400. Oggi mascherato perché i tool sono per lo più chiamati in sequenza, ma è una bomba a orologeria. Fix: merge dei messaggi tool consecutivi in un unico messaggio user multi-block nell'adapter. Importante.

4e. Semantica retry confusa (nice-to-have)
retry_count è globale per il turno e si resetta su qualunque successo (agent.py:587): fallisci run_mql, chiami describe (successo, reset), rifallisci run_mql → contatore sempre a 1, il budget vero è max_iterations. Inoltre _retry_message inietta sempre le stesse due regole hardcoded ("$ mancante", "quote extra") anche quando l'errore non c'entra — rumore che diluisce l'errore reale. Meglio: contatore per (tool, firma errore) e messaggio di retry che riporta solo l'errore.

5. Memoria ChromaDB — poisoning by design (importante)
5a. Auto-save senza verifica = self-poisoning
agent.py:579-586: ogni run_mql che non lancia eccezioni diventa pending_memory e viene persistito a fine turno. "Successo" = "MongoDB non ha dato errore", non "risposta corretta". Una query sbagliata-ma-valida (filtro sul campo sbagliato, 0 righe, join errato) entra in memoria e viene ripresentata come "Similar past interactions" alle domande future. Il sistema rinforza i propri errori: è un ciclo di feedback positivo senza segnale di verità. DeleteLastMemoryEntryTool mitiga solo se l'utente si accorge E lo dice.

Fix: (a) non auto-salvare risultati a 0 righe (già sospetti); (b) stato unconfirmed di default, promozione a confirmed su feedback utente esplicito o endpoint dedicato; peso/threshold più alto per unconfirmed in retrieval; (c) TTL o cap con eviction LRU. Trade-off: memoria cresce più lentamente — è il prezzo della correttezza.

5b. Training entries: istruzione di fiducia cieca con soglia 0.5
agent.py:427-429: "use these directly without additional exploration... call the tool with those exact args". Ma get_training_entries usa similarity_threshold=0.5 (chromadb.py:313) — a 0.5 di cosine con MiniLM ci matcha quasi tutto. "Quanti ristoranti a Brooklyn?" può pescare "Quanti ristoranti a Manhattan?" e l'istruzione dice di copiare gli args esatti → risposta sbagliata con confidenza massima. Combinazione peggiore possibile: retrieval permissivo + istruzione imperativa. Fix: soglia ≥0.85 per il "copy exact args"; sotto, presenta come "reference, adapt as needed". Inoltre: le similarity mostrate all'LLM non ci sono (il prompt non le include per i training) — includerle aiuterebbe la calibrazione.

5c. Multi-tenancy assente
Nome collezione Chroma fisso "mango_memory" (chromadb.py:121), nessun namespace per database/tenant. Due agenti su DB diversi con lo stesso persist_dir si condividono le memorie. Il filtro post-hoc in agent.py:441-452 (droppa entry con collection ignota) mitiga per le tool-memory ma: (a) è per nome collezione — DB diversi con collezioni omonime passano; (b) le text memory non hanno filtro alcuno. Fix: deriva il collection name da un fingerprint del DB (nome + host), o parametro tenant obbligatorio. Bassa complessità.

5d. Retrieval: parti solide
Il re-ranking DAIL-SQL-style (tag strutturali + Jaccard, α=0.7, over-fetch 3×) in chromadb.py:164-218 è ben fatto — idea giusta, implementazione pulita, threshold sulla similarità semantica prima del re-rank. Anche il filtro schema-consistency delle memory in _prepare_turn è un tocco raro e giusto. Nota minore: _QUESTION_TAG_MAP e lo stemmer sono English-only, mentre il prompt promette risposte nella lingua dell'utente — domanda in italiano ("quanti", "per ogni") → zero tag → re-rank degrada a solo-semantico, e _select_relevant_collections perde i match lessicali. Coerenza linguistica da risolvere (embedding MiniLM multilingue scarso, valuta paraphrase-multilingual-MiniLM).

6. Tool design e schemi (misto)
Solido: la suite dei tool è ben pensata. inspect_field è un ottimo tool con descrizione esemplare (dice cosa NON è — "DIAGNOSTIC only, not the answer"). run_mql unico tool con operation enum invece di 4 tool separati: scelta giusta, meno ambiguità di dispatch. explain_query con trigger phrases esplicite: bene. Il grouping UUID/prefissi in list_collections per DB grandi: bene.

Difetti:

✅ FATTO — describe_collection su collezione inesistente ritorna successo con schema vuoto (mongo_tools.py:150-153 → _introspect_collection non valida l'esistenza). L'LLM riceve {fields: [], document_count: 0} e conclude "collezione vuota" invece di "nome sbagliato". Il validator di run_mql suggerisce nomi simili; describe no. Fix: check esistenza + difflib suggestions come nel validator. Importante (il prompt ordina di chiamare describe per primo: è la prima trappola che l'agente incontra). [Implementato: check esistenza + difflib.get_close_matches. Cache TTL introspezione: ✅ FATTA — vedi bullet sotto.]
✅ FATTO — describe_collection ri-campiona 100 documenti a ogni chiamata, nessuna cache, e accede al privato _backend._introspect_collection (fuori dall'ABC NoSQLRunner — vedi §7). inspected_collections è locale a _run_loop (agent.py:498) quindi si resetta a ogni domanda: stessa collezione ri-descritta ogni turno. Cache con TTL nel runner: banale, alto ROI. [Implementato: cache TTL in MongoRunner (_introspect_cache, introspect_ttl_s default 300s; 0 = disabilitata). _introspect_collection controlla la cache prima di campionare, salva dopo. Evita ri-campionamento 100-doc + ri-lettura index a ogni describe della stessa collezione tra domande diverse (inspected_collections si resetta per domanda, ma il MongoRunner persiste). Guadagno = latenza DB + compute, NON token (il payload all'LLM è identico cache o no). Accuracy: non benchmarkata, ragionata neutra (stessa info schema; unica differenza: sample_documents congelati invece che ri-randomizzati da $sample → più deterministico). Test: +3 (cache-hit per identità, TTL-expiry, ttl=0 bypass), suite 406/406.]
filter/projection/sort dichiarati come object nudo senza descrizione della sintassi attesa (Extended JSON? ISO date? come esprimere $gte su date?). Il modello lo sa per training, ma l'unica documentazione della coercizione date (§3e) non esiste da nessuna parte nello schema del tool — l'LLM non sa che le stringhe ISO verranno convertite. Documentare il contratto nel description.
ToolParam non supporta additionalProperties: false né descrizioni per proprietà annidate — limite accettabile ora, ma sappi che stai rinunciando allo strumento più efficace per vincolare gli args.
kwargs["operation"] / kwargs["pattern"] con KeyError se l'LLM omette il parametro → il registry cattura e ritorna error="'operation'" — messaggio criptico per il retry. Valida i required con messaggio esplicito.
7. NoSQLRunner e ToolRegistry — estensibilità
L'ABC NoSQLRunner dichiara portabilità multi-backend ("MongoDB, Redis, Cassandra...") ma l'astrazione è già rotta: QueryRequest.pipeline è MQL puro, MQLValidator è Mongo-only, i tool importano MongoRunner concreto (mongo_tools.py:38), e tre tool accedono a privati (_introspect_collection, _database) bypassando l'ABC (mongo_tools.py:153, 291, 748). Opinione diretta: l'astrazione multi-backend oggi è finzione che costa (pandas nel contratto, metodi che non generalizzano a Redis). O la onori — aggiungi profile_field, explain, introspect_collection(name) all'ABC e togli gli accessi privati — o dichiari Mongo-only e semplifichi. La seconda è più onesta per un progetto chiamato "mango". Nice-to-have ma decide la direzione del progetto.
ToolRegistry (base.py:87-137): minimale e corretto. execute che ritorna errore invece di lanciare per tool ignoti: giusto per il loop LLM. Ok così.
✅ FATTO — ToolResult.as_text con indent=2 (base.py:48): su 100 righe di risultato raddoppia i token per pura estetica che l'LLM non ripaga. Usa compact separators. Facile, risparmio reale. [Implementato: separators=(",",":"). Misurato -36.5% char su payload 100-righe tipico; test format-agnostici (json.loads) verdi 14/14.]
8. Prompt engineering
Buona base (prompt_builder.py): regole comportamentali concrete (verifica plausibilità post-query, inspect_field prima dei filtri categorici), iniezione schema per-query con selezione di rilevanza, datetime corrente nel prompt dinamico, split cacheable/dynamic ben progettato per il caching Anthropic (agent.py:474-477).

Problemi:

Contraddizioni interne: "ALWAYS call describe_collection before writing a query" (regola statica) vs "Do NOT call describe_collection when a training example covers the question" (sezione dinamica) vs auto-schema injection che rende describe ridondante (agent.py:559-571). Tre meccanismi sovrapposti con istruzioni conflittuali; l'LLM risolve il conflitto in modo non deterministico. Gerarchia esplicita da scrivere ("regola X vince su Y") o rimozione della regola statica visto che l'auto-injection esiste.
La regola date (§3e) è sabotata dal codice. Una delle due deve cambiare.
"If the question is ambiguous, ask one clarifying question": senza definire ambiguità, o scatta sempre o mai. E "retry once" nella regola vs max_retries=2 nel codice: incoerenza numerica minore ma gratuita.
Difesa write-op affidata a una sola riga ("NEVER perform write operations") — va bene come defense-in-depth, ma oggi è l'unico strato per $out (§1). Dopo il fix del validator diventa accettabile.
❌ SCARTATO (l'idea "includere le collezioni referenziate") — _select_relevant_collections (agent.py:342-403): euristica lessicale con stemming — onesta e deterministica, i tie-break alfabetici sono un bel dettaglio. Ma schema_top_k=3 fisso: domanda con $lookup su 4 collezioni ne perde una; il fallback auto-schema arriva solo dopo che l'LLM ha già scritto la pipeline con nomi di campo indovinati. Considera: includere sempre le collezioni referenziate (is_reference) da quelle selezionate — l'informazione è già in FieldInfo.reference_collection. [TESTATO E SCARTATO: implementato _expand_with_references + A/B su subset join dedicato. Risultato: +0 accuratezza su qwen3.6-27b e qwen3.5-9b; describe_collection mai chiamato (nessun round-trip da risparmiare); +~270 token input/query. Il modello indovina già i nomi comuni (tier/country/brand/status). Revert completo.]
✅ FATTO — 9. Contesto multi-turno
_prune_conversation (agent.py:641-656) rimuove turni interi — corretto rispetto al formato API, bene. Ma dentro i 5 turni tenuti, i tool_result restano integrali: 5 turni × più run_mql × 100 righe JSON indentato = decine di kToken trascinati a ogni chiamata, per informazione che l'assistente ha già riassunto nella risposta. Fix standard: dopo che il turno è concluso, comprimi i tool_result storici a un summary (row_count + prime 3 righe). Trade-off: i follow-up "e la seconda riga di prima?" perdono i dati grezzi — accettabile, l'agente può ri-eseguire la query. Importante per costi/latenza in conversazioni reali. [Implementato: _compact_historical_tool_results chiamato a inizio _prepare_turn (a quel punto tutti i tool message sono di turni conclusi); riscrive SOLO il content (tool_call_id intatto → nessun 400), a row_count + 3 sample rows via _summarize_tool_result (raw_decode del JSON, fallback truncation per payload non-rows). Idempotente (marker _compacted + soglia 600 char). Misurato -94.5% char sul contesto storico (~-3.8k token a 2 turni, ~-9.6k a 5). NON visibile nel bench single-turn. Test: +4, suite verde.]

10. Osservabilità
Stato: logging module, token count aggregati in AgentResponse, eventi tipizzati nello stream, callback on_tool_call. Sufficiente per debug locale, insufficiente per produzione:

Nessun correlation ID: impossibile ricostruire "domanda → retrieval hits → prompt → tool calls → retry → risposta" da log interleaved di più sessioni. Fix: turn_id generato in _prepare_turn, incluso in ogni log record ed evento.
Le decisioni di retrieval sono invisibili: quali memory entries sono state iniettate, con che score, quali collezioni selezionate da _select_relevant_collections e con che punteggio. Quando l'agente sbaglia per colpa di una memoria avvelenata non hai modo di saperlo. Fix: evento context nello stream/log con {memory_ids+scores, collections+scores, training_ids}.
logger.info("Tool call: %s(%s)") (agent.py:550) logga gli args integrali → filtri con PII nei log a livello INFO. Va reso opt-in.
Nice-to-have: hook OpenTelemetry opzionale. Non prima del resto.
Piano d'azione prioritizzato
Fase 1 — Sicurezza (bloccante, ~1 giorno):

✅ Denylist ricorsiva $out/$merge/$where/$function/$accumulator/$changeStream/$list*/$planCacheStats nel validator e nel runner (non disattivabile). Test: pipeline con stage vietata a ogni profondità di annidamento ($facet, $lookup sub-pipeline, $expr).
⏳ README: raccomandazione utente Mongo con ruolo read.
⏳ Auth minima sul server FastAPI (API key header) + fix CORS + niente str(exc) raw nel SSE.
Fase 2 — Correttezza server e loop (bloccante, ~2-3 giorni):
4. ✅ Sessioni reali: mappa session_id → agent.new_session() con lock per-sessione ed eviction.
5. ✅ $limit iniettato nelle aggregate + maxTimeMS ovunque + cap su distinct.
6. ⏳ Integrità conversazione: commit transazionale del turno / repair dei tool_use orfani.
7. ⏳ Merge dei tool_result consecutivi nell'adapter Anthropic (tool call parallele).
8. ⏳ asyncio.to_thread attorno a llm.chat.

Fase 3 — Robustezza query (importante, ~3 giorni):
9. ✅ Classificazione errori per tipo (via error_kind in ToolResult), non per substring.
10. ✅ Coercizione date schema-aware + fix timezone; coercizione ObjectId simmetrica; [contratto documentato nella description di run_mql: ancora ⏳].
11. ✅ describe_collection: errore + suggerimenti su collezione inesistente; ✅ cache introspezione con TTL.
12. ✅ Eliminare roundtrip DataFrame (o almeno preservare assenza-vs-null).

Fase 4 — Memoria e prompt (importante, ~2 giorni):
13. Stato confirmed/unconfirmed per auto-save; skip auto-save su 0 righe; threshold training ≥0.85 per il "use exact args", altrimenti wording "reference".
14. Namespace memoria per fingerprint DB.
15. Risolvere le contraddizioni del prompt (describe vs training vs auto-schema; regola date vs coercizione).
16. ✅ Compattazione tool_result nei turni storici; ✅ as_text senza indent.

Fase 5 — Osservabilità e pulizia (nice-to-have):
17. turn_id + evento context con score di retrieval.
18. Decisione strategica NoSQLRunner: onorare l'ABC o dichiararsi Mongo-only.
19. Multilingua: tag map e stemmer, embedding multilingue.

Verifica a posteriori (esplicitamente dopo l'implementazione, mai come giustificazione): fase 1-2 si verificano con test unitari/integration mirati (pipeline malevole, sessioni concorrenti, disconnect SSE) — mango-bench non c'entra. Per le fasi 3-4 (coercizione date, memoria, prompt), dopo il merge ha senso rigirare mango-bench come sanity check che le ipotesi di design non abbiano regredito l'accuracy — se una metrica scende, si indaga l'ipotesi, non si ritocca la modifica per inseguire il numero.

Cosa è già solido (nessun intervento): re-ranking strutturale del retrieval, split cacheable del system prompt, design a tool unico run_mql con enum, inspect_field, pruning per turni interi, grouping collezioni per DB grandi, filtro schema-consistency sulle memorie, tie-break deterministici nella selezione collezioni.