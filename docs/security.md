# Security boundaries

Run the Streamlit dashboard on loopback. It has no multi-user authentication or authorization and is intended for a trusted local user. Do not change its bind address to a public/shared network without designing authentication and transport protection.

The optional application API and UNO Q operational endpoints fail closed without environment-configured Bearer tokens. Tokens are compared with constant-time comparison. HTTP request bodies are capped before parsing; KB imports, schema lists, strings and generated outputs have separate limits. Worker queues, file inventories, agent logs and retained agent tasks are bounded.

Secrets stay in the process environment or ignored `.env`, not preferences, history, generated reports or normal logs. Provider error bodies are deliberately omitted. Do not paste credentials into contextual notes or KB content: those fields are engineering data sent to your explicitly selected inference provider. Raw-response retention is off by default. Enabling it can retain sensitive source/model text locally.

Backend URLs are explicitly configured by the operator, never derived from model or KB instructions. Cloud endpoints use HTTPS and a hostname allowlist. Local inference uses loopback. Agent addresses must resolve privately; unspecified, multicast and link-local destinations are blocked. Redirects and implicit environment proxies are disabled. DNS validation is a guardrail, not complete protection against a malicious DNS server/rebinding; use trusted local DNS and host configuration. TLS certificate validation remains enabled for HTTPS.

Retrieval inserts content as JSON data in the user message, under an explicit system instruction to ignore embedded commands. Prompt injection can still affect generated prose; strict schema/reference checks and human review are required. Generated content never changes settings, executes code, downloads files, loads firmware or triggers MCU calls. Source links are displayed as text, not automatically fetched. CSV exports prefix dangerous formula-leading text.

Model files are registered only inside operator-permitted existing directories; resolved symlinks cannot escape those roots. GGUF files are not opened or executed by the dashboard. The separately operated model runtime has its own security responsibility.

The UNO Q agent offers no shell endpoint. Its optional MCU allowlist contains only a capability handshake and read-only ping. The server rechecks firmware support and confirmation on every test. No actuators are implemented. No remote installation, networking change, public exposure or firmware upload is performed automatically.

SQLite files contain engineering content and human attribution. Back up/export them under your retention policy. Confirmed history clearing deletes local records permanently, retaining dataset versions/settings. SQLite deletion does not guarantee forensic erasure from backups or storage media. Active tasks are process-local and are not crash-resumable.
