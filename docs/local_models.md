# Laptop inference with llama.cpp

## Ollama fast response

For Ollama, use `http://127.0.0.1:11434`, select the exact installed model ID (for example `qwen3:1.7b`), and enable **Settings → Ollama fast response: disable thinking and request JSON**. The option defaults on for the standard Ollama port. It sends `reasoning_effort: none` and the pipeline's JSON schema as `response_format: {type: json_schema, json_schema: ...}` through the existing laptop adapter; ordinary prompts without a schema request JSON-object mode. Other server addresses keep the ordinary request unless explicitly enabled. Unsupported options fail visibly without silent fallback. The checkbox allows opting out.

The requested runtime options and provider-reported token usage are retained separately from requested/applied generation settings in attempts and reports. A successful request is not proof every requested setting was applied. Fast mode changes local reasoning behavior; include that difference when comparing providers.

Draft prompts retain full evidence descriptions and required links but omit risk/reviewer/search metadata; the complete evidence stays frozen for verification and engineering reports. For a quick CPU test, request 1-3 scenarios and 1024-1800 output tokens. More scenarios or longer answers take longer. Increasing the HTTP timeout only permits waiting; it does not speed up generation.

When retrieved evidence contains a complete linked group in the supported scope, the draft prompt schema requests at least one candidate. Empty output remains allowed when no such group is available. This prevents a small non-reasoning model from choosing an empty response to satisfy an unconstrained JSON grammar. Every candidate still undergoes independent evidence/text verification and can be rejected.

The scenario manager rejects a second laptop run while its first laptop request is active, avoiding accidental duplicate requests queuing on the same local runtime. Other backends can still run concurrently. Separate processes and external clients can also occupy Ollama; stop redundant jobs there before measuring speed.

On 18 October 2026, a live local Qwen3 `qwen3:1.7b` test on Ollama 0.40.6, running entirely on CPU, returned one KB-screened scenario in 26.671 seconds with 813 prompt tokens and 140 completion tokens. It used four retrieved camera-failure/rain records, a one-scenario limit, a 1024-token ceiling, requested temperature 0 / seed 42, no-thinking request and schema decoding. The sample KB remains illustrative; passing its screening is not engineering approval. This is one warm local test, not a representative benchmark or a like-for-like speedup against the earlier roughly seven-minute run. Longer batches, cold loads, retries and queued requests can take longer. Generation settings remain requested, with applied values unmeasured unless explicitly reported.

## llama.cpp setup

Install llama.cpp from its [official project](https://github.com/ggml-org/llama.cpp) using the release/build appropriate for your OS and hardware. The [server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) describes the OpenAI-compatible endpoints used here. No llama.cpp Python extension or model binary is bundled.

Obtain a compatible instruction-tuned GGUF yourself after checking the source, download size, available memory/storage and license. SafeScenario never downloads a model automatically. A GGUF extension alone does not establish chat-template compatibility, memory fit or JSON-generation quality.

Start your installed server (replace the paths with your actual executable and model):

```powershell
& 'C:\llama.cpp\llama-server.exe' -m 'C:\models\your-model.gguf' --host 127.0.0.1 --port 8080 -c 8192
```

```bash
./llama-server -m /opt/models/your-model.gguf --host 127.0.0.1 --port 8080 -c 8192
```

Use a context size the model and device can support. The output schema and evidence consume context; lower retrieval counts if necessary. The server must expose `GET /v1/models` and `POST /v1/chat/completions` with `model`, `messages`, `stream:false`, `temperature` and `max_tokens`, returning the selected model ID and text completion. Non-chat model endpoints are not compatible.

1. Set `LAPTOP_BASE_URL=http://127.0.0.1:8080` in `.env`, or save it in Settings. The default backend request timeout is 900 seconds. Set it directly in the Settings page, or use `SAFESCENARIO_BACKEND_TIMEOUT_SECONDS` in `.env`; values are bounded to 30-900 seconds.
2. Optionally configure existing permitted model directories. Scan/register files in Model Studio. Scanned files remain explicitly runtime-unverified.
3. Refresh the laptop catalogue in Model Studio and save the exact model ID returned by the server.
4. Click **Test real scenario inference**, then inspect progress, strict output checks and any runtime errors. A failed JSON response is preserved as diagnostic hash/size/errors; enable raw retention only if needed.
5. Build your request and run it from Scenario Builder.

Load/unload buttons are disabled because the generic single-model server protocol does not provide portable safe model lifecycle endpoints. To change the loaded file, explicitly restart/reconfigure your runtime and refresh its catalogue. `LAPTOP_API_KEY` can optionally authenticate a local runtime configured with a key. Runtime listing alone is not proof the selected model can generate usable output.
