# Laptop inference with llama.cpp

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

1. Set `LAPTOP_BASE_URL=http://127.0.0.1:8080` in `.env`, or save it in Settings.
2. Optionally configure existing permitted model directories. Scan/register files in Model Studio. Scanned files remain explicitly runtime-unverified.
3. Refresh the laptop catalogue in Model Studio and save the exact model ID returned by the server.
4. Click **Test real scenario inference**, then inspect progress, strict output checks and any runtime errors. A failed JSON response is preserved as diagnostic hash/size/errors; enable raw retention only if needed.
5. Build your request and run it from Scenario Builder.

Load/unload buttons are disabled because the generic single-model server protocol does not provide portable safe model lifecycle endpoints. To change the loaded file, explicitly restart/reconfigure your runtime and refresh its catalogue. `LAPTOP_API_KEY` can optionally authenticate a local runtime configured with a key. Runtime listing alone is not proof the selected model can generate usable output.
