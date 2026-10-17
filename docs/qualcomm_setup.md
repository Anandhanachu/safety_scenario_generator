# Qualcomm Imagine / Cirrascale configuration

The [official live OpenAPI](https://aisuite.cirrascale.com/imagine-api-docs) was inspected on 17 October 2026. It identifies Qualcomm Imagine APIs, API version 1.0.0, server `/apis` and HTTP Bearer JWT authentication. This implementation uses:

| Operation | Documented contract |
| --- | --- |
| `GET https://aisuite.cirrascale.com/apis/v2/models` | Object mapping model-type strings to arrays of model-ID strings; optional `model_type` query not required here |
| `POST https://aisuite.cirrascale.com/apis/v2/chat/completions` | Required `model` and `messages`; roles are lower-case `system`, `user`, `assistant`, `tool`; supported `stream`, `temperature`, `max_tokens` |
| Completion response | `choices` with `delta.content` in the documented schema, plus `model`; adapter also accepts OpenAI-compatible `message.content` |

The catalogue includes model categories beyond chat. Select an account-authorized chat model. No model name, quota, free allowance or service availability is hardcoded.

1. Obtain authorized Imagine API access through the provider. Use the provider-issued Bearer token, not an invented or unrelated platform credential.
2. Set `QUALCOMM_API_KEY` in your local environment/ignored `.env`. Keep `QUALCOMM_BASE_URL=https://aisuite.cirrascale.com/apis` unless your account documents another compatible endpoint.
3. For a documented alternative, explicitly add its hostname to `QUALCOMM_ALLOWED_HOSTS` and configure its HTTPS base URL. The adapter still expects the documented v2 routes and response format.
4. Restart the app after environment changes. Settings shows whether the key is configured without displaying it.
5. In Model Studio, refresh Qualcomm models. This sends the real authenticated GET. No catalogue is shown if setup or the request fails.
6. Select a returned chat model, save its selection, then click **Test real scenario inference**. This sends an actual POST and may consume paid account quota.
7. Inspect the completion, model identity, timings and strict verification results. Listing and successful inference are displayed separately.

HTTP 401/403 report authentication/permission failures; 429 reports rate limiting; other status codes and timeouts report failures without echoing provider bodies or credentials. Redirects are rejected. Inference POST requests are not automatically retried because duplicate execution can incur costs. Cancel requests discard late output locally; they cannot cancel provider billing or guarantee server-side termination.

Credentials and account access were unavailable during development. Protocol behavior is covered by explicitly mocked tests, but live listing, inference and provider-specific operational behavior still need verification with your account.
