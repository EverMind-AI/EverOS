# EverOS demo relay on Netlify

This project deploys the public `everos demo` relay as a Netlify Function. The
client sends no credentials. The function accepts only the demo API surface,
injects the platform key stored in Netlify, applies distributed per-IP quotas
with Upstash Redis, and forwards the request to EverOS Cloud.

## Required Netlify environment variables

Configure these under **Project configuration -> Environment variables**. Never
put their values in this repository.

| Variable | Required | Default |
| --- | --- | --- |
| `EVEROS_CLOUD_API_KEY` | yes | none |
| `UPSTASH_REDIS_REST_URL` | yes | none |
| `UPSTASH_REDIS_REST_TOKEN` | yes | none |
| `EVEROS_CLOUD_UPSTREAM` | no | `https://api.evermind.ai` |
| `RELAY_RATE_PER_MIN` | no | `30` |
| `RELAY_DAILY_ROUNDS` | no | `3` |
| `RELAY_UPSTREAM_TIMEOUT_MS` | no | `20000` |
| `RELAY_MAX_BODY_BYTES` | no | `1000000` |

The per-minute limit counts every relay request. The daily limit counts only
`POST /api/v2/memory/add`, which starts a demo round; flush and search requests
do not spend additional rounds. The quota service fails closed:
demo API requests return `503` when Redis is not configured or unavailable.
`/healthz` remains available for diagnostics and reports only whether secrets
are configured, never their values.

## Deploy from GitHub

1. In Netlify, choose **Add new project -> Import an existing project -> GitHub**.
2. Select the EverOS fork and the `feat/demo-cloud-interactive` branch.
3. Set **Base directory** to `deploy/netlify_relay`.
4. Leave the build command empty. `netlify.toml` supplies the publish and
   functions directories.
5. Add the required environment variables and deploy.
6. Verify `https://<site>.netlify.app/healthz` reports both configuration flags
   as `true`.
7. Test the client with
   `EVEROS_CLOUD_DEMO_URL=https://<site>.netlify.app everos demo`.
8. After validation, add the production custom domain.

## Local checks

```bash
cd deploy/netlify_relay
npm run check
```

The test suite is credential-free and mocks both Redis and EverOS Cloud.
