# Public API Deployment

The static GitHub Pages site needs a separately running API. This repository includes
a `Dockerfile` suitable for a Hugging Face Docker Space, Render, Railway, or another
container host.

## Required before a public launch

1. Train and evaluate a reviewed `checkpoints/model.pt` using the documented held-out
   data and robustness suite. Never deploy an unvalidated or randomly initialized
   classifier.
2. Create a Docker-capable app on the chosen host and deploy this repository.
3. Set `PROOFSHIELD_ALLOWED_ORIGINS=https://marcel330-ait.github.io` on the host.
4. After the host returns its HTTPS URL, edit the Pages branch's `runtime-config.js`:

```js
window.__PROOFSHIELD_API_BASE_URL__ = "https://your-proofshield-api.example";
```

No API key should be embedded in the public browser build. The current API accepts no
credentials; add rate limiting, file scanning, observability, and host-level request
limits before an unrestricted public release.
