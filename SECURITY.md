# Security Policy

## Supported versions

Security fixes are applied to the current `1.x` release line. Older versions are not
supported.

## Reporting a vulnerability

Do not report vulnerabilities through public issues, discussions, pull requests, or
social media.

Use GitHub's **Report a vulnerability** form in the repository's Security/Advisories
section. Repository administrators must enable GitHub private vulnerability reporting
in the repository security settings before the form is available.

Include:

- A clear description of the vulnerability and affected versions.
- Reproduction steps or a minimal proof of concept.
- Impact assessment and any suggested mitigation.
- Whether the report contains sensitive data requiring special handling.

Do not include real API tokens, private datasets, personal data, or proprietary models
in a report. Use redacted or synthetic examples.

Maintainers will acknowledge valid reports, investigate privately, coordinate a fix or
mitigation, and credit reporters when requested and appropriate. Timing depends on
severity, reproducibility, release coordination, and maintainer availability.

## Security boundaries and expectations

Fine-Tuning Studio runs training locally. It reads `HF_TOKEN` or the legacy
`HUGGING_FACE_HUB_TOKEN` only from the user environment and reports token presence
without exposing its value. Uploaded filenames are sanitized and job paths are kept
inside the configured application workspace.

Model repositories can execute code when `trust_remote_code` is enabled. Treat this as
an explicit trust decision: review the repository and leave the option disabled unless
it is required. The UI requires typing `I UNDERSTAND`. Custom reward modules have the
same confirmation and execute without a sandbox.

The application binds to loopback, keeps Streamlit XSRF/CORS protections enabled, does
not load `.env` files, and reads Hugging Face authentication only from `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN`.

GPU process termination is opt-in. The UI shows the PID and executable, accepts only
same-user non-system candidates, requires typing `TERMINATE`, and revalidates process
identity immediately before termination. The current app, active training workers,
Ollama-managed runners, operating-system services, and inaccessible processes are
protected. Fine-Tuning Studio never attempts a privileged GPU-driver reset.

The global app shutdown control is separately confirmed. It waits for cooperative job
cancellation, revalidates registered worker identity or app ancestry immediately before
terminating a process tree, and refuses to close the UI when cleanup cannot be completed
safely. It does not stop Ollama, unrelated processes, or operating-system services.

## Out of scope

The following are generally out of scope unless they demonstrate a vulnerability in
Fine-Tuning Studio itself:

- Vulnerabilities in a user's model, dataset, local operating system, GPU driver, or
  Ollama installation.
- Social engineering attempts.
- Reports requiring access to real user credentials or private data.
- Missing features or unsupported hardware paths without a security impact.
