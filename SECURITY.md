# Security Policy

## Supported versions

Security fixes are applied to the current `0.1.x` release line. Older versions are not
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

Fine-Tuning Studio runs training locally. It reads `HF_TOKEN` only from the user
environment and reports token presence without exposing its value. Uploaded filenames
are sanitized and job paths are kept inside the configured application workspace.

Model repositories can execute code when `trust_remote_code` is enabled. Treat this as
an explicit trust decision: review the repository and leave the option disabled unless
it is required.

## Out of scope

The following are generally out of scope unless they demonstrate a vulnerability in
Fine-Tuning Studio itself:

- Vulnerabilities in a user's model, dataset, local operating system, GPU driver, or
  Ollama installation.
- Social engineering attempts.
- Reports requiring access to real user credentials or private data.
- Missing features or unsupported hardware paths without a security impact.
