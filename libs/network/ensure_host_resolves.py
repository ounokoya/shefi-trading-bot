from __future__ import annotations

import os
import socket
from urllib.parse import urlparse


def ensure_host_resolves(host: str, *, port: int = 443, timeout_s: float = 5.0) -> None:
    proxy_url = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
        or ""
    )
    check_host = host
    check_port = port

    if proxy_url:
        parsed = urlparse(proxy_url)
        if parsed.hostname:
            check_host = parsed.hostname
            check_port = parsed.port or 443

    try:
        socket.getaddrinfo(check_host, check_port)
    except socket.gaierror as e:
        raise RuntimeError(
            f"DNS error: cannot resolve '{check_host}'. "
            "Check your DNS configuration (or your proxy settings if you use a proxy)."
        ) from e

    try:
        socket.create_connection((check_host, check_port), timeout=timeout_s).close()
    except socket.gaierror as e:
        raise RuntimeError(
            f"DNS error: cannot resolve '{check_host}'. "
            "Check your DNS configuration (or your proxy settings if you use a proxy)."
        ) from e
    except OSError as e:
        raise RuntimeError(
            f"Network error: cannot connect to '{check_host}:{check_port}'. "
            "Check network access/proxy/firewall."
        ) from e
