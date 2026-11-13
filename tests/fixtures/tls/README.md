A throwaway self-signed certificate, used to make a real TLS handshake fail.

It is not a secret: the private key exists only so a local server can complete
(or rather, refuse to complete) a handshake with a client that does not trust
it. Regenerate with:

    openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem \
      -days 36500 -nodes -subj "/CN=localhost" \
      -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
