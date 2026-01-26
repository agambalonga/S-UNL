"""
Disabilita SSL verification per compatibilità con Zscaler proxy.
Questo modulo deve essere importato PRIMA di qualsiasi altra libreria che usa HTTPS.
"""
import os
import ssl
import warnings

# Variabili d'ambiente
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"

# Disabilita warnings SSL
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# SSL context non verificato
ssl._create_default_https_context = ssl._create_unverified_context

# Patch requests per disabilitare SSL verification
try:
    import requests
    original_request = requests.Session.request
    def no_ssl_verification(self, method, url, **kwargs):
        kwargs['verify'] = False
        return original_request(self, method, url, **kwargs)
    requests.Session.request = no_ssl_verification
except ImportError:
    pass

print("⚠️  SSL verification DISABLED (Zscaler proxy compatibility)")
