API CENTRAL DE IA SEMANTICA - EXEMPLO

1) Instale Python 3.10+
2) Crie um ambiente virtual
3) Instale os pacotes:
   pip install -r requirements.txt
4) Opcional: defina um token no ambiente:
   LINKAGE_API_TOKEN=seu-token
5) Suba a API:
   uvicorn app:app --host 0.0.0.0 --port 8000

ENDPOINTS
- GET /health
- POST /v1/linkage/semantic-check

NO APPS SCRIPT (Script Properties)
- LINKAGE_API_URL = https://SEU-ENDERECO-DA-API
- LINKAGE_API_TOKEN = seu-token   (opcional, se usar token)
- LINKAGE_API_TIMEOUT_MS = 20000  (opcional)

EXEMPLO DE URL
http://IP-OU-DOMINIO:8000

OBSERVACAO
A IA semantica desta API nao decide a classificacao final. Ela so sugere:
- nome-base semelhante
- endereco-base semelhante
- mesmo socio provavel
- alerta de parentesco provavel

O sistema continua usando suas regras para checklist, classificacao e dossie.
