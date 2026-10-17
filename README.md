# Comparative Study of Model Poisoning Strategies and Security Mechanisms in Federated Learning

Cryptography and Network Security project. [Your name, college]

PyTorch simulation on MNIST with 5 clients: three poisoning attacks (sign flipping,
random update, scaling), FedAvg vs coordinate-wise median vs trimmed mean, IID and
non-IID data, HMAC-SHA256 message authentication with replay detection, and an
overhead analysis. TLS is not implemented (single-process simulation).

## Structure
- `src/` code (clients, attacks, aggregation, security, experiments)
- `results/` result CSV files and graphs
- `logs/` HMAC and replay test tables and security logs
- `app.py` Streamlit dashboard

## Run
    pip install -r requirements.txt
    python -m streamlit run app.py

MNIST downloads automatically on the first training run.