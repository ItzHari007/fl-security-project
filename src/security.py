import hmac
import hashlib
import struct
import secrets
import time
import pandas as pd

def generate_keys(client_ids):
    """One random 32-byte secret key per client (assumed shared with the server in advance)."""
    return {cid: secrets.token_bytes(32) for cid in client_ids}

def delta_to_bytes(delta):
    """Convert an update vector to bytes in a fixed format (little-endian float32)."""
    return delta.detach().cpu().numpy().astype("<f4").tobytes()

def serialize(client_id, round_num, msg_id, body):
    """Fixed-size header + update bytes, so sender and receiver authenticate identical bytes."""
    header = struct.pack(">IIQ", client_id, round_num, msg_id)
    return header + body

def compute_tag(key, message_bytes):
    return hmac.new(key, message_bytes, hashlib.sha256).digest()

def make_message(client_id, round_num, delta, key):
    msg_id = secrets.randbits(64)
    body = delta_to_bytes(delta)
    tag = compute_tag(key, serialize(client_id, round_num, msg_id, body))
    return {"client_id": client_id, "round": round_num, "msg_id": msg_id,
            "delta_bytes": body, "tag": tag}

def verify_message(msg, keys):
    """HMAC check only. Return (accepted, reason). Tags are compared in constant time."""
    cid = msg["client_id"]
    if cid not in keys:
        return False, "unknown_client"
    expected = compute_tag(keys[cid], serialize(cid, msg["round"], msg["msg_id"], msg["delta_bytes"]))
    if not hmac.compare_digest(expected, msg["tag"]):
        return False, "bad_tag"
    return True, "ok"

class SecureServerChannel:
    """Server-side receiver: HMAC check, then round check, then duplicate check."""
    def __init__(self, keys):
        self.keys = keys
        self.seen = set()     # (client_id, msg_id) of messages already accepted
        self.log = []         # one record per decision

    def receive(self, msg, current_round):
        ok, reason = verify_message(msg, self.keys)
        if ok:
            if msg["round"] < current_round:
                ok, reason = False, "stale_round"
            elif msg["round"] > current_round:
                ok, reason = False, "future_round"
            elif (msg["client_id"], msg["msg_id"]) in self.seen:
                ok, reason = False, "replay_duplicate"
            else:
                self.seen.add((msg["client_id"], msg["msg_id"]))
        self.log.append({"time": time.time(), "client_id": msg["client_id"],
                         "msg_round": msg["round"], "current_round": current_round,
                         "msg_id": str(msg["msg_id"]),
                         "decision": "accept" if ok else "reject", "reason": reason})
        return ok, reason

    def save_log(self, path):
        pd.DataFrame(self.log).to_csv(path, index=False)
