import secrets

api_key = secrets.token_hex(32)  # 32 bytes = 256 bits
print(api_key)
