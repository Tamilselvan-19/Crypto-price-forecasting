import hashlib
import json
from time import time


class Blockchain:

    def __init__(self):
        self.chain = []
        self.create_block("Genesis Block", "0")

    def create_block(self, data, previous_hash):

        block = {
            "index": len(self.chain) + 1,
            "timestamp": time(),
            "data": data,
            "previous_hash": previous_hash
        }

        block["hash"] = self.hash(block)

        self.chain.append(block)

        return block

    def hash(self, block):

        encoded = json.dumps(block, sort_keys=True).encode()

        return hashlib.sha256(encoded).hexdigest()

    def add_prediction(self, symbol, predicted_price):

        last_block = self.chain[-1]

        previous_hash = last_block["hash"]

        data = {
            "symbol": symbol,
            "predicted_price": predicted_price
        }

        return self.create_block(data, previous_hash)

    def show_chain(self):
        return self.chain