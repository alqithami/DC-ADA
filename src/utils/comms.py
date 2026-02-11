class CommunicationLogger:
    def __init__(self, schema=None):
        self.schema = schema or {}
        self.messages = []
        self.total_bytes = 0

    def log_broadcast(self, msg_type, size_bytes):
        self.messages.append({"type": msg_type, "size": size_bytes})
        self.total_bytes += size_bytes

    def get_summary(self):
        return {
            "num_messages": len(self.messages),
            "total_bytes": self.total_bytes,
            "bandwidth_bytes_per_sec": self.total_bytes / max(1, len(self.messages))
        }

    def reset(self):
        self.messages = []
        self.total_bytes = 0
