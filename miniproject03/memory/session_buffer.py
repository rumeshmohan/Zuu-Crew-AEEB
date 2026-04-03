class SessionBuffer:
    """Buffers chat history while maintaining persistent safety constraints."""

    def __init__(self, max_pairs: int = 5):
        self.history = []
        self.max_pairs = max_pairs
        self.persistent_context = ""

    def set_persistent_context(self, context: str):
        """Set non-expiring constraints like allergies or recipient info."""
        self.persistent_context = context

    def requires_safety_check(self) -> bool:
        """Return True if the persistent context contains allergy-related information."""
        context = self.persistent_context.lower()
        return "allergy" in context or "allergic" in context

    def add_message(self, role: str, content: str):
        """Append a message to history, evicting oldest pairs beyond max_pairs."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_pairs * 2:
            self.history = self.history[-(self.max_pairs * 2):]

    def get_history_string(self) -> str:
        """Return the conversation history as a formatted string, prefixed by the persistent context."""
        header = f"USER PROFILE: {self.persistent_context}\n" if self.persistent_context else ""
        lines = [
            f"{'Customer' if m['role'] == 'user' else 'Concierge'}: {m['content']}"
            for m in self.history
        ]
        return header + "\n".join(lines)

    def clear(self):
        """Clear all messages from history."""
        self.history = []