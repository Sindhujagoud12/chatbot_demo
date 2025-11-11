# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:34:03 2025

@author: AAA
"""

"""
chatbot.py

A small, modular chatbot engine with two backends:
 - "local": rule-based simple intent matching
 - "openai": calls OpenAI Chat Completion (optional)

Usage:
    from chatbot import Chatbot
    cb = Chatbot(backend="local")         # local mode
    cb = Chatbot(backend="openai", openai_api_key="...")  # OpenAI mode
    reply = cb.get_reply("hello")
"""

import re
import os
from typing import List, Dict, Optional

# Try to import openai only if user wants that backend. We keep it optional so the package
# works without openai installed.
try:
    import openai
except Exception:
    openai = None


class Chatbot:
    """
    Chatbot class: manages conversation history and chooses a backend to produce replies.
    """

    def __init__(
        self,
        backend: str = "local",
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",  # placeholder model name; change as needed
        system_prompt: str = "You are a helpful assistant."
    ):
        """
        backend: "local" or "openai"
        openai_api_key: required only if backend == "openai"
        openai_model: model to call (only for openai backend)
        system_prompt: system message for chat model
        """
        self.backend = backend.lower()
        self.history: List[Dict[str, str]] = []  # stores messages as {role: "user"/"assistant"/"system", "content": "..."}

        # store system prompt for LLM context (used only in openai backend)
        self.system_prompt = system_prompt

        # Local rule-based intent-response map
        self.intents = [
            (re.compile(r"\bhi\b|\bhello\b|\bhey\b", re.I), "Hello! How can I help you today?"),
            (re.compile(r"\bhow are you\b|\bhow's it going\b", re.I), "I'm a program, so I don't have feelings, but I'm ready to help! 😊"),
            (re.compile(r"\bbye\b|\bgoodbye\b|\bsee you\b", re.I), "Goodbye! Have a great day."),
            (re.compile(r"\bthanks\b|\bthank you\b", re.I), "You're welcome! Anything else I can do?"),
        ]
        self.default_response = "Sorry, I didn't understand that. Can you rephrase?"

        # Configure OpenAI client if requested
        self.openai_model = openai_model
        if self.backend == "openai":
            if openai is None:
                raise RuntimeError("openai package not installed. Install with `pip install openai` to use the openai backend.")
            if not openai_api_key:
                # Accept API key from env var if not passed directly
                openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise RuntimeError("openai_api_key is required for backend='openai'.")
            openai.api_key = openai_api_key

            # Optionally set a timeout or other OpenAI client settings here if needed.

    def reset_history(self):
        """Clear past conversation turns."""
        self.history = []

    def update_history(self, role: str, content: str):
        """Append a message to conversation history."""
        self.history.append({"role": role, "content": content})

    def get_reply(self, user_message: str) -> str:
        """
        Return a reply string to the user's message.

        This function:
         - updates history with the user message,
         - runs the chosen backend,
         - updates history with assistant response,
         - returns the assistant response string.
        """
        # sanitize user message
        user_message = user_message.strip()
        if not user_message:
            return "Please send a non-empty message."

        # add user to history
        self.update_history("user", user_message)

        if self.backend == "local":
            reply = self._local_reply(user_message)
        elif self.backend == "openai":
            reply = self._openai_reply(user_message)
        else:
            reply = f"Unknown backend '{self.backend}'."

        # append assistant reply to history and return
        self.update_history("assistant", reply)
        return reply

    # ---------- Local rule-based backend ----------
    def _local_reply(self, user_message: str) -> str:
        """
        A simple rule-based intent matcher. Checks regex patterns in order and returns
        the first matched response. If nothing matches, returns default_response.
        """
        for pattern, response in self.intents:
            if pattern.search(user_message):
                return response

        # small fallback logic — echo with a suggestion
        if len(user_message.split()) <= 2:
            return f"I got '{user_message}'. Can you share a bit more detail?"
        return self.default_response

    # ---------- OpenAI backend ----------
    def _openai_reply(self, user_message: str) -> str:
        """
        Compose messages from system + current history and call OpenAI chat API.
        This implementation uses the pattern of providing the full history to the model.
        """
        # Build the messages list expected by chat completion endpoint
        messages = [{"role": "system", "content": self.system_prompt}]
        # use last N turns to avoid huge payloads; we include the whole history here for clarity
        for turn in self.history:
            # only include user/assistant roles, system is handled separately
            if turn["role"] in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})

        # Trim messages if they are too long (very naive trimming: keep last 12 messages)
        if len(messages) > 20:
            # keep system + last 18 entries
            messages = [messages[0]] + messages[-18:]

        # Call OpenAI ChatCompletion
        try:
            resp = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                n=1,
            )
            # standard location for the assistant text depends on API response shape
            answer = resp["choices"][0]["message"]["content"].strip()
            return answer
        except Exception as e:
            # graceful error for debugging; in production you might want to log instead
            return f"[OpenAI error] {str(e)}"

    # ---------- Utility methods ----------
    def get_history(self) -> List[Dict[str, str]]:
        """Return a copy of the conversation history."""
        return list(self.history)
