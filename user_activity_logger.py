#!/usr/bin/env python3
import logging
import os
from datetime import datetime

USER_ACTIVITY_LOG = "user_activity.log"

class UserActivityLogger:
    def __init__(self, log_path=USER_ACTIVITY_LOG):
        self.log_path = log_path
        self.logger = logging.getLogger("UserActivityLogger")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.logger.propagate = False

    def log_event(self, user, document=None, query=None, response=None, feedback=None):
        entry = {
            "user": user,
            "document": document,
            "query": query,
            "response": response,
            "feedback": feedback
        }
        # Only log fields that are not None
        filtered = {k: v for k, v in entry.items() if v is not None}
        self.logger.info(" | ".join(f"{k}: {v}" for k, v in filtered.items()))

# Example usage:
if __name__ == "__main__":
    logger = UserActivityLogger()
    # Log a document upload
    logger.log_event(user="alice", document="gov_policy.pdf")
    # Log a query and response
    logger.log_event(user="alice", query="What is the eligibility?", response="Eligibility is ...")
    # Log feedback
    logger.log_event(user="alice", query="What is the eligibility?", feedback="thumbs_up")
