"""A small JSON-backed internship contact manager."""

import json
import logging
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "contacts.json"
logging.basicConfig(
    filename=BASE_DIR / "contact_manager.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class Contact:
    """Represent one internship contact."""

    def __init__(self, name, role, score):
        if not name.strip() or not role.strip():
            raise ValueError("name and role are required")
        if not 0 <= score <= 100:
            raise ValueError("score must be between 0 and 100")
        self.name = name.strip()
        self.role = role.strip()
        self.score = float(score)

    def to_dict(self):
        return {"name": self.name, "role": self.role, "score": self.score}


class ContactManager:
    """Manage contacts and persist them as JSON."""

    def __init__(self, data_file):
        self.data_file = data_file
        self.contacts = []

    def load(self):
        try:
            if self.data_file.exists():
                records = json.loads(self.data_file.read_text(encoding="utf-8"))
                self.contacts = [Contact(**record) for record in records]
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            logger.error("Could not load contacts: %s", error)
            self.contacts = []

    def add(self, contact):
        self.contacts.append(contact)
        self.save()
        logger.info("Added contact: %s", contact.name)

    def save(self):
        records = [contact.to_dict() for contact in self.contacts]
        self.data_file.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def summary(self):
        if not self.contacts:
            return {"count": 0, "average_score": 0.0}
        scores = np.array([contact.score for contact in self.contacts])
        return {
            "count": len(self.contacts),
            "average_score": round(float(scores.mean()), 2),
            "highest_score": float(scores.max()),
        }


def main():
    manager = ContactManager(DATA_FILE)
    manager.load()
    if not manager.contacts:
        for name, role, score in [
            ("Ayesha Khan", "Mentor", 92),
            ("Bilal Ahmed", "Intern", 84),
            ("Sara Noor", "Reviewer", 88),
        ]:
            manager.add(Contact(name, role, score))

    print("Internship Contact Manager")
    for contact in manager.contacts:
        print(f"- {contact.name}: {contact.role} ({contact.score:.0f})")
    print(f"Summary: {json.dumps(manager.summary())}")


if __name__ == "__main__":
    main()
