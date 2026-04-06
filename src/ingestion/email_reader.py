"""
email_reader.py - Email file text extraction (.eml format).
"""

import email
from email import policy
from pathlib import Path


def read_email(file_path: str | Path) -> list[dict]:
    """
    Extract text from an .eml email file.
    Extracts subject, sender, date, and body.

    Returns a list with a single dict containing the full email:
    [
        {
            "page": 1,
            "text": "From: ...\nSubject: ...\n\nBody text...",
            "method": "email",
            "source": "filename.eml"
        }
    ]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Email file not found: {file_path}")

    with open(file_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    # Extract headers
    subject = msg.get("subject", "No Subject")
    sender = msg.get("from", "Unknown Sender")
    date = msg.get("date", "Unknown Date")
    to = msg.get("to", "Unknown Recipient")

    # Extract body text
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_content()
    else:
        body = msg.get_content()

    # Combine into searchable text
    full_text = (
        f"From: {sender}\n"
        f"To: {to}\n"
        f"Date: {date}\n"
        f"Subject: {subject}\n\n"
        f"{body.strip()}"
    )

    return [{
        "page": 1,
        "text": full_text,
        "method": "email",
        "source": file_path.name
    }]
