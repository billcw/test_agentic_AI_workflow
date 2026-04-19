"""
email_reader.py - Email file text extraction (.eml and .msg formats).

Two very different file formats both carry the word "email":

.eml files are plain text RFC 822 format — standard internet email.
Python's built-in email library reads these directly.

.msg files are Microsoft OLE2 compound document format — a binary
mini-filesystem that Outlook uses to store emails. Python's built-in
email library cannot read these at all. We use the extract-msg library
which understands the OLE2 structure and pulls out the fields we need.

Why not just convert .msg to .eml first?
That would add an extra step and a dependency on external tools.
extract-msg handles it in pure Python, cleanly, inside the pipeline.
"""
import email
from email import policy
from pathlib import Path

try:
    import extract_msg
    EXTRACT_MSG_AVAILABLE = True
except ImportError:
    EXTRACT_MSG_AVAILABLE = False


def _read_eml(file_path: Path) -> list[dict]:
    """
    Extract text from an .eml (RFC 822) email file.
    Uses Python's built-in email library.
    """
    with open(file_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    subject = msg.get("subject", "No Subject")
    sender = msg.get("from", "Unknown Sender")
    date = msg.get("date", "Unknown Date")
    to = msg.get("to", "Unknown Recipient")

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_content()
    else:
        body = msg.get_content()

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


def _read_msg(file_path: Path) -> list[dict]:
    """
    Extract text from a .msg (Microsoft OLE2) email file.
    Uses the extract-msg library.

    extract-msg opens the OLE2 binary structure and exposes the
    standard email fields (sender, subject, date, body) as Python
    attributes — no manual binary parsing needed.
    """
    if not EXTRACT_MSG_AVAILABLE:
        raise ImportError(
            "extract-msg is required to read .msg files. "
            "Install it with: pip install extract-msg"
        )

    msg = extract_msg.openMsg(str(file_path))

    try:
        subject = msg.subject or "No Subject"
        sender = msg.sender or "Unknown Sender"
        date = str(msg.date) if msg.date else "Unknown Date"
        to = msg.to or "Unknown Recipient"

        # extract-msg exposes body as plain text via .body
        # and HTML via .htmlBody — we want plain text
        body = msg.body or ""
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        body = body.strip()

    finally:
        msg.close()

    full_text = (
        f"From: {sender}\n"
        f"To: {to}\n"
        f"Date: {date}\n"
        f"Subject: {subject}\n\n"
        f"{body}"
    )

    return [{
        "page": 1,
        "text": full_text,
        "method": "email",
        "source": file_path.name
    }]


def read_email(file_path: str | Path) -> list[dict]:
    """
    Extract text from an email file (.eml or .msg).
    Dispatches to the correct reader based on file extension.

    Returns a list with a single dict containing the full email:
    [
        {
            "page": 1,
            "text": "From: ...\nTo: ...\nDate: ...\nSubject: ...\n\nBody...",
            "method": "email",
            "source": "filename.msg"
        }
    ]
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Email file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".msg":
        return _read_msg(file_path)
    else:
        # .eml and any other RFC 822 variants
        return _read_eml(file_path)
