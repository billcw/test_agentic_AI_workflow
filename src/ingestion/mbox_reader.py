"""
mbox_reader.py - Reader for MBOX email archive files.

MBOX is a plain-text format that stores emails concatenated together,
each separated by a "From " line (note: one space, not a colon).
It is the standard export format for Gmail (via Google Takeout),
Yahoo Mail, Thunderbird, Apple Mail, and most other email clients.

Python's built-in mailbox library reads MBOX files lazily — it does
NOT load the entire file into memory at once. This is critical because
Gmail exports can easily be 10-20GB. The mailbox.mbox() iterator
reads one message at a time, processes it, and moves on.

Why not just convert to .eml files first?
That would require extracting potentially millions of individual files,
which is slow and wastes disk space. The MBOX reader handles the archive
directly in one pass — faster, cleaner, and no intermediate files.

Output format is identical to pst_reader.py and email_reader.py so
the rest of the pipeline (chunker, vector store, metadata DB) needs
no changes at all.
"""

import mailbox
import email as email_lib
from email import policy
from pathlib import Path


def _decode_header(raw_value: str) -> str:
    """
    Safely decode an email header value that may be encoded.

    Email headers can be encoded in various ways (e.g. UTF-8, quoted-printable,
    base64) using RFC 2047 encoding like =?UTF-8?B?...?=. Python's
    email.header.decode_header() handles all of these and returns the
    plain text string.
    """
    if not raw_value:
        return ""
    try:
        from email.header import decode_header, make_header
        return str(make_header(decode_header(raw_value)))
    except Exception:
        return str(raw_value)


def _extract_body(msg) -> str:
    """
    Extract plain text body from an email message object.
    Handles both multipart and single-part messages.
    Prefers text/plain over text/html.
    """
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))

            # Skip attachments
            if "attachment" in disposition:
                continue

            if content_type == "text/plain":
                try:
                    charset = part.get_content_charset() or "utf-8"
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode(charset, errors="replace")
                except Exception:
                    pass
    else:
        try:
            charset = msg.get_content_charset() or "utf-8"
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(charset, errors="replace")
        except Exception:
            body = str(msg.get_payload())

    return body.strip()


def read_mbox(file_path: str | Path) -> list[dict]:
    """
    Read all emails from an MBOX archive file.

    Args:
        file_path: Path to the .mbox file

    Returns:
        List of page dicts compatible with chunk_pages():
        [
            {
                "text":   "From: ...\nDate: ...\nSubject: ...\n\nbody...",
                "page":   1,
                "source": "gmail_export.mbox",
                "method": "email",
                "metadata": {
                    "email_date":    "2024-03-15 09:32",
                    "email_sender":  "colleague@example.com",
                    "email_subject": "Re: Weekly Status",
                    "source_type":   "email",
                }
            },
            ...
        ]
    """
    file_path = Path(file_path)
    source_name = file_path.name
    pages = []

    # mailbox.mbox() reads lazily — safe for very large files
    mbox = mailbox.mbox(str(file_path))

    for i, msg in enumerate(mbox):
        try:
            subject = _decode_header(msg.get("Subject", "(no subject)"))
            sender  = _decode_header(msg.get("From",    "(unknown sender)"))
            date    = _decode_header(msg.get("Date",    "(no date)"))
            to      = _decode_header(msg.get("To",      "(unknown recipient)"))

            # Normalize date to YYYY-MM-DD HH:MM format where possible
            date_str = date
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(date)
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                date_str = date  # Keep raw string if parsing fails

            body = _extract_body(msg)

            # Skip messages with no usable content
            if not body and not subject:
                continue

            # Build header stamp injected every 800 chars (same pattern as pst_reader)
            # so every chunk produced from a long email body carries context
            header_stamp = (
                f"[From: {sender} | Date: {date_str} | Subject: {subject}]"
            )

            stamped_body_parts = []
            interval = 800
            body_text = body or ""
            for pos in range(0, max(1, len(body_text)), interval):
                stamped_body_parts.append(header_stamp)
                stamped_body_parts.append(body_text[pos:pos + interval])
            stamped_body = "\n".join(stamped_body_parts)

            page_text = (
                f"From: {sender}\n"
                f"To: {to}\n"
                f"Date: {date_str}\n"
                f"Subject: {subject}\n\n"
                f"{stamped_body}"
            )

            pages.append({
                "text":   page_text,
                "page":   len(pages) + 1,
                "source": source_name,
                "method": "email",
                "metadata": {
                    "email_date":    date_str,
                    "email_sender":  sender,
                    "email_subject": subject,
                    "source_type":   "email",
                },
            })

        except Exception:
            # Skip malformed messages silently — large archives always have some
            continue

    mbox.close()
    return pages
