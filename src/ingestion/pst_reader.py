"""
pst_reader.py - Reader for Outlook PST and OST archive files.

Uses libpff-python (pypff) to extract emails recursively from all folders.
Returns the same 'pages' list format as every other reader in this pipeline.

Each email becomes one page with subject, sender, date, folder path, and body.

Folders in SKIP_FOLDERS are excluded from ingestion — these are high-volume
automated alert folders that would flood the index with noise.

Senders in SKIP_SENDERS are excluded regardless of which folder they appear in.
"""

import pypff
from pathlib import Path
from datetime import datetime

# Folder names to skip entirely (case-insensitive match).
# Add any other high-volume automated folders here.
SKIP_FOLDERS = {
    "qradar",
    "qradar prod",
    "qradar qas",
    "qradar localhost",
    "spam search folder 2",
    "search root",
    "deleted items",
}

# Sender name/address substrings to skip (case-insensitive).
# Any message whose sender contains one of these strings is excluded.
SKIP_SENDERS = {
    "qradar",
}


def _should_skip_folder(folder_name: str) -> bool:
    """Return True if this folder should be excluded from ingestion."""
    if not folder_name:
        return False
    return folder_name.strip().lower() in SKIP_FOLDERS


def _should_skip_sender(sender: str) -> bool:
    """Return True if this sender should be excluded from ingestion."""
    if not sender:
        return False
    sender_lower = sender.strip().lower()
    return any(skip in sender_lower for skip in SKIP_SENDERS)


def _extract_messages(folder, folder_path: str, source_name: str,
                      pages: list) -> None:
    """
    Recursively walk a pypff folder and extract all messages into pages.

    Args:
        folder:      pypff folder object
        folder_path: human-readable path string like 'Inbox/EMS Weekly Status'
        source_name: filename of the PST/OST file (used as source in chunk IDs)
        pages:       list to append extracted page dicts into
    """
    folder_name = folder.get_name() or ""

    if _should_skip_folder(folder_name):
        return

    # Extract messages in this folder
    num_messages = folder.get_number_of_sub_messages()
    for i in range(num_messages):
        try:
            msg = folder.get_sub_message(i)

            subject = msg.get_subject() or "(no subject)"
            sender  = msg.get_sender_name() or "(unknown sender)"

            # Skip automated noise senders regardless of folder
            if _should_skip_sender(sender):
                continue

            # Date comes back as a datetime object or None
            date_obj = msg.get_delivery_time()
            date_str = date_obj.strftime("%Y-%m-%d %H:%M") if date_obj else "(no date)"

            # Body is bytes — decode carefully
            body_bytes = msg.get_plain_text_body()
            if body_bytes:
                try:
                    body_text = body_bytes.decode("utf-8", errors="replace")
                except Exception:
                    body_text = str(body_bytes)
            else:
                body_text = ""

            # Skip messages with no usable content
            if not body_text.strip() and subject == "(no subject)":
                continue

            # Build a short header stamp that will be prepended to every chunk
            # so no chunk is ever context-free when the body spans multiple chunks.
            header_stamp = (
                f"[From: {sender} | Date: {date_str} | Subject: {subject}]"
            )

            # Inject the header stamp every 800 characters into the body so
            # that every chunk produced by the chunker carries context.
            body = body_text.strip()
            stamped_body_parts = []
            interval = 800
            for pos in range(0, max(1, len(body)), interval):
                stamped_body_parts.append(header_stamp)
                stamped_body_parts.append(body[pos:pos + interval])
            stamped_body = "\n".join(stamped_body_parts)

            # Format the page text the way the chunker expects it
            page_text = (
                f"From: {sender}\n"
                f"Date: {date_str}\n"
                f"Folder: {folder_path}\n"
                f"Subject: {subject}\n\n"
                f"{stamped_body}"
            )

            pages.append({
                "text":   page_text,
                "page":   len(pages) + 1,   # unique page number = unique chunk IDs
                "source": source_name,       # filename, e.g. outlook_inbox_backup_102021.pst
                "method": "email",
                "metadata": {
                    "subject":     subject,
                    "sender":      sender,
                    "date":        date_str,
                    "folder_path": folder_path,
                    "source_type": "email",
                }
            })

        except Exception:
            # Skip malformed messages silently — large archives always have some
            continue

    # Recurse into sub-folders
    num_sub = folder.get_number_of_sub_folders()
    for i in range(num_sub):
        sub = folder.get_sub_folder(i)
        sub_name = sub.get_name() or ""
        sub_path = f"{folder_path}/{sub_name}" if sub_name else folder_path
        _extract_messages(sub, sub_path, source_name, pages)


def read_pst(file_path: str | Path) -> list[dict]:
    """
    Read all emails from a PST or OST file.

    Args:
        file_path: Path to the .pst or .ost file

    Returns:
        List of page dicts compatible with chunk_pages():
        [
            {
                "text":   "From: ...\nDate: ...\nSubject: ...\n\nbody...",
                "page":   1,
                "source": "outlook_inbox_backup_102021.pst",
                "method": "email",
                "metadata": { "subject": ..., "sender": ..., ... }
            },
            ...
        ]

    Raises:
        OSError: if the file cannot be opened (corrupt or unsupported format)
    """
    file_path = Path(file_path)
    source_name = file_path.name
    pages = []
    pf = pypff.file()
    opened = False

    try:
        pf.open(str(file_path))
        opened = True
        root = pf.get_root_folder()
        _extract_messages(root, "root", source_name, pages)
    finally:
        # Only close if open() succeeded — closing an unopened file crashes
        if opened:
            try:
                pf.close()
            except Exception:
                pass

    return pages
