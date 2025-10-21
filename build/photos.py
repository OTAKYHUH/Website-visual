# build/photos.py — reusable (importable) + runnable (HTML preview)
from pathlib import Path
import base64, mimetypes, re, sys, webbrowser
import pandas as pd

# ====== YOUR FOLDER (edit if needed) ======
#PREFERRED_FOLDER = Path(r"C:\Users\ahgua\PSA International\(PSAC-CNBD)-YOD-efile - Cess\Staff Photo")
PREFERRED_FOLDER = Path(__file__).resolve().parents[1] / "static" / "Staff Photo"
# =========================================

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# Specific name corrections (UPPERCASE keys/values)
NAME_OVERRIDES = {
    "JACINTA NISHANTI DO DHARMARAJ": "JACINTA NISHANTI D/O DHARMARAJ",
}

def _upper_name_from_stem(stem: str) -> str:
    """Normalize a filename stem into UPPERCASE name (spaces, underscores, dashes → single spaces)."""
    s = re.sub(r"[-_\.]+", " ", stem).strip()
    s = re.sub(r"\s+", " ", s)
    return s.upper()

def _apply_overrides(name_upper: str) -> str:
    """Apply explicit per-name corrections."""
    return NAME_OVERRIDES.get(name_upper, name_upper)

def _to_data_uri(p: Path) -> str:
    """Read an image file and return a data URI (base64)."""
    mime, _ = mimetypes.guess_type(p.name)
    if not mime:
        mime = "image/jpeg"
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _iter_photo_files(folder: Path):
    """Yield supported image files in folder."""
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS and not p.name.startswith("~$"):
            yield p

def load_staff_photos(photo_dir: str | Path):
    """
    Return (photos_dict, photos_df)
      photos_dict: { 'NAME IN UPPER': 'data:image;base64,...' }
      photos_df:   DataFrame ['Name','Image']
    """
    photo_dir = Path(photo_dir)
    rows = []
    for p in _iter_photo_files(photo_dir):
        try:
            nm = _apply_overrides(_upper_name_from_stem(p.stem))
            rows.append({"Name": nm, "Image": _to_data_uri(p), "File": p.name})
        except Exception:
            # skip unreadable files
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Name").reset_index(drop=True)

    photos = {r["Name"]: r["Image"] for _, r in df.iterrows()} if not df.empty else {}
    # keep original 2 cols for compatibility
    return photos, df[["Name", "Image"]] if not df.empty else (photos, pd.DataFrame(columns=["Name", "Image"]))

def render_photos_table(photo_dir: str | Path, out_html: str | Path | None = None, title: str = "Staff Photos Table") -> Path:
    """Build a simple, self-contained HTML file with a table preview (images included)."""
    photos_dict, _ = load_staff_photos(photo_dir)

    rows = []
    for p in _iter_photo_files(Path(photo_dir)):
        name = _apply_overrides(_upper_name_from_stem(p.stem))
        rows.append({"Name": name, "File": p.name, "Image": photos_dict.get(name, "")})
    df = pd.DataFrame(rows).sort_values("Name").reset_index(drop=True) if rows else pd.DataFrame(columns=["Name", "File", "Image"])

    # HTML table with inline images
    fmt = {"Image": lambda uri: f'<img src="{uri}" style="height:64px;border-radius:8px;object-fit:cover;">'}
    table_html = df.to_html(escape=False, index=False, justify="left", classes="grid", formatters=fmt)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body{{background:#0f0f10;color:#e6e6e6;font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Arial,sans-serif;margin:24px}}
  h1{{font-size:20px;margin:0 0 16px}}
  .meta{{opacity:.75;margin-bottom:12px}}
  table.grid{{width:100%;border-collapse:collapse;min-width:680px;background:#121213;border:1px solid #25262b;border-radius:12px;overflow:hidden}}
  thead th{{position:sticky;top:0;background:#15161a;padding:10px;border-bottom:1px solid #25262b;text-align:left}}
  tbody td{{padding:10px;border-bottom:1px dashed #24252a;vertical-align:middle;white-space:nowrap}}
  tbody tr:hover{{background:#16171c}}
  .count{{display:inline-block;background:#1e90ff22;border:1px solid #1e90ff55;color:#bcdcff;
          padding:2px 8px;border-radius:999px;font-weight:700;}}
  .note{{margin-top:12px;opacity:.75}}
</style>
</head>
<body>
  <h1>{title} <span class="count">{{len(df)}} photos</span></h1>
  <div class="meta">Folder: <code>{Path(photo_dir).resolve()}</code></div>
  {table_html}
  <div class="note">Supported: {", ".join(sorted(ALLOWED_EXTS))}</div>
</body>
</html>"""

    out_path = Path(out_html) if out_html else Path(__file__).with_name("photos_preview.html")
    out_path.write_text(html, encoding="utf-8")
    return out_path

def get_tables(photo_dir: str | Path) -> dict[str, "pd.DataFrame | dict"]:
    """
    Return a dict compatible with your NonShift.py expectations:
      {
        'photos_df': DataFrame ['Name','Image'],
        'photos_dict': {NAME: data_uri}
      }
    """
    photos_dict, photos_df = load_staff_photos(photo_dir)
    return {"photos_df": photos_df, "photos_dict": photos_dict}

# ------------------ Public API ------------------
__all__ = ["load_staff_photos", "render_photos_table", "get_tables", "PREFERRED_FOLDER"]

# --------- Optional: runnable preview ----------
def _pick_folder_interactive() -> Path | None:
    """Try Tk folder picker; fall back to console prompt if Tk isn't available."""
    # 1) Try tkinter filedialog, if available
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="Select folder with staff photos")
        root.update()
        root.destroy()
        if path:
            p = Path(path)
            if p.exists() and p.is_dir():
                return p
    except Exception:
        pass

    # 2) Fallback: console prompt
    try:
        inp = input("Enter folder path that contains photos: ").strip('"').strip()
        if inp:
            p = Path(inp).expanduser()
            if p.exists() and p.is_dir():
                return p
    except (EOFError, KeyboardInterrupt):
        return None
    return None

def _auto_guess_folder() -> Path | None:
    """Prefer user's fixed path, then look in common places."""
    # 1) Your fixed path
    try:
        if PREFERRED_FOLDER.exists() and PREFERRED_FOLDER.is_dir():
            if any(_iter_photo_files(PREFERRED_FOLDER)):
                return PREFERRED_FOLDER
    except Exception:
        pass

    # 2) Common relatives
    here = Path(__file__).parent
    candidates = [
        here / "photos",
        here / "static" / "photos",
        Path.cwd() / "photos",
        Path.cwd() / "static" / "photos",
        Path.cwd(),  # last resort
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir() and any(_iter_photo_files(cand)):
            return cand
    return None

if __name__ == "__main__":
    folder = _auto_guess_folder()
    if folder is None:
        print("[i] No photos found in your preferred folder or common locations.")
        print("[i] Please pick a folder…")
        folder = _pick_folder_interactive()

    if not folder:
        print("[!] No folder selected. Exiting.")
        sys.exit(1)

    if not any(_iter_photo_files(folder)):
        print(f"[!] Folder has no supported images: {folder}")
        print(f"[!] Supported extensions: {', '.join(sorted(ALLOWED_EXTS))}")
        sys.exit(2)

    out_file = render_photos_table(folder)
    print(f"[i] Wrote: {out_file}")
    try:
        webbrowser.open(out_file.as_uri(), new=2)
    except Exception:
        print(f"[i] Open this in your browser: {out_file}")
