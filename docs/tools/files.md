# Tool group: files

Files and documents, by voice. Code: `services/grace_tools/files.py` (paths, writing, moving,
deleting, zip, undo) and `services/grace_tools/files_docs.py` (reading, searching, creating,
charts, comparing, converting). Tests: `tests/tools/test_files.py`. Sim scenarios:
`fixtures/tools/scenarios/files.yaml`. Packages: `services/grace_tools/files.requirements.txt`.

## Tools

| Tool | Risk | Core | What it does |
|---|---|---|---|
| `read_document` | read | yes | Text of txt, md, PDF, docx, xlsx, pptx, csv, json, in parts of ~1200 characters (`part=2`...). |
| `write_file` | visible | yes | A new text or Markdown file. Refuses to overwrite. |
| `search_in_files` | read | | Words inside files (text, Word, PDF, Excel, PowerPoint) under a folder. Slow. |
| `recent_files` | read | | Newest files, all folders or one, optionally one kind and the last N days. |
| `compare_files` | read | | What changed between two files: lines added, removed, changed; Excel/CSV cell by cell. |
| `append_to_file` | visible | | Adds text to a text, Markdown or Word file. |
| `rename_file` | visible | | New name in the same folder (keeps the extension). |
| `move_file` | visible | | Into another folder. Refuses when the name is taken there. |
| `copy_file` | visible | | Into a folder or as a new name. Refuses when the name is taken. |
| `make_folder` | visible | | A new folder (with parents). |
| `zip_files` / `unzip_file` | visible | | Pack into a .zip; unpack into a new folder (zip slip and 2 GB checks). |
| `create_document` | visible | | Word, PDF, Excel, PowerPoint or Markdown from structured args (title, sections, slides, table, chart). |
| `make_chart` | visible | | Bar, line or pie chart as a PNG. |
| `convert_file` | visible | | docx/pptx/xlsx to PDF, md to docx/pdf, PDF to docx/txt, csv to xlsx and back, json to xlsx, images png/jpg/webp/bmp/gif and image to PDF. Original kept. Slow. |
| `undo_last` | visible | | Takes back Grace's last file change. |
| `edit_file` | destructive | | Replace or remove text in a text, Markdown or Word file, or replace a text file's whole content. |
| `replace_file` | destructive | | Move or copy a file onto an existing one (what move/copy refuse). |
| `delete_files` | destructive | | To the Recycle Bin (send2trash), never a hard delete. |

Risk is static per tool (the registry enforces it), so "move onto an existing file" is its own
destructive tool, `replace_file`; `move_file` and `copy_file` refuse and name it.
Generated files (`create_document`, `convert_file`, `make_chart`, `zip_files`) never overwrite:
a taken name becomes `name (2).ext`. `write_file` refuses instead, since he named the file.

## Paths and safety
- Accepted: full paths from earlier results, `~/...`, Danish folder names (`skrivebordet`,
  `dokumenter/rapport.docx`, `overførsler`, `billeder`, `grace`; a copy of tools_server's
  FOLDER_ALIASES plus a few), and bare names, looked for in Documents/Grace, Documents, Desktop,
  Downloads and then the server's file index (`ctx.files.candidates`).
- Refused: anything outside the home folder, any hidden (`.ssh`, `.config`) or system part
  (AppData, NTUSER), moving/renaming/deleting home or its main folders, and creating or renaming to
  program or script extensions (.exe, .bat, .ps1, .py...).
- Default folder for new files: `~/Documents/Grace/`.
- `ctx.dry` is honoured by every tool that changes something: it reports and changes nothing.

## Undo
`ctx.data_dir/files-undo.jsonl` keeps the last 30 changes (older backups are removed with them); edits, appends and replacements keep a
backup copy in `ctx.data_dir/files-undo/`. `undo_last` reverses the last entry: created files go
to the Recycle Bin, moves go back, edits are restored from the backup. It refuses when he changed
the file after Grace did (mtime and size are recorded), so his own work is never thrown away.
Deletes can't be undone by Grace; the result tells him they are in the Recycle Bin.

## Converting to PDF
docx, pptx and xlsx go through Word, PowerPoint or Excel by COM (a PowerShell script, opened
read-only, no window, 120 s timeout) when registered, else LibreOffice (`soffice --headless`) when
found, else the built-in fpdf2 renderer (text, headings, lists, tables, pictures; not the
layout, and the result says so). On this PC Word, PowerPoint and Excel are installed and all three
converted a test file (6-13 s each, checked by hand outside the tests); LibreOffice is not.
PDFs use Arial from `C:\Windows\Fonts` so Danish letters, dashes and € survive; without it the
core font (latin-1) is used.

## Tested (tests/tools/test_files.py, 64 cases with the parametrised ones)
Every tool's normal case; Danish names and folder names; æøå in file names and in PDF text;
refusals (outside home, AppData, .ssh, scripts, main folders, taken names, zip slip, missing
folders); destructive tools returning a confirmation first; undo for write, append, edit, rename,
move, replace, make_folder and convert, and undo refusing after his own change; dry-run for write,
edit, delete, create and convert; reading in parts, scanned PDFs, old .doc, pictures; charts with
bad values; compare for text, Word and Excel; every conversion pair listed above with Office
switched off, plus the Office path through a stub; search skipping node_modules; recent files
from a walk and from the server's index; a multi-step shopping list Word -> PDF -> read -> undo.
Home is a tmp folder, the Recycle Bin a folder, Office and LibreOffice switched off.

## Not tested, and why
- Real COM conversions run only in the by-hand check above (tests must not start Office).
- The real Recycle Bin (send2trash) is replaced in tests; it is a thin, well-used library.
- The sim scenarios were written but not run (the organiser runs grace-tools-sim).
- Paths go into results as `path: ...` for later calls; that Grace never reads them aloud is
  up to the prompt and is only checked in two scenarios (`not_answer: C:\`).

## Not done
- No OCR for scanned PDFs or pictures (read_document says so).
- Old .doc/.xls/.ppt are not read (could go through Office COM later).
- Comparing a file with its own earlier version needs two files; the undo backups are not offered
  as "the version before".
- `tests/tools/test_registry.py::test_find_matches_words` now ties with the real `move_file`
  (score 7 each, `move_file` wins on name order). The test's fake tool needs a word no real tool
  uses; it is outside this group's files.
