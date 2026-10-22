# Research tools

`services/grace_tools/research.py` (tools), `research_web.py` (search, page reading, source
ranking), `research_stt.py` (the transcription worker). Tests: `tests/tools/test_research.py`,
sim: `fixtures/tools/scenarios/research.yaml`. None of the tools is core: Grace finds them with
`find_tools`.

| Tool | Risk | What it does |
|---|---|---|
| `deep_research` | visible | Plans 3-4 sub-questions with Gemma, searches each (SearXNG), reads the 3 best pages per sub-question, writes a sourced Markdown report to `~/Documents/Grace/research/`, returns a spoken summary with the sources by site name. Time cap 3 min (`minutes`, at most 10). |
| `fact_check` | read | Searches the claim, reads the 3 best pages, Gemma gives `supported / contradicted / mixed / not enough` and names the sources. |
| `translate_text` | read | Short text via Gemma, in chunks of ~1200 characters. |
| `translate_document` | visible | txt, md, docx: a new file next to it, `name.en.docx` (`rapport.da.docx` becomes `rapport.en.docx`). docx keeps paragraph styles and tables; formatting inside a paragraph (bold words) is lost. |
| `download_file` | visible | Into `~/Downloads` (or a folder he named). Size checked first (HEAD, else a 1-byte range request), refused over 2 GB, and stopped at 2 GB if the server didn't say. Web pages are refused. Never run or opened; programs and scripts (by extension or an `MZ` header) are flagged in the result. Gets Windows' mark of the web (`Zone.Identifier`, ZoneId=3) so SmartScreen still warns. |
| `download_video` | visible | yt-dlp, single videos, `audio_only` for the sound (m4a), else up to 1080p mp4. 2 GB limit from yt-dlp's size estimate. |
| `transcribe` | visible | A media file (mp3, wav, m4a, mp4, mkv, webm, ...), a folder (its newest media file: "videoen i mine overførsler"), or a video link (yt-dlp audio). ffmpeg makes 16 kHz mono, the local STT transcribes, a `.txt` and `.srt` go next to the file (links: `~/Documents/Grace/transcripts/`), Gemma sums it up. |
| `research_jobs` | visible | Status of the jobs above (done, result, path) or `stop`. |

Never overwrite: every new file gets ` (2)` etc. if the name is taken. Dry run (`--dry-run`):
every tool says what it would do and touches neither the network nor Gemma.

## Background jobs
The engine gives up on a tool call after 25 s (`TOOL_TIMEOUT`), so research, translations,
downloads and transcriptions run as jobs on their own thread. The tool waits up to 15 s; if the
job is done by then he gets the result at once (a short transcription, a small download), else
Grace says she's on it and `research_jobs` has the result later. A job stops at its time cap
(research 3 min, translation ~size/20 s, transcription 10 min + file size) or when he says stop.
The engine's `alive()` only ends the 15 s wait: once the call has returned, the connection says
nothing about whether he still wants the job.

Nothing tells Grace when a job finishes; he has to ask. Spoken results at a pause are roadmap
item #21 (engine).

## Search and sources
Ranking (`research_web.score`): search position, +3 official/primary (`.gov`, `.edu`, `.int`,
`europa.eu`, journals, Danish authorities such as `retsinformation.dk`, `sundhed.dk`, `ft.dk`,
universities), +2 for the official site of the thing asked about (`python.org` for Python),
+1.5 reference works (Wikipedia, lex.dk), -3 social media and content farms. One page per site,
sites already used last. A page must share two of the topic's distinctive words: in the first
live run a UK pension bill article ("royal assent") made it into a report on Danish law-making.

**Blocked sites.** A fetch that fails with `SSL: UNEXPECTED_EOF_WHILE_READING` is usually
Mikkel's antivirus refusing a flagged site. It is logged and reported as "possibly blocked by
security software", never retried, and listed under "Not read" in the report. No page is
fetched twice in one job.

**403s.** Some sites refuse a browser User-Agent from a non-browser (w3.org), others refuse
anything that isn't a browser. Every request tries the browser UA and, on 403, once more with an
honest `Grace/2` UA. ft.dk and thedanishparliament.dk still answer 403 to both (bot protection),
so the Folketing's own pages can't be read; reports fall back to Wikipedia and others.

## Transcription: which STT
Measured 2026-10-22 on public FLEURS clips (`fixtures/stt`, 60 Danish, 60 English, mean ~10 s;
never Mikkel's voice), RTX 5090 Laptop with Gemma loaded, CTranslate2 int8_float16, greedy.
"Long" is all 60 clips of a set joined with 0.5 s gaps into one 10-11 min recording, as a lecture
or podcast would be. Reproduce: `bench/.venv/Scripts/python.exe services/grace_tools/research_stt.py bench edda|turbo|large-v3`.

| Model | da clips WER | da long WER | da long speed | en clips WER | en long WER | en long speed |
|---|---|---|---|---|---|---|
| **edda-v0.2** (Danish, = speech_server) | **7.9 %** | **9.8 %** | 45x real time | 56.8 % | 58.4 % | 9x |
| **Whisper large-v3-turbo** | 13.2 % | 14.2 % | 38x | **6.2 %** | **14.9 %** | 48x |
| Whisper large-v3 | 12.3 % | 14.0 % | 9x | 6.1 % | 13.6 % | 9x |

Language detection (turbo, first 30 s): 60/60 Danish, 60/60 English.

**Choice:** Danish goes to edda-v0.2 (loaded the way `speech_server.py` loads it), English and
anything else to large-v3-turbo, both via faster-whisper. large-v3 is no better on English (6.1 vs 6.2 %, long 13.6 vs 14.9 %) and 5x slower. `language` auto: turbo detects it on
the first 30 s. edda is a Danish fine-tune and turns English into poor Danish (57 % WER), so it
can't serve English.

**How long media is cut** mattered more than the model. Whole-file decoding with timestamps (the
faster-whisper default, and what an `.srt` seems to want) doubled edda's WER (clips 14.9 % vs
7.5 % without timestamps; 11 min file 24-28 %); faster-whisper's batched pipeline reached 16 %.
What works: Silero VAD (from faster-whisper, 300 ms pauses) cuts speech into pieces of at most
10 s, each transcribed alone without timestamps, as the live STT does. Long-form edda WER by
piece length: 6 s 10.0 %, 10 s 9.8 %, 15 s 11.4 %, 30 s 20.5 %. The `.srt` cues are those
pieces (≤ 10 s), which suits subtitles.

**Where it runs.** The tools server's venv (`.venv-openvino`) has no CUDA faster-whisper, so
`research_stt.py` runs as a subprocess with the speech venv (`bench/.venv`). The model holds VRAM
(~1-2 GB) only while a job runs, a stuck job can be killed, and the live STT on port 7810 isn't
blocked by a one-hour file. Each job pays ~5-10 s to load the model (a 19 s YouTube clip took
22 s end to end, with language detection). The worker exits with `os._exit(0)`:
CTranslate2's CUDA teardown sometimes crashes (exit 127) after the work is done.

ffmpeg: from PATH (winget ffmpeg 8.1 here), else `imageio-ffmpeg` if installed, else the tool
says ffmpeg is missing. Every subprocess (ffmpeg, the worker) runs with `CREATE_NO_WINDOW`;
yt-dlp hides its own ffmpeg window.

## Tested
`tests/tools/test_research.py`, no internet: recorded SearXNG results and trimmed pages from a
live run (`fixtures/tools/recorded/research/`: a deep research on Danish law-making with its 403s,
and a fact check of the Great Wall myth), Gemma stubbed by prompt, yt-dlp and the STT replaced,
audio a generated 440 Hz tone. Covered: ranking, blocked sites (named, fetched once), report
contents and paths, no overwrite, background jobs, stop, time cap, dry runs, search down; fact
check verdicts; text and document translation (Danish language names, txt/md chunks, docx
styles and tables, Gemma dropping paragraph markers, unsupported types); downloads (size limit
known and unknown, programs flagged and never run, pages refused, no overwrite, SSL block);
video downloads (audio only, size limit); transcription of a file, the newest video in
"overførsler", a link, long transcripts summed up in parts, no speech, bad input; real ffmpeg
to 16 kHz mono; and one optional test that runs the real worker and edda on a public FLEURS clip
(skipped without the speech venv, a GPU or the FLEURS files).

Live, headless, on this PC (2026-10-22, files in a scratch home): deep research 23-24 s, 6 Gemma
calls of 2-3 s; fact check 4-6 s; docx and md translation from Danish (correct, styles kept);
a PDF download from w3.org with the mark of the web; a YouTube video's audio; transcription of a
YouTube link (English, correct summary).

## Not tested / known limits
- **Gemma's 10 s timeout** (`tools_server.gemma`) caps every call at ~500 tokens, hence the
  1200-character chunks; a failed chunk is split once and retried.
- **Shared Gemma.** Background jobs call Gemma while Grace may be talking. llama-server has 2
  slots, so they run in parallel, but decode slows for both and a job may take the slot holding
  Grace's prompt cache. Not measured.
- No live test with the speakers on, and not in the tools sim (the organiser runs it).
- PDFs and .doc can't be translated (txt, md, docx only). Bold/italic inside a docx paragraph is
  lost in translation.
- yt-dlp needs updating when YouTube changes (`pip install -U yt-dlp`); it uses Node for
  YouTube's JavaScript.
- English long-form WER (15 %) is well above clip WER (6 %); the joined FLEURS file switches
  speaker every 10 s, which real recordings don't, so this may overstate it.
