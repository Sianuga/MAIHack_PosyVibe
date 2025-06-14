
import os
import openai
import dotenv
import pretty_midi
import subprocess
import ast
import json
import datetime
from typing import Any

dotenv.load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY"))

MODEL = "gpt-4.1"
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent
RES_DIR        = BASE_DIR / "resources"
OUT_DIR        = BASE_DIR / "data" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOUNDFONT_PATH = RES_DIR / "FatBoy-v0.786.sf2"

stamp          = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_MIDI_PATH = f"ai_gen_{stamp}.mid"
OUTPUT_WAV_PATH  = f"ai_gen_{stamp}.wav"



SYSTEM_PROMPT = """
You are a creative, professional musical composer.  
Your job: **deliver expressive, varied, human-sounding MIDI arrangements** in pure JSON (no comments, no prose) for pretty_midi.

────────────────  CORE FORMAT  ────────────────
Each top-level key (e.g. "lead", "pad", "bass", "drums", "arp") is an instrument.control_changes.
Add a top-level key "bpm" with the piece’s tempo in beats-per-minute. Use the bpm to correctly set the start and end time in seconds for each note.

{
  "bpm": 80,
  "lead":  { "program": 84, "notes": [ … ] },
  "pad":   { "program": 89, "notes": [ … ] },
  "bass":  { "program": 34, "notes": [ … ] },
  "drums": { "is_drum": true, "program": 0, "notes": [ … ] }
}

Every note object =  
{ "velocity": 1-127, "pitch": MIDI#, "start": sec, "end": sec  (> start) }

• **Drums**: key = "drums", set `"is_drum": true`, use GM pitches  
  36 Kick  38 Snare  42 CHH  46 OHH  49 Crash  51 Ride  etc.  
  Hits 0.05–0.3 s; include ghost notes, fills, and bar-4 / bar-8 variations.

────────────────  COMPOSITION RULES  ────────────────
1. **Follow the requested key/scale & chord progression**  
   → Build chords from the interval table below.  
   → Bass **begins each chord/bar on its root** (passing tones & octaves welcome).

2. **Rhythmic & dynamic variety**  
   • Mix at least three different note lengths per part.  
   • Use syncopation, off-beats, rests, and accented/ghost velocities.

3. **Phrase & motif**  
   • Lead should develop motifs (repeat & vary), not just climb scales.  
   • Pads may use inversions, broken chords, or stabs—not only block chords.  
   • Arps should alternate directions or rhythms, not a constant stair-step.

4. **Drum groove**  
   • Kick + snare pocket; hats/perc drive; fill or crash every 4–8 bars.  
   • This is primarily electronic music so even for calm tracks keep some regular kicks

5. **Ending**  
   All parts finish **together** on the final down-beat; no trailing silence.

6. **No “repeat” instructions** – spell out every note bar-by-bar.

────────────────  REFERENCE: SCALES & MOODS  ────────────────
Major  [0,2,4,5,7,9,11]  happy | Natural Minor  [0,2,3,5,7,8,10]  sad | …  
Pentatonic Major, Blues Minor, Dorian, Lydian, Mixolydian, etc.  
Japanese: In [0,1,5,7,8], Yo [0,2,5,7,9], etc.

────────────────  REFERENCE: CHORD BUILDING  ────────────────
Intervals from root → note list:  
Major [0,4,7] | Minor [0,3,7] | Maj7 [0,4,7,11] | Dom7 [0,4,7,10] | Min7 [0,3,7,10]

Example in C:  
C  → [60,64,67] G  → [67,71,74] Am → [69,72,76] F → [65,69,72]

────────────────  MINI-EXAMPLE: 1-bar I–V–vi–IV pad  ────────────────
{ "pad": { "program":89, "notes":[
  {"velocity":80,"pitch":60,"start":0.0,"end":1.0},
  {"velocity":80,"pitch":64,"start":0.0,"end":1.0},
  {"velocity":80,"pitch":67,"start":0.0,"end":1.0},

  {"velocity":80,"pitch":67,"start":1.0,"end":2.0},
  {"velocity":80,"pitch":71,"start":1.0,"end":2.0},
  {"velocity":80,"pitch":74,"start":1.0,"end":2.0},
  … repeat for Am and F …
] } }



────────────────  SAMPLE PROGRESSIONS & MOODS  ────────────────
• Pop (hopeful):  I–V–vi–IV → C–G–Am–F  
• Film-score (grand):  I–vi–IV–V → G–Em–C–D–G  
• Neo-soul (smooth):  ii7–V7–Imaj7–vi7 → Cm7–F7–Bbmaj7–Gm7  
• Dorian folk (mysterious):  i–VII–IV–i → Dm–C–G–Dm  
• Indie (bittersweet):  vi–IV–I–V → Am–F–C–G

────────────────  EXTENDED SCALE / MOOD TABLE  ────────────────
Major                   [0,2,4,5,7,9,11]   happy, uplifting, confident
Natural Minor           [0,2,3,5,7,8,10]   sad, introspective, moody
Pentatonic Major        [0,2,4,7,9]        simple, open, folksy
Pentatonic Minor        [0,3,5,7,10]       dark, rocky, exciting
Blues Minor             [0,3,5,6,7,10]     expressive, defiant
Harmonic Minor          [0,2,3,5,7,8,11]   exotic, dramatic
Melodic Minor Ascend.   [0,2,3,5,7,9,11]   jazzy, ambiguous
Dorian                  [0,2,3,5,7,9,10]   cool, hopeful, mellow
Phrygian                [0,1,3,5,7,8,10]   tense, mysterious
Lydian                  [0,2,4,6,7,9,11]   epic, filmic, soaring
Mixolydian              [0,2,4,5,7,9,10]   blues-rock, funky
Locrian                 [0,1,3,5,6,8,10]   unresolved, eerie

Japanese In             [0,1,5,7,8]        sparse, haunting
Japanese Yo             [0,2,5,7,9]        bright, cheerful, folk
Hirajoushi              [0,2,3,7,8]        sombre, calm
Iwato                   [0,1,5,6,10]       edgy, resonant
Kumoi                   [0,2,3,7,9]        nostalgic, wistful

────────────────  TARGETED MOOD PLAYBOOKS  ────────────────
(Use these when a user says “Make calmer / happier / energize”.)

► **Make Calmer**
  • BPM range : 60-90  
  • Friendly keys/scales :  D major, G Lydian, A Dorian, Japanese Yo  
  • Chord progressions :  
      – I – vi – IV – V   (e.g., D – Bm – G – A)  
      – Imaj7 – iii7 – vi7 – ii7 (jazzy ballad)  
      – In scale: i – bII – V  (e.g., A – Bb – E in A-in)  
  • Melody tips : stay on the beat not too syncopated, long sustained notes, stepwise motion, gentle leaps ≤ 6 st; leave rests; soften velocities 55-85.

► **Make Happier**
  • BPM range : 100-128  
  • Friendly keys/scales :  C major, A Pentatonic Major, F Mixolydian  
  • Chord progressions :  
      – I – V – vi – IV (classic pop)  
      – I – IV – V – IV (’50s feel-good)  
      – I – V/vi – vi – IV  (sunshine pop variant)  
  • Melody tips : bright pentatonic hooks, octave jumps, syncopated 8th/16th rhythms, velocities 75-110.

► **Energize**
  • BPM range : 135-175 (drum-&-bass / EDM)  
  • Friendly keys/scales :  E minor, F# Phrygian (dark energy), D Harmonic Minor  
  • Chord progressions :  
      – vi – IV – I – V   (e.g., Em – C – G – D)  
      – i – bVI – bVII – v   (rock-phrygian drive)  
      – i7 – IV7  (jungle/d’n’b 2-bar loop)  
  • Melody tips : short aggressive riffs, rhythmic 16ths, call-and-response, stutter edits, velocities 90-127; drums add syncopated kicks & rapid hats.

  ────────────────  MOOD-SPECIFIC GUIDELINES  ────────────────
Always aim for (at least 16 bars at the chosen BPM).  
Finish every part at the final down-beat—no trailing silence.

► CALMER
  • Target BPM : 60 – 90   • Feel : spacious, soothing, reflective  
  • Texture : sustained pad chords, soft legato lead, sparse bass (1-2 notes/bar).  
  • Drums : light kick, brushed hats/ride; gentle fill or cymbal swell only at major section transitions (e.g., bar 8, 16).  
  • Dynamics : velocities 50-85, gradual swells/decays, leave audible rests.  
  • Harmony : use major, lydian, dorian, or Japanese Yo; favour warm chord tones (add 9ths/maj7s).  

► HAPPIER
  • Target BPM : 100 – 128   • Feel : upbeat, bright, friendly  
  • Texture : rhythmic pads or piano stabs, catchy repeating 2- or 4-note lead motif, supportive walking bass.  
  • Drums : lively kick/snare with syncopated closed hats, occasional hand-clap or tambourine; small hi-hat or snare fill every 4 bars.  
  • Dynamics : velocities 70-115, accented off-beats, light side-chain-style pad pulsation.  
  • Harmony : major, mixolydian, pentatonic major; classic pop progressions (I-V-vi-IV, etc.).  

► ENERGIZE
  • Target BPM : 135 – 175   • Feel : driving, intense, exciting  
  • Texture : punchy bass (root + octave jumps), short riff-based lead, optional arpeggiator.  
  • Drums : aggressive kick pattern, 16-note hats or rides, crash/FX every 4 bars; mini-fill each bar, bigger fill bar 8/16.  
  • Dynamics : velocities 90-127, stark accents, call-and-response rhythms, build-ups & drops.  
  • Harmony : natural/harmonic minor, phrygian, blues minor; progressions with tension (vi-IV-I-V, i-bVI-bVII-v).  

(If the user doesn’t specify a BPM, pick a value in the listed range for the chosen mood.)
────────────────────────────────────────────────────────────

────────────────  (end mood playbooks)  ────────────────


────────────────  DELIVERABLE  ────────────────
Return **only** the final JSON object—no markdown, no comments, no prose.



"""


def run_music_prompt(prompt_text: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ],
        temperature=0.0,
    )
    raw = response.choices[0].message.content
    print(f"Raw:{raw}")
    processed=parse_music_json(raw)
    print(processed)
    return processed


import re
import ast
import json

import json, ast, re

def parse_music_json(raw: str) -> dict:
    """
    Strip markdown, comments, and parse to dict.
    Keeps top-level 'bpm' and any instrument entries containing 'notes'.
    """
    # 1. drop ```json fences (multiline-safe)
    txt = re.sub(r"^\s*```[\w-]*", "", raw, flags=re.MULTILINE).strip()
    txt = re.sub(r"```$", "", txt).strip()

    # 2. remove // and # comments
    cleaned = []
    for line in txt.splitlines():
        line = re.split(r"//|#", line)[0]          # keep code left of comment
        if line.strip():
            cleaned.append(line)
    txt = "\n".join(cleaned)

    # 3. JSON first, fallback to Python literal
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        data = ast.literal_eval(txt)

    # 4. normalise
    out: dict[str, Any] = {}
    bpm_val = data.get("bpm")
    if bpm_val is not None:
        out["bpm"] = float(bpm_val)

    for key, val in data.items():
        if isinstance(val, dict) and "notes" in val:
            out[key] = val
    return out




import pretty_midi
import subprocess

def discover_bar_grid_max(music_dict: dict, bar_len: float = 2.0) -> float:
    """
    Find the last bar line <= max_note_end.
    Default bar_len = 2 s (60 BPM four-beat bar) – adjust if you pass BPM.
    """
    max_end = 0.0
    for conf in music_dict.values():
        for n in conf["notes"]:
            max_end = max(max_end, float(n["end"]))
    # snap to last bar grid
    return (int(max_end / bar_len)) * bar_len


# ───────────────── instrument helper ─────────────────
def instrument_from_dict(
    name: str,
    config: dict,
    clip_to: float | None = None
) -> tuple[pretty_midi.Instrument, float]:
    """
    Create a pretty_midi.Instrument from one JSON block.
    Returns (instrument_obj, max_end_time).
    If clip_to is provided, any note end > clip_to is trimmed to clip_to.
    """
    program = int(config.get("program", 0))
    is_drum = name.lower() == "drums"
    instr = pretty_midi.Instrument(program=program, is_drum=is_drum)

    max_end = 0.0
    for n in config["notes"]:
        start = float(n["start"])
        end   = float(n["end"])
        if clip_to is not None and end > clip_to:
            end = clip_to
            if end <= start:          # avoid zero-length notes
                continue
        note = pretty_midi.Note(
            velocity=int(n["velocity"]),
            pitch=int(n["pitch"]),
            start=start,
            end=end
        )
        instr.notes.append(note)
        max_end = max(max_end, end)

    return instr, max_end

# ───────────────────────── constants ─────────────────────────
BEATS_PER_BAR = 4
DEFAULT_BPM   = 80                    # used if LLM forgets bpm
CUSHION       = 0.1                # 10 ms early-cut

# ───────────────── master pipeline ─────────────────

import math
from typing import Optional


def play_midi_live(midi_obj: pretty_midi.PrettyMIDI):
    """OPTIONAL: quick real-time preview through server speakers."""
    import time, fluidsynth

    sfid, fs = None, None
    try:
        fs   = fluidsynth.Synth()
        fs.start(driver="coreaudio" if os.name == "posix" else "dsound")
        sfid = fs.sfload(str(SOUNDFONT_PATH))
        # program / channel assignment
        for ch, inst in enumerate(midi_obj.instruments):
            bank = 128 if inst.is_drum else 0
            prog = 0 if inst.is_drum else inst.program
            fs.program_select(ch, sfid, bank, prog)

        # schedule events (simple)
        events = []
        for ch, inst in enumerate(midi_obj.instruments):
            for note in inst.notes:
                events.append(("on",  note.start, ch, note.pitch, note.velocity))
                events.append(("off", note.end,   ch, note.pitch, 0))

        events.sort(key=lambda e: e[1])
        t0 = time.time()
        for typ, t, ch, p, v in events:
            while (time.time() - t0) < t:
                time.sleep(0.001)
            (fs.noteon if typ=="on" else fs.noteoff)(ch, p, v)
    finally:
        if fs:  fs.delete()

import pathlib
def generate_and_render_music(prompt_text: str,
                              midi_path: str,
                              wav_path: str,
                              *,
                              realtime_preview: bool = False) -> str:
    """Return path to rendered WAV.  If realtime_preview=True the MIDI is also
    played through the server’s audio device (useful only on dev boxes)."""

    # 1️⃣  LLM → JSON
    music = run_music_prompt(prompt_text)

    # 2️⃣  tempo / bar length
    bpm = float(music.pop("bpm", DEFAULT_BPM))
    bar_len = 60.0 / bpm * BEATS_PER_BAR

    # 3️⃣  find latest note
    max_end = max(float(n["end"]) for part in music.values() for n in part["notes"])

    # 4️⃣  clip at last whole bar
    clip_to = math.floor(max_end / bar_len) * bar_len
    if clip_to <= 0:            # safety net
        clip_to = max_end

    # 5️⃣  build MIDI
    midi = pretty_midi.PrettyMIDI()
    for name, conf in music.items():
        inst, _ = instrument_from_dict(name, conf, clip_to=clip_to)
        midi.instruments.append(inst)
    midi.write(midi_path)

    # 6️⃣  optional real-time preview (DEV only)
    if realtime_preview and os.getenv("ALLOW_REALTIME_PREVIEW", "false").lower() == "true":
        play_midi_live(midi)           # blocks until done

    # 7️⃣  offline render (always)
    result = subprocess.run(
        ["fluidsynth", "-ni", "-g", "1.0",
        str(SOUNDFONT_PATH), str(midi_path),
        "-F", str(wav_path), "-r", "44100"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FluidSynth failed ({result.returncode}):\n{result.stderr}"
    )
    
    return wav_path