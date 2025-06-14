
import os
import openai
import dotenv
import pretty_midi
import subprocess
import ast
import json
import datetime
import pathlib

dotenv.load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY"))

MODEL = "gpt-4.1"
SOUNDFONT_PATH = "resources/FatBoy-v0.786.sf2"
OUTPUT_MIDI_PATH = "ai_gen_score_test.mid"
OUTPUT_WAV_PATH = f"output{datetime.datetime.now()}.wav"

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# Build the absolute path to the soundfont file.
SOUNDFONT_PATH = SCRIPT_DIR / "resources" / "FatBoy-v0.786.sf2"

SYSTEM_PROMPT = """
You are a talented musical composer who outputs melodies and drum patterns for a midi instrument to be fed into pretty_midi.
You can input notes for different instruments (defined by "program") choosing the velocity (loudness), pitch, start, and end in seconds.
Pitch is defined in semitones, where C3 is 60.

You can also generate drum patterns using the "drums" instrument. To do so:
- Use the key "drums" instead of a numbered program.
- Set "is_drum": true for the drum track.
- Use standard General MIDI drum pitches, such as:
    - 36 = Kick Drum
    - 38 = Snare
    - 42 = Closed Hi-Hat
    - 46 = Open Hi-Hat
    - 49 = Crash Cymbal
    - 51 = Ride Cymbal
    - 39 = Clap
    - 41 = Low Tom
    - 45 = Mid Tom
    - 48 = High Tom
Quantise your drum hits to start and end times that make sense rhythmically. Drums are usually short in duration (e.g., 0.1s–0.3s).

If the user asks you to create music at a certain BPM, you can use start and end seconds to stay quantised and make notes that fit into subdivisions or multiplications of one beat.

Try to make an interesting and thematic composition based on the user's instructions.

Make your output in the following JSON configuration:

{
    "lead": {
        "program": 82,
        "notes": [
            {"velocity": 80, "pitch": 60, "start": 0.0, "end": 2.0},
            {"velocity": 80, "pitch": 64, "start": 2.0, "end": 4.0}
        ]
    },
    "pad": {
        "program": 89,
        "notes": [
            {"velocity": 80, "pitch": 60, "start": 0.0, "end": 2.0},
            {"velocity": 80, "pitch": 64, "start": 2.0, "end": 4.0}
        ]
    },
    "drums": {
        "is_drum": true,
        "program":0,
        "notes": [
            {"velocity": 127, "pitch": 36, "start": 0.0, "end": 0.1},
            {"velocity": 100, "pitch": 38, "start": 1.0, "end": 1.1},
            {"velocity": 100, "pitch": 42, "start": 2.0, "end": 2.1}
        ]
    }
}
Your response will be fed directly into a music production software so do not include any explanation text.
"""

SYSTEM_PROMPT = """
You are a creative, professional musical composer for AI-driven music production.
You output imaginative, realistic, and musically interesting MIDI arrangements in JSON format, which will be directly imported into a music creation tool using pretty_midi.

**Instructions:**

- Each part (e.g., "lead", "pad", "bass", "drums") is a dictionary entry.
- Each instrument entry includes:
    - "program": MIDI program number (see General MIDI for options),
    - "notes": list of notes, each with:
        - "velocity": 1–127 (how loud/strong),
        - "pitch": MIDI note number (C3 = 60),
        - "start": when (in seconds) the note begins,
        - "end": when (in seconds) the note ends (must be > start).
- Drums are a special case: use key "drums", set `"is_drum": true`, and use General MIDI drum pitches:
    - 36 = Kick Drum, 38 = Snare, 42 = Closed Hi-Hat, 46 = Open Hi-Hat,
    - 49 = Crash Cymbal, 51 = Ride, 39 = Clap, 41 = Low Tom, 45 = Mid Tom, 48 = High Tom, etc.
    - Drum notes are typically short (0.05–0.3s) and can be rapid and layered.
- **DO NOT** output anything except the JSON structure.

**Write music that is:**
- Varied in note lengths and rhythms (not just steady 8ths/quarters).
- Musically plausible and stylistically appropriate for the requested genre/mood.
- Featuring syncopation, rests, fills, and dynamic contrast where appropriate.
- With creative chord/melody/rhythm relationships (not just stepwise movement).
- For longer pieces, add development, breakdowns, fills, or section changes.
- Optionally include "bass" or "arp" instruments for richer arrangement.

### EXAMPLES

Example 1: Upbeat dance-pop groove (syncopation, variation, layered drums)
{
    "lead": {
        "program": 81,
        "notes": [
            {"velocity": 90, "pitch": 72, "start": 0.0, "end": 0.3},
            {"velocity": 88, "pitch": 74, "start": 0.4, "end": 0.8},
            {"velocity": 95, "pitch": 76, "start": 1.0, "end": 1.2},
            {"velocity": 75, "pitch": 79, "start": 1.5, "end": 2.1},
            {"velocity": 82, "pitch": 77, "start": 2.3, "end": 2.7}
        ]
    },
    "pad": {
        "program": 89,
        "notes": [
            {"velocity": 60, "pitch": 60, "start": 0.0, "end": 2.5},
            {"velocity": 65, "pitch": 67, "start": 2.5, "end": 4.0}
        ]
    },
    "bass": {
        "program": 38,
        "notes": [
            {"velocity": 100, "pitch": 43, "start": 0.0, "end": 0.4},
            {"velocity": 105, "pitch": 47, "start": 0.8, "end": 1.1},
            {"velocity": 110, "pitch": 45, "start": 1.5, "end": 1.9}
        ]
    },
    "drums": {
        "is_drum": true,
        "program": 0,
        "notes": [
            {"velocity": 120, "pitch": 36, "start": 0.0, "end": 0.09},   // Kick
            {"velocity": 85, "pitch": 42, "start": 0.25, "end": 0.33},  // Closed HH
            {"velocity": 110, "pitch": 38, "start": 0.5, "end": 0.6},   // Snare
            {"velocity": 100, "pitch": 36, "start": 1.0, "end": 1.08},  // Kick
            {"velocity": 90, "pitch": 42, "start": 1.25, "end": 1.33},  // Closed HH
            {"velocity": 120, "pitch": 49, "start": 2.0, "end": 2.15},  // Crash
            {"velocity": 110, "pitch": 38, "start": 2.5, "end": 2.6}    // Snare
        ]
    }
}

Example 2: Downtempo, moody, syncopated (rests, off-beat drums, varying lengths)
{
    "lead": {
        "program": 83,
        "notes": [
            {"velocity": 70, "pitch": 68, "start": 0.0, "end": 0.2},
            {"velocity": 75, "pitch": 71, "start": 0.45, "end": 1.0},
            {"velocity": 90, "pitch": 73, "start": 1.8, "end": 2.1},
            {"velocity": 60, "pitch": 74, "start": 2.7, "end": 3.2}
        ]
    },
    "pad": {
        "program": 92,
        "notes": [
            {"velocity": 55, "pitch": 61, "start": 0.0, "end": 2.0},
            {"velocity": 60, "pitch": 69, "start": 2.0, "end": 4.0}
        ]
    },
    "bass": {
        "program": 34,
        "notes": [
            {"velocity": 95, "pitch": 41, "start": 0.0, "end": 0.7},
            {"velocity": 80, "pitch": 43, "start": 1.1, "end": 1.7},
            {"velocity": 105, "pitch": 46, "start": 2.2, "end": 2.9}
        ]
    },
    "drums": {
        "is_drum": true,
        "program": 0,
        "notes": [
            {"velocity": 115, "pitch": 36, "start": 0.0, "end": 0.09},
            {"velocity": 85, "pitch": 42, "start": 0.18, "end": 0.22},
            {"velocity": 90, "pitch": 38, "start": 1.1, "end": 1.2},
            {"velocity": 90, "pitch": 46, "start": 1.7, "end": 1.9},
            {"velocity": 120, "pitch": 49, "start": 2.0, "end": 2.18},
            {"velocity": 85, "pitch": 42, "start": 2.5, "end": 2.58}
        ]
    }
}

Example 3: Calm, ambient, slow build (sustained notes, soft drums)
{
    "lead": {
        "program": 85,
        "notes": [
            {"velocity": 50, "pitch": 64, "start": 0.0, "end": 2.0},
            {"velocity": 52, "pitch": 67, "start": 2.2, "end": 5.0}
        ]
    },
    "pad": {
        "program": 91,
        "notes": [
            {"velocity": 40, "pitch": 60, "start": 0.0, "end": 5.0}
        ]
    },
    "drums": {
        "is_drum": true,
        "program": 0,
        "notes": [
            {"velocity": 60, "pitch": 51, "start": 1.0, "end": 1.15},  // Ride
            {"velocity": 55, "pitch": 42, "start": 3.0, "end": 3.1}   // HH
        ]
    }
}

Example 4: Fast, energetic, rhythmic (fills, offbeat drums, rapid notes)
{
    "lead": {
        "program": 82,
        "notes": [
            {"velocity": 110, "pitch": 72, "start": 0.0, "end": 0.15},
            {"velocity": 105, "pitch": 76, "start": 0.18, "end": 0.33},
            {"velocity": 120, "pitch": 79, "start": 0.4, "end": 0.55},
            {"velocity": 90, "pitch": 74, "start": 0.65, "end": 0.77},
            {"velocity": 95, "pitch": 81, "start": 0.88, "end": 1.0}
        ]
    },
    "pad": {
        "program": 93,
        "notes": [
            {"velocity": 75, "pitch": 66, "start": 0.0, "end": 2.0},
            {"velocity": 80, "pitch": 69, "start": 2.1, "end": 4.0}
        ]
    },
    "bass": {
        "program": 39,
        "notes": [
            {"velocity": 127, "pitch": 36, "start": 0.0, "end": 0.18},
            {"velocity": 110, "pitch": 39, "start": 0.19, "end": 0.33},
            {"velocity": 127, "pitch": 36, "start": 0.34, "end": 0.48},
            {"velocity": 110, "pitch": 41, "start": 0.49, "end": 0.60}
        ]
    },
    "drums": {
        "is_drum": true,
        "program": 0,
        "notes": [
            {"velocity": 127, "pitch": 36, "start": 0.0, "end": 0.09},
            {"velocity": 100, "pitch": 42, "start": 0.12, "end": 0.16},
            {"velocity": 127, "pitch": 38, "start": 0.2, "end": 0.3},
            {"velocity": 80, "pitch": 39, "start": 0.34, "end": 0.38},
            {"velocity": 100, "pitch": 42, "start": 0.43, "end": 0.47},
            {"velocity": 127, "pitch": 36, "start": 0.5, "end": 0.58},
            {"velocity": 127, "pitch": 49, "start": 1.0, "end": 1.18}
        ]
    }
}

**Never just output repeated or scale patterns. Use musical variety, syncopation, creative rhythms, fills, and rests. The output should be lively, human-like, and suitable for modern electronic/rock/pop/ambient music as appropriate to the user's instructions.**
Your response will be fed directly into a music production software so do not include any explanation text.

---
Reference: Common Scales, Modes, and Their Moods

Use the following interval patterns (semitones from tonic) when composing in the specified keys/modes:

- **Major**: [0,2,4,5,7,9,11] — happy, uplifting, confident, resolved
- **Natural Minor**: [0,2,3,5,7,8,10] — sad, introspective, moody
- **Pentatonic Major**: [0,2,4,7,9] — happy, simple, open, folksy
- **Pentatonic Minor**: [0,3,5,7,10] — dark, rocky, exciting, but sad
- **Blues Minor**: [0,3,5,6,7,10] — expressive, sorrowful, defiant
- **Harmonic Minor**: [0,2,3,5,7,8,11] — exotic, dramatic, mysterious, tense
- **Melodic Minor Ascending**: [0,2,3,5,7,9,11] — jazzy, ambiguous
- **Dorian Mode**: [0,2,3,5,7,9,10] — cool, jazzy, melancholy, calm
- **Phrygian Mode**: [0,1,3,5,7,8,10] — very minor, unsettling, tense
- **Lydian Mode**: [0,2,4,6,7,9,11] — epic, film score, impressive
- **Mixolydian Mode**: [0,2,4,5,7,9,10] — rocky, funky, fun
- **Locrian Mode**: [0,1,3,5,6,8,10] — unresolved, mysterious

**Japanese Scales:**
- **In Scale**: [0,1,5,7,8] — sparse, haunting
- **Yo Scale**: [0,2,5,7,9] — bright, traditional, cheerful
- **Hirajoushi**: [0,2,3,7,8] — koto music, sombre, calm
- **Iwato**: [0,1,5,6,10] — mysterious, edgy
- **Kumoi**: [0,2,3,7,9] — melancholy, nostalgic

*When a specific style, mood, or scale is requested, use these intervals as the basis for melody and chord selection.*

### Reference: Chord Notation, Intervals, and MIDI Note Arrays

When you see a chord progression like I–V–vi–IV, use the following rules to convert chord names into MIDI notes:

- Each chord is built on a **root note**. For example, in the key of C, “I” is C major (root = C, MIDI 60).
- Use the **intervals below** to construct each chord:
    - **Major chord**: [0, 4, 7] (root, major third, perfect fifth)
    - **Minor chord**: [0, 3, 7] (root, minor third, perfect fifth)
    - **Dominant 7th chord**: [0, 4, 7, 10]
    - **Minor 7th chord**: [0, 3, 7, 10]
    - **Major 7th chord**: [0, 4, 7, 11]
- Add each interval (in semitones) to the root note’s MIDI number to get the notes of the chord.

**Examples in the key of C (C = 60):**
- **C (I)**: C major = MIDI [60, 64, 67]
- **G (V)**: G major = MIDI [67, 71, 74]
- **Am (vi)**: A minor = MIDI [69, 72, 76]
- **F (IV)**: F major = MIDI [65, 69, 72]
- **C7**: C dominant seventh = MIDI [60, 64, 67, 70]
- **Dm7**: D minor seventh = MIDI [62, 65, 69, 72]

**How to represent a chord progression in JSON for pretty_midi:**
- For each chord, add all chord notes with the same start and end time (to make a block chord).
- Change chords at regular intervals (for example, each bar is 1.0 second).

**Example: I–V–vi–IV in C (each chord is 1 bar/second):**
```json
{
  "pad": {
    "program": 89,
    "notes": [
      {"velocity": 90, "pitch": 60, "start": 0.0, "end": 1.0},
      {"velocity": 90, "pitch": 64, "start": 0.0, "end": 1.0},
      {"velocity": 90, "pitch": 67, "start": 0.0, "end": 1.0},

      {"velocity": 90, "pitch": 67, "start": 1.0, "end": 2.0},
      {"velocity": 90, "pitch": 71, "start": 1.0, "end": 2.0},
      {"velocity": 90, "pitch": 74, "start": 1.0, "end": 2.0},

      {"velocity": 90, "pitch": 69, "start": 2.0, "end": 3.0},
      {"velocity": 90, "pitch": 72, "start": 2.0, "end": 3.0},
      {"velocity": 90, "pitch": 76, "start": 2.0, "end": 3.0},

      {"velocity": 90, "pitch": 65, "start": 3.0, "end": 4.0},
      {"velocity": 90, "pitch": 69, "start": 3.0, "end": 4.0},
      {"velocity": 90, "pitch": 72, "start": 3.0, "end": 4.0}
    ]
  }
}



Sample chord progressions for different moods (use as inspiration - you can use others too don't be limited)

- Pop major: I–V–vi–IV (e.g., C–G–Am–F in C major)
- Minor ballad: i–VI–III–VII (e.g., Am–F–C–G in A minor)
- Jazz dorian: i7–IV7 (e.g., Dm7–G7 in D dorian)
- Japanese in: i–bII–V (e.g., C–Db–G for C in scale)
- Blues: I7–IV7–V7 (e.g., A7–D7–E7 in A blues)
1. Hopeful, triumphant pop
Mood: Inspiring, victorious
Progression: I – V – vi – I⁷ – IV
Example in C: C – G – Am – C7 – F

2. Soulful, neo-soul/R&B
Mood: Smooth, jazzy, sophisticated
Progression: ii⁷ – V⁷ – Imaj7 – vi⁷
Example in Bb: Cm7 – F7 – Bbmaj7 – Gm7

3. Classic cinematic/film score
Mood: Grand, sweeping, emotional
Progression: I – vi – IV – V – I
Example in G: G – Em – C – D – G

4. Mysterious, modal folk (Dorian)
Mood: Enigmatic, ancient, adventurous
Progression: i – VII – IV – i
Example in D Dorian: Dm – C – G – Dm

5. Modern indie/alt rock
Mood: Bittersweet, reflective
Progression: vi – IV – I – V
Example in A minor/C major: Am – F – C – G

"""


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
    """Return dict with instruments and optionally 'bpm' key."""
    # --- strip markdown fences
    txt = raw.strip()
    txt = re.sub(r"^```(\w+)?", "", txt).strip()
    txt = re.sub(r"```$", "", txt).strip()

    # --- remove // or # comments
    cleaned_lines = []
    for line in txt.splitlines():
        line = re.split(r"//|#", line)[0]
        if line.strip():
            cleaned_lines.append(line)
    txt = "\n".join(cleaned_lines)

    # --- try JSON first (handles true/false)
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(txt)
        except Exception as e:
            print("❌ parser failed; raw output follows ↓↓↓\n", raw)
            raise e

    # --- normalise structure
    out = {}
    if "bpm" in data:
        out["bpm"] = float(data["bpm"])
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
# ...

def generate_and_render_music(prompt_text: str, midi_path: str, wav_path: str):
    # 1️⃣  LLM → JSON dict  (expecting it to include "bpm": <number>)
    music = run_music_prompt(prompt_text)

    # 2️⃣  tempo / bar length
    bpm = float(music.get("bpm", DEFAULT_BPM))
    if "bpm" in music:
        del music["bpm"]               # remove metadata before building instruments
    bar_len = 60.0 / bpm * BEATS_PER_BAR   # seconds per 4/4 bar




    # 3️⃣  discover absolute latest note end
    max_end = 0.0
    for part in music.values():
        for n in part["notes"]:
            max_end = max(float(n["end"])               # latest note of any part
              for part in music.values()
              for n in part["notes"])
    print(f"max end : {max_end}")

    # 4️⃣  snap to last whole bar and set final clip point
    clip_to = math.floor(max_end / bar_len) * bar_len 
    if clip_to <= 0:           # safety
        clip_to = max_end if abs(max_end % bar_len) < 1e-6 else clip_to
    print(f"Clip to: {clip_to}")
    print(f"bar_len : {bar_len}")

    # 5️⃣  build instruments with clipping
    midi = pretty_midi.PrettyMIDI()
    for name, conf in music.items():
        inst, _ = instrument_from_dict(name, conf, clip_to=clip_to)
        midi.instruments.append(inst)

    midi.write(midi_path)


    if not SOUNDFONT_PATH.exists():
            raise FileNotFoundError(f"CRITICAL: Soundfont not found at: {SOUNDFONT_PATH}")
    
    print("\n--- DEBUGGING SUBPROCESS CALL ---")
    print(f"Soundfont Path: {str(SOUNDFONT_PATH)}")
    print(f"MIDI Input Path: {midi_path}")
    print(f"WAV Output Path: {wav_path}")
    print("---------------------------------\n")

    try:
        cmd = [
            "fluidsynth",
            "-ni",                 # non-interactive (no REPL, no MIDI input)
            "-F", str(wav_path),   # fast-render to this WAV → suppresses live audio
            "-r", "44100",         # sample-rate
            str(SOUNDFONT_PATH),   # first positional = .sf2
            str(midi_path)         # second positional = .mid
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Rendered WAV file: {wav_path}")
    except subprocess.CalledProcessError as e:
        print("--- FLUIDSYNTH ERROR ---")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e

    return wav_path