Wav Assembler

Wav Assembler cuts audio track segments from input WAV file(s) and
assembles them into an output WAV file.

Wav Assembler is configured using a YAML file. A YAML file must contain
a list of entries, each of which describes an output segment containing
  - the length of the segment in samples
  - a list of input tracks. Each input track contains
    * an input WAV file
    * a timestamp to the beginning of the track in the source file
    * a channel index in the source file
    * a channel index in the output file

The largest output channel index defines the number of channels in the
output WAV file.

In order to avoid audible artifacts when concatenating consecutive
segments, Wav Assembler applies a short fade out and fade in to the
segments.
