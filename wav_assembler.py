"""
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

"""

import argparse
import numpy
import pathlib
import scipy.io.wavfile as wavfile
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "yaml_file", metavar="YAML_FILE", type=str,
        help="YAML file that describes the input audio segments to cut",
    )
    parser.add_argument(
        "-o", "--output-file", default="out.wav", metavar="STR",
        help="Path to the output WAV file (default: out.wav)",
    )
    parser.add_argument(
        "-c", "--crossfade", action="store_true", default=False,
        help="Crossfade concatenated channel segments. Otherwise, a fade-in and"
        " a fade-out is applied.",
    )
    parser.add_argument(
        "-d", "--fade-duration", default=128, metavar="INT", type=int,
        help="The total crossfade or fade-in and fade-out duration in samples."
        " (default: 128)",
    )
    parser.add_argument(
        "-b", "--output-bit-depth", default=16, choices={16, 32}, type=int,
        help="The bit depth of the output WAV file. Currently, WAV assembler"
             " only supports 16 and 32 bit integer.",
    )

    args = parser.parse_args()
    args.yaml_file = pathlib.Path(args.yaml_file)
    args.output_file = pathlib.Path(args.output_file)

    return args


def parse_yaml(filepath):
    """ Opens a YAML file and reads the data into a dictionary.

    :param pathlib.Path filepath:
    :return dict yaml_data:
    """
    with filepath.open(mode='r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    return yaml_data


def read_wav_files(filenames):
    sample_rates, data = list(zip(*[wavfile.read(wav_file) for wav_file in filenames]))
    assert len(set(sample_rates)) == 1, "WAV files have different sample rates"

    # Convert mono files to 2-dimensional arrays, so that indexing works consistently.
    data = [d.reshape((d.size, 1)) if d.ndim == 1 else d for d in data]

    # Convert to 64-bit float.
    data_as_float64 = [d.astype(numpy.float64)for d in data]
    max_values = [numpy.iinfo(d.dtype).max for d in data]
    data_as_float64 = [
        d / max_value for d, max_value in zip(data_as_float64, max_values)
    ]

    return sample_rates[0], dict(zip(filenames, data_as_float64))


def init_output_data(yaml_data):
    largest_channel_per_segment = list()

    for segment in yaml_data:
        largest_channel_per_segment.append(
            max(
                mono_clip["channel"] for mono_clip in segment["mono_clips"]
            )
        )

    num_output_channels = max(largest_channel_per_segment) + 1
    num_output_samples = sum(segment["length"] for segment in yaml_data)

    return numpy.zeros((num_output_samples, num_output_channels), dtype=numpy.float64)


def assemble_segments(yaml_data, input_data, output_data):
    sample_pos = 0

    for segment in yaml_data:
        num_samples_to_copy = segment["length"]

        for mono_clip in segment["mono_clips"]:
            source_filename = mono_clip["wav_file"]
            source_data = input_data[source_filename]
            in_start = mono_clip["timestamp"]
            in_end = in_start + num_samples_to_copy
            out_start, out_end = sample_pos, sample_pos + num_samples_to_copy
            output_data[out_start:out_end, mono_clip["channel"]] \
                = source_data[in_start:in_end, mono_clip["source_channel"]]

        sample_pos += num_samples_to_copy


def cosine_ramp(num_samples, ascending=True):
    amplitude = -1 if ascending else 1
    return numpy.array([
        (amplitude * numpy.cos(2 * numpy.pi * x) + 1) / 2
        for x in numpy.linspace(0, 0.5, num=num_samples)
    ])


def apply_fading(samples, clip_boundaries, fade_length):
    ramp_length = int(fade_length // 2)
    fade_in = cosine_ramp(ramp_length, ascending=True)
    fade_out = cosine_ramp(ramp_length, ascending=False)
    num_channels = samples.shape[1]

    for start, end in zip(clip_boundaries[:-1], clip_boundaries[1:]):
        for ch in range(num_channels):
            samples[start:start + ramp_length, ch] *= fade_in
            samples[end - ramp_length:end, ch] *= fade_out


def apply_xfading(samples, clip_boundaries, fade_length):
    apply_fading(samples, clip_boundaries, fade_length * 2)

    num_samples, num_channels = samples.shape
    num_xfades = len(clip_boundaries) - 2  # Exclude start and end.
    new_num_samples = num_samples - num_xfades * fade_length
    _samples = numpy.zeros((new_num_samples, num_channels), dtype=numpy.float64)

    new_clip_start_indices = [
        sample_idx - i * fade_length for i, sample_idx in enumerate(clip_boundaries)
    ]

    for start, end, new_start in zip(
            clip_boundaries[:-1],
            clip_boundaries[1:],
            new_clip_start_indices
    ):
        new_end = new_start + (end-start)
        _samples[new_start:new_end, :] += samples[start:end, :]

    return _samples


def run(yaml_filepath, output_filepath, output_bit_depth=16, xfade=False, fade_duration=128):
    assert output_bit_depth in (16, 32), "Only 16 and 32 bit WAV output supported."

    yaml_data = parse_yaml(yaml_filepath)

    all_wav_files = set(
        mono_clip["wav_file"] for segment in yaml_data for mono_clip in
        segment["mono_clips"]
    )

    sample_rate, input_data = read_wav_files(all_wav_files)
    output_data = init_output_data(yaml_data)
    assemble_segments(yaml_data, input_data, output_data)

    # Clip boundaries contains the start timestamps of each audio segment
    # and the end timestamp in samples.
    clip_boundaries = numpy.cumsum([0] + [segment["length"] for segment in yaml_data])

    if xfade:
        output_data = apply_xfading(output_data, clip_boundaries, fade_duration)
    else:
        apply_fading(output_data, clip_boundaries, fade_duration)

    # Convert to 16 bit integer.
    output_dtype = {16: numpy.int16, 32: numpy.int32}[output_bit_depth]
    output_data *= numpy.iinfo(output_dtype).max
    output_data = output_data.astype(output_dtype)

    wavfile.write(output_filepath, sample_rate, output_data)

    print(f"Switches at {[int(x / sample_rate * 1000) for x in clip_boundaries[1:-1]]}")


def main():
    args = parse_arguments()
    run(
        args.yaml_file,
        args.output_file,
        output_bit_depth=args.output_bit_depth,
        xfade=args.crossfade,
        fade_duration=args.fade_duration
    )


if __name__ == "__main__":
    main()
