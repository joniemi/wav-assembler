from pathlib import Path
import unittest
import wav_assembler
import wave
import yaml

HERE = Path(__file__).parent.resolve()


class TestParseYaml(unittest.TestCase):
    def test_read_into_dictionary(self):
        input_string = """
        foo: 1
        2: bar
        True:
            - 4
            - 5
            - 6
        """
        input_path = HERE / "input.yaml"
        with input_path.open("w") as f:
            f.write(input_string)

        expected = {"foo": 1, 2: "bar", True: [4, 5, 6]}
        actual = wav_assembler.parse_yaml(input_path)

        self.assertDictEqual(expected, actual)

    def test_nonexisting_file_raises_an_error(self):
        input_path = HERE / "I_dont_exist.yaml"

        with self.assertRaises(FileNotFoundError):
            wav_assembler.parse_yaml(input_path)

    def test_illegal_yaml_raises_an_error(self):
        input_string = """
            lonely key:
            --
        """
        input_path = HERE / "malformed.yaml"
        with input_path.open("w") as f:
            f.write(input_string)

        with self.assertRaises(yaml.scanner.ScannerError):
            wav_assembler.parse_yaml(input_path)


class TestReadWav(unittest.TestCase):
    def setUp(self):
        file_specs = [
            # filename, sample_rate, num_channels, num_samples
            (HERE / "mono_long.wav", 48_000, 1, 96000),
            (HERE / "64ch_short.wav", 48_000, 64, 1),
        ]

        for path, sample_rate, num_channels, num_samples in file_specs:
            write_wav(path, sample_rate, num_channels, num_samples)

        self.wav_paths = [name for name, _, _ in file_specs]
        self.sample_rate, self.data = wav_assembler.read_wav_files(self.wav_paths)

    def test_different_sample_rates_raise_an_error(self):
        wav_path = HERE / "sample_rate_32k.wav"
        write_wav(wav_path, 32_000, 2, 3)
        self.wav_paths.append(wav_path)

        with self.assertRaises(AssertionError):
            wav_assembler.read_wav_files(self.wav_paths)

    def test_num_keys_equals_num_files(self):
        self.assertEqual(2, len(self.data))

    def test_data_matches_filename(self):
        # Todo: dict keys should maybe be  strings instead?
        self.assertEqual(2, self.data[""])

    def test_num_samples_matches_file(self):
        pass

    def test_num_channels_matches_file(self):
        pass

    def test_datatype_always_float64(self):
        pass


def write_wav(wav_path, sample_rate, num_channels, num_samples):
    bytes_per_sample = 2  # 16 bit PCM

    with wave.open(str(wav_path), "wb") as f:
        num_bytes = num_samples * bytes_per_sample * num_channels
        f.setframerate(sample_rate)
        f.setsampwidth(bytes_per_sample)
        f.setnchannels(num_channels)
        f.setnframes(num_samples)
        f.writeframes(bytes(num_bytes))
