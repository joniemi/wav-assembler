from setuptools import setup

setup(
    name="wav-assembler",
    version="0.1",
    description="A command line tool that cuts audio track segments from input"
                " WAV file(s) and assembles them into an output WAV file.",
    url="",
    author="Jussi Nieminen",
    author_email="j.n.8.9@hotmail.com",
    license="MIT",
    keywords="wav, audio",
    classifiers=[
        "Intended Audience :: Developers",
        "Operating System :: Microsoft",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "platforms = unix, linux, osx, win32"
    ],
    entry_points={
        'console_scripts': [
            'wav_assembler = wav_assembler:main',
        ]
    },
    install_requires=["pyyaml", "scipy"],
    extras_require={
        'doc': [
            "myst-parser",
            "sphinx",
            "sphinx-autorun",
            "sphinx-rtd-theme",
        ],
    },
    python_requires=">=3.7",
)
