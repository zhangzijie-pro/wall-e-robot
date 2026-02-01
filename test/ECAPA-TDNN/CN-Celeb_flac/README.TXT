Introduction
============

CN-Celeb, a large-scale speaker recognition dataset collected `in the wild'.
The dataset contains more than 126,000 utterances from 997 Chinese
celebrities, and covers 11 different genres in real world.
All the audio files are coded as singlechannel and sampled at 16kHz with 16-bit precision.

The dataset is described in more detail in the paper:

```
@inproceedings{fan2020cn,
  title={CN-CELEB: a challenging Chinese speaker recognition dataset},
  author={Fan, Yue and Kang, JW and Li, LT and Li, KC and Chen, HL and 
          Cheng, ST and Zhang, PY and Zhou, ZY and Cai, YQ and Wang, Dong},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7604--7608},
  year={2020},
  organization={IEEE}
}
```

The layout of this data pack is the following:

  ``data``
    - CN-Celeb data directory, contains 997 speakers in total.
      ``id***``
        - speaker identity
           ``*.wav``
             - audio file
             - format: <genre>-<session_id>-<utter_id>.wav

  ``dev``
    - A subset of CN-Celeb dataset, which contains 196 speakers.
    - It is intended to be used to develop and tune speaker recognition systems.

  ``eval``
    - A subset of CN-Celeb dataset, which contains 200 speakers.
    - There is no overlap between ``dev`` and ``eval``.
    - It is a standard evaluation set for speaker recognition systems.

  ``README.TXT``
    - this file


Data Profile
============

Data distribution over the genres are as follows:

    ===============  ===========  =============  ============
     Genres           # of Spks    # of Utters    # of Hours
    ===============  ===========  =============  ============
     Advertisement         17            121          0.21
     Drama                132          4,994          4.55
     Entertainment        487         22,137         33.88
     Interview            786         59,556        136.33
     Live Broadcast       129          8,747         16.36
     Movie                 51          1,127          1.16
     Play                  69          4,244          4.96
     Recitation            41          2,747          5.00
     Singing              319         12,561         28.86
     Speech               123          8,404         36.24
     Vlog                  41          1,894          4.17
    ===============  ===========  =============  ============
     Overall              997        126,532        271.72
    ===============  ===========  =============  ============


Local Download (not recommended)
============
The data can be downloaded from our local server at CSLT@Tsinghua.
- wav.tgz: http://cslt.riit.tsinghua.edu.cn/~data/CN-Celeb/wav.tgz


Public Download (recommended)
============
The above links are from our own web server at Tsinghua University, 
which may be not stable and slow for some connections. 

The mirrors in the public OpenSLR cloud can be used:
- OpenSLR: https://openslr.org/82/ or https://openslr.magicdatatech.com/82/ (mirror in China) 


License
============
- CN-Celeb is freely available for research institutes and individuals. 
- No commercial usage is permitted.


Version
============
- v1.0 -> v2.0
- ChangeLog: (1) modify error genre labels, e.g., from `id00021/drama-01-026.wav` to `id00021/entertainment-20-001.wav`.
             (2) remove error data, including error speakers, dubbing in movie/drama, silence, etc.
- Time: updated May 30, 2021.


Authors
============
- Dong Wang
- Yue Fan
- Jiawen Kang
- Lantian Li
- Kaicheng Li
- Haolin Chen
- Sitong Cheng
- Pengyuan Zhang
- Ziya Zhou
- Yunqi Cai


Contactor
============
- wangdong99@mails.tsinghua.edu.cn
- lilt@cslt.org
- fanyue@cslt.org
- kangjw@cslt.org

- CSLT, Tsinghua University
- ROOM1-303, BLDG FIT
- Tsinghua University


Homepage
============
- http://project.cslt.org
