dTect V4.2
Tracking Setup
Mon 19 Dec 2011, 14:08:05
!
Seed Connection mode: 0
Section.0.SectionID: 0
Section.0.Tracker.Use adjuster: Yes
Section.0.Tracker.Seed only propagation: No
Section.0.Adjuster.Threshold value: 0.5
Section.0.Adjuster.Remove on Failure: Yes
Section.0.Tracker.Track event: Peak (Max)
Section.0.Tracker.Permitted range: -0.012`0.012
Section.0.Tracker.Value threshhold: 1e30
Section.0.Tracker.Allowed variance: 0.2
Section.0.Tracker.Allowed variances: 0.01`0.02`0.05`0.1`0.2
Section.0.Tracker.Use abs threshhold: No
Section.0.Tracker.Similarity window: -0.04`0.04
Section.0.Tracker.Similarity threshhold: 0.8
Section.0.Tracker.Track by value: Yes
Section.0.Tracker.Normalize similarity: No
Attribs.0.Definition: Storage id=100010.4 output=0
Attribs.0.UserRef: 4 Dip steered median filter
Attribs.0.Hidden: No
Attribs.1.Definition: Storage id=100010.9 output=0
Attribs.1.UserRef: 3 Steering BG Background
Attribs.1.Hidden: Yes
Attribs.2.Definition: Storage id=100010.9 output=1
Attribs.2.UserRef: 3 Steering BG Background
Attribs.2.Hidden: Yes
Attribs.3.Definition: FullSteering phlock=no output=0
Attribs.3.UserRef: FullSteering 100010.y
Attribs.3.Hidden: Yes
Attribs.3.Input.0: 1
Attribs.3.Input.1: 2
Attribs.4.Definition: Similarity gate=[-40,40] pos0=0,1 pos1=0,-1 extension=90 steering=yes normalize=yes output=4
Attribs.4.UserRef: Similarity [-40,40] Parallel
Attribs.4.Hidden: No
Attribs.4.Input.0: 0
Attribs.4.Input.1: 3
Attribs.5.Definition: Similarity gate=[-40,40] pos0=1,1 pos1=-1,-1 extension=90 steering=yes normalize=yes output=4
Attribs.5.UserRef: Similarity [-40,40] Diagonal
Attribs.5.Hidden: No
Attribs.5.Input.0: 0
Attribs.5.Input.1: 3
Attribs.6.Definition: Shift pos=0,0 time=-100 steering=no output=0
Attribs.6.UserRef: Down 100ms {Similarity [-40,40] Parallel}
Attribs.6.Hidden: No
Attribs.6.Input.0: 4
Attribs.7.Definition: Shift pos=0,0 time=100 steering=no output=0
Attribs.7.UserRef: Up 100ms {Similarity [-40,40] Parallel}
Attribs.7.Hidden: No
Attribs.7.Input.0: 4
Attribs.7.Input.1: 3
Attribs.8.Definition: Shift pos=0,0 time=-100 steering=no output=0
Attribs.8.UserRef: Down 100ms {Similarity [-40,40] Diagonal}
Attribs.8.Hidden: No
Attribs.8.Input.0: 5
Attribs.8.Input.1: 3
Attribs.9.Definition: Shift pos=0,0 time=100 steering=no output=0
Attribs.9.UserRef: Up 100ms {Similarity [-40,40] Diagonal}
Attribs.9.Hidden: No
Attribs.9.Input.0: 5
Attribs.10.Definition: Math expression=max(x0,x1) output=0
Attribs.10.UserRef: Similarity [-40,40] AllDir
Attribs.10.Hidden: No
Attribs.10.Input.0: 4
Attribs.10.Input.1: 5
Attribs.11.Definition: Shift pos=0,0 time=-100 steering=no output=0
Attribs.11.UserRef: Down 100ms {Similarity [-40,40] AllDir}
Attribs.11.Hidden: No
Attribs.11.Input.0: 10
Attribs.11.Input.1: 3
Attribs.12.Definition: Shift pos=0,0 time=100 steering=no output=0
Attribs.12.UserRef: Up 100ms {Similarity [-40,40] AllDir}
Attribs.12.Hidden: No
Attribs.12.Input.0: 10
Attribs.13.Definition: Math expression=med(x0,x1,x2) output=0
Attribs.13.UserRef: Simple Chimney Attribute
Attribs.13.Hidden: No
Attribs.13.Input.0: 10
Attribs.13.Input.1: 11
Attribs.13.Input.2: 12
Attribs.14.Definition: PolarDip inlcrl2dipazi=yes azidir=0 output=0
Attribs.14.UserRef: NoNNInput PolarDip
Attribs.14.Hidden: No
Attribs.14.Input.0: 1
Attribs.14.Input.1: 2
Attribs.15.Definition: Energy gate=[-40,40] dograd=no output=1
Attribs.15.UserRef: RMS [-40,40]
Attribs.15.Hidden: No
Attribs.15.Input.0: 0
Attribs.16.Definition: Shift pos=0,0 time=-100 steering=no output=0
Attribs.16.UserRef: Down 100ms {RMS [-40,40]}
Attribs.16.Hidden: No
Attribs.16.Input.0: 15
Attribs.17.Definition: Shift pos=0,0 time=100 steering=no output=0
Attribs.17.UserRef: Up 100ms {RMS [-40,40]}
Attribs.17.Hidden: No
Attribs.17.Input.0: 15
Attribs.18.Definition: Math expression=x0-x1 output=0
Attribs.18.UserRef: NoNNInput FilterResidual
Attribs.18.Hidden: No
Attribs.18.Input.0: 0
Attribs.18.Input.1: 19
Attribs.19.Definition: VolumeStatistics stepout=3,3 shape=Ellipse gate=[0,0] allowedgeeffects=no nrtrcs=1 steering=yes output=1
Attribs.19.UserRef: NoNNInput Median Dip Filter Stepout=3
Attribs.19.Hidden: No
Attribs.19.Input.0: 0
Attribs.19.Input.1: 3
Attribs.20.Definition: Energy gate=[-40,40] dograd=no output=1
Attribs.20.UserRef: Noise {RMS [-40,40] FilterResidual}
Attribs.20.Hidden: No
Attribs.20.Input.0: 18
Attribs.21.Definition: Shift pos=0,0 time=-100 steering=no output=0
Attribs.21.UserRef: Down 100ms {Noise}
Attribs.21.Hidden: No
Attribs.21.Input.0: 20
Attribs.22.Definition: Shift pos=0,0 time=100 steering=no output=0
Attribs.22.UserRef: Up 100ms {Noise}
Attribs.22.Hidden: No
Attribs.22.Input.0: 20
Attribs.23.Definition: Energy gate=[-40,40] dograd=no output=1
Attribs.23.UserRef: NoNNInput Signal {RMS {-40,40] Median Dip Filter}
Attribs.23.Hidden: No
Attribs.23.Input.0: 19
Attribs.24.Definition: Math expression=x0/x1 output=0
Attribs.24.UserRef: Signal/Noise
Attribs.24.Hidden: No
Attribs.24.Input.0: 23
Attribs.24.Input.1: 20
Attribs.25.Definition: Shift pos=0,0 time=-100 steering=no output=0
Attribs.25.UserRef: Down 100ms {Signal/Noise}
Attribs.25.Hidden: No
Attribs.25.Input.0: 24
Attribs.26.Definition: Shift pos=0,0 time=100 steering=no output=0
Attribs.26.UserRef: Up 100ms {Signal/Noise}
Attribs.26.Hidden: No
Attribs.26.Input.0: 24
Attribs.27.Definition: Reference output=2
Attribs.27.UserRef: TWT
Attribs.27.Hidden: No
Attribs.27.Input.0: 0
Attribs.28.Definition: Hilbert halflen=30 output=0
Attribs.28.UserRef: _NoNNInput Median Dip Filter Stepout=3_imag
Attribs.28.Hidden: Yes
Attribs.28.Input.0: 19
Attribs.29.Definition: Frequency gate=[-140,140] normalize=no window=Hanning dumptofile=no output=1
Attribs.29.UserRef: Average Freq [-140,140]
Attribs.29.Hidden: No
Attribs.29.Input.0: 19
Attribs.29.Input.1: 28
Attribs.30.Definition: Math expression=(1+c0*x0)*x1 constant0=0.333  output=0
Attribs.30.UserRef: Av Freq [-140,140] with SimpleTimeCorrection
Attribs.30.Hidden: No
Attribs.30.Input.0: 27
Attribs.30.Input.1: 29
Attribs.31.Definition: VolumeStatistics stepout=1,1 shape=Rectangle gate=[-12,12] allowedgeeffects=no nrtrcs=1 steering=no output=1
Attribs.31.UserRef: MedianFilter PolarDip
Attribs.31.Hidden: No
Attribs.31.Input.0: 14
Attribs.32.Definition: FreqFilter type=LowPass maxfreq=12 nrpoles=4 isfftfilter=no window=CosTaper fwindow=CosTaper paramval=0.95 isfreqtaper=yes highfreqparamval=10 lowfreqparamval=55 output=0
Attribs.32.UserRef: No NNInput LowPass12Hz
Attribs.32.Hidden: No
Attribs.32.Input.0: 19
Attribs.32.Input.1: 28
Attribs.33.Definition: Energy gate=[-140,140] dograd=no output=1
Attribs.33.UserRef: No NNInput RMS [-140,140] {LowPass12Hz}
Attribs.33.Hidden: No
Attribs.33.Input.0: 32
Attribs.34.Definition: FreqFilter type=HighPass minfreq=36 nrpoles=4 isfftfilter=no window=CosTaper fwindow=CosTaper paramval=0.95 isfreqtaper=yes highfreqparamval=10 lowfreqparamval=55 output=0
Attribs.34.UserRef: No NNInput HighPass36Hz
Attribs.34.Hidden: No
Attribs.34.Input.0: 19
Attribs.34.Input.1: 28
Attribs.35.Definition: Energy gate=[-140,140] dograd=no output=1
Attribs.35.UserRef: No NNInput RMS [-140,140] {HighPass36Hz}
Attribs.35.Hidden: No
Attribs.35.Input.0: 34
Attribs.36.Definition: Math expression=x0/x1 output=0
Attribs.36.UserRef: Frequency Wash-out Ratio {RMSLowPass/RMSHighPass]
Attribs.36.Hidden: No
Attribs.36.Input.0: 33
Attribs.36.Input.1: 35
Attribs.37.Definition: VolumeStatistics stepout=1,1 shape=Rectangle gate=[-40,40] allowedgeeffects=no nrtrcs=1 steering=no output=2
Attribs.37.UserRef: Variance [-40,40] PolarDip
Attribs.37.Hidden: No
Attribs.37.Input.0: 14
Attribs.38.Definition: Math expression=x1/(1+c0*x0) constant0=1.333  output=0
Attribs.38.UserRef: Frequency Wash-out Ratio With SimpleTimeCorrection
Attribs.38.Hidden: No
Attribs.38.Input.0: 36
Attribs.38.Input.1: 27
Attribs.39.Definition: Shift pos=0,0 time=-100 steering=no output=0
Attribs.39.UserRef: Down 100ms {Variance [-40,40] PolarDip}
Attribs.39.Hidden: No
Attribs.39.Input.0: 37
Attribs.40.Definition: Shift pos=0,0 time=100 steering=no output=0
Attribs.40.UserRef: Up 100ms {Variance [-40,40] PolarDip}
Attribs.40.Hidden: No
Attribs.40.Input.0: 37
Attribs.41.Definition: NN specification=100060.8 output=0
Attribs.41.UserRef: 100030.9
Attribs.41.Hidden: Yes
Attribs.41.Input.0: 4
Attribs.41.Input.1: 5
Attribs.41.Input.2: 6
Attribs.41.Input.3: 7
Attribs.41.Input.4: 8
Attribs.41.Input.5: 9
Attribs.41.Input.6: 10
Attribs.41.Input.7: 11
Attribs.41.Input.8: 12
Attribs.41.Input.9: 13
Attribs.41.Input.10: 15
Attribs.41.Input.11: 16
Attribs.41.Input.12: 17
Attribs.41.Input.13: 20
Attribs.41.Input.14: 21
Attribs.41.Input.15: 22
Attribs.41.Input.16: 24
Attribs.41.Input.17: 25
Attribs.41.Input.18: 26
Attribs.41.Input.19: 27
Attribs.41.Input.20: 29
Attribs.41.Input.21: 30
Attribs.41.Input.22: 31
Attribs.41.Input.23: 36
Attribs.41.Input.24: 37
Attribs.41.Input.25: 38
Attribs.41.Input.26: 39
Attribs.41.Input.27: 40
Attribs.MaxNrKeys: 41
Attribs.Type: 3D
!
